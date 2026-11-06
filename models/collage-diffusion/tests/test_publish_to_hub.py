from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATH = PACKAGE_ROOT / "scripts" / "publish_to_hub.py"
LOADER = SourceFileLoader("collage_publish_to_hub", str(PUBLISHER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = PUBLISHER
LOADER.exec_module(PUBLISHER)

REVISION = "a" * 40
README = """---
license: apache-2.0
tags:
  - collage-diffusion
---
![hub](https://img.shields.io/static/v1?label=hub&message=published-code-only&color=brightgreen&style=flat-square&logo=huggingface&logoColor=white)
# Collage Diffusion

```bash
pip install \\
  "laygen @ https://github.com/creative-graphic-design/design-generators/archive/main.zip#subdirectory=lib/laygen" \\
  "collage-diffusion @ https://github.com/creative-graphic-design/design-generators/archive/main.zip#subdirectory=models/collage-diffusion"
```

```python
pipe = DiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1-base",
    custom_pipeline="creative-graphic-design/collage-diffusion",
    trust_remote_code=True,
)
```

Package license: Apache-2.0. The source assets have separate terms.
"""


def test_prepares_only_pipeline_and_model_card_and_pins_the_source_archives() -> None:
    files = PUBLISHER.prepare_upload_files(b"canonical pipeline", README, REVISION)

    assert set(files) == {"README.md", "pipeline.py"}
    card = files["README.md"].decode("utf-8")
    assert "license: apache-2.0" not in card
    assert "Package license: Apache-2.0." in card
    assert "message=published-code-only&color=brightgreen" in card
    assert "planned-code-only" not in card
    assert card.count(f"/archive/{REVISION}.zip") == 2
    assert "custom_revision" not in card
    assert files["pipeline.py"] == b"canonical pipeline"


def test_card_pins_the_laygen_archive_url_to_the_package_revision() -> None:
    readme_text = PUBLISHER.README_SOURCE.read_text(encoding="utf-8")
    files = PUBLISHER.prepare_upload_files(b"pipeline", readme_text, REVISION)
    card = files["README.md"].decode("utf-8")
    url_pattern = re.compile(
        r'(?m)^\s*"laygen @ (?P<url>https:'
        r'//[^"\s]+)"(?:\s*\\)?\s*$'
    )
    readme_urls = url_pattern.findall(readme_text)
    card_urls = url_pattern.findall(card)
    expected_archive = "https:" + (
        "//github.com/creative-graphic-design/design-generators/archive"
    )

    assert len(readme_urls) == 1
    assert readme_urls[0] == f"{expected_archive}/main.zip#subdirectory=lib/laygen"
    assert len(card_urls) == 1
    assert card_urls[0] == (
        f"{expected_archive}/{REVISION}.zip#subdirectory=lib/laygen"
    )


def test_card_generation_requires_a_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character commit SHA"):
        PUBLISHER.prepare_upload_files(b"pipeline", README, "main")


def test_card_generation_requires_exactly_one_frontmatter_license() -> None:
    invalid_readme = README.replace("license: apache-2.0\n", "")

    with pytest.raises(ValueError, match="exactly one frontmatter license"):
        PUBLISHER.prepare_upload_files(b"pipeline", invalid_readme, REVISION)


def test_write_access_failure_names_the_repository() -> None:
    class UnscopedTokenApi:
        def auth_check(self, **kwargs: str | bool) -> None:
            assert kwargs == {
                "repo_id": "creative-graphic-design/collage-diffusion",
                "repo_type": "model",
                "write": True,
            }
            raise RuntimeError("repository is outside token scope")

    with pytest.raises(
        RuntimeError,
        match="Could not verify HF_TOKEN content-write access to "
        "creative-graphic-design/collage-diffusion",
    ):
        PUBLISHER.hub_publishing.verify_hub_write_access(
            UnscopedTokenApi(),
            repo_id=PUBLISHER.HUB_REPOSITORY_ID,
        )


def test_publish_stops_before_upload_when_token_is_not_scoped_for_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnscopedTokenApi:
        def __init__(self, *, token: str) -> None:
            assert token == "fine-grained-token"

        def auth_check(self, **kwargs: str | bool) -> None:
            raise RuntimeError("repository is outside token scope")

        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            pytest.fail("repository metadata must not be read after failed auth check")

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            pytest.fail("upload must not run after failed auth check")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=UnscopedTokenApi),
    )
    monkeypatch.setattr(
        PUBLISHER,
        "_read_source_files",
        lambda revision: {"README.md": b"card", "pipeline.py": b"pipeline"},
    )

    with pytest.raises(
        RuntimeError,
        match="Could not verify HF_TOKEN content-write access to "
        "creative-graphic-design/collage-diffusion",
    ):
        PUBLISHER.publish(token="fine-grained-token", revision=REVISION)


def test_publish_uses_parent_commit_and_calls_hub_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent_commit = "b" * 40
    uploaded_commit = "c" * 40
    readback_bytes = b"read back from the Hub"
    readback_path = tmp_path / "readback.bin"
    readback_path.write_bytes(readback_bytes)
    download_arguments: list[dict[str, str]] = []
    upload_arguments: dict[str, object] = {}
    source_asset_revisions: list[str] = []

    class ScopedTokenApi:
        token = "fine-grained-token"

        def __init__(self, *, token: str) -> None:
            assert token == self.token

        def auth_check(self, **kwargs: str | bool) -> None:
            assert kwargs == {
                "repo_id": "creative-graphic-design/collage-diffusion",
                "repo_type": "model",
                "write": True,
            }

        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == "creative-graphic-design/collage-diffusion"
            assert revision == "main"
            return SimpleNamespace(sha=parent_commit)

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            upload_arguments.update(kwargs)
            return SimpleNamespace(oid=uploaded_commit)

    def hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        token: str,
    ) -> str:
        download_arguments.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
                "token": token,
            }
        )
        return str(readback_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=ScopedTokenApi,
            hf_hub_download=hf_hub_download,
        ),
    )
    monkeypatch.setattr(
        PUBLISHER,
        "_read_source_files",
        lambda revision: {"README.md": b"card", "pipeline.py": b"pipeline"},
    )

    def source_assets(api: object, *, revision: str) -> set[str]:
        source_asset_revisions.append(revision)
        return {"source-assets/original.txt"}

    monkeypatch.setattr(PUBLISHER, "_source_asset_files", source_assets)

    def upload_and_verify(
        api: object,
        *,
        repo_id: str,
        files: dict[str, bytes],
        allow_patterns: tuple[str, ...],
        parent_commit: str,
        commit_message: str,
        staging_prefix: str,
        read_file: Callable[[str, str], bytes],
    ) -> str:
        upload_arguments.update(
            {
                "repo_id": repo_id,
                "files": files,
                "allow_patterns": allow_patterns,
                "parent_commit": parent_commit,
                "commit_message": commit_message,
                "staging_prefix": staging_prefix,
                "read_file": read_file,
            }
        )
        upload_arguments["readback_bytes"] = read_file("README.md", uploaded_commit)
        return uploaded_commit

    monkeypatch.setattr(
        PUBLISHER.hub_publishing,
        "upload_and_verify",
        upload_and_verify,
    )

    PUBLISHER.publish(token="fine-grained-token", revision=REVISION)

    assert upload_arguments["parent_commit"] == parent_commit
    assert upload_arguments["allow_patterns"] == PUBLISHER.UPLOAD_ALLOW_PATTERNS
    assert upload_arguments["readback_bytes"] == readback_bytes
    assert download_arguments == [
        {
            "repo_id": PUBLISHER.HUB_REPOSITORY_ID,
            "filename": "README.md",
            "repo_type": "model",
            "revision": uploaded_commit,
            "token": "fine-grained-token",
        }
    ]
    assert source_asset_revisions == [parent_commit, uploaded_commit]


def test_publish_requires_a_hub_main_commit_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingMainCommitApi:
        def __init__(self, *, token: str) -> None:
            assert token == "fine-grained-token"

        def auth_check(self, **kwargs: str | bool) -> None:
            assert kwargs == {
                "repo_id": "creative-graphic-design/collage-diffusion",
                "repo_type": "model",
                "write": True,
            }

        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == "creative-graphic-design/collage-diffusion"
            assert revision == "main"
            return SimpleNamespace(sha=None)

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            pytest.fail("upload requires a Hub main commit SHA")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=MissingMainCommitApi),
    )
    monkeypatch.setattr(
        PUBLISHER,
        "_read_source_files",
        lambda revision: {"README.md": b"card", "pipeline.py": b"pipeline"},
    )
    monkeypatch.setattr(
        PUBLISHER,
        "_source_asset_files",
        lambda api, *, revision: pytest.fail("source assets need a pinned revision"),
    )

    with pytest.raises(
        RuntimeError,
        match="Hub repository creative-graphic-design/collage-diffusion main revision has no commit SHA",
    ):
        PUBLISHER.publish(token="fine-grained-token", revision=REVISION)
