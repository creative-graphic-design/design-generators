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
LOADER = SourceFileLoader("structure_diffusion_publish_to_hub", str(PUBLISHER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = PUBLISHER
LOADER.exec_module(PUBLISHER)

REVISION = "a" * 40
REPOSITORY_URL = "https://github.com/creative-graphic-design/design-generators"
ARCHIVE_URL = f"{REPOSITORY_URL}/archive"


def prepare_source_files() -> dict[str, bytes]:
    return {
        "pipeline.py": PUBLISHER.PIPELINE_SOURCE.read_bytes(),
        "modeling_structure_diffusion.py": PUBLISHER.MODELING_SOURCE.read_bytes(),
        "processing_structure_diffusion.py": PUBLISHER.PROCESSING_SOURCE.read_bytes(),
    }


def test_prepares_only_the_pipeline_dependencies_and_card_from_canonical_sources() -> (
    None
):
    source_files = prepare_source_files()
    files = PUBLISHER.prepare_upload_files(
        pipeline_source=source_files["pipeline.py"],
        modeling_source=source_files["modeling_structure_diffusion.py"],
        processing_source=source_files["processing_structure_diffusion.py"],
        readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
        revision=REVISION,
    )

    assert set(files) == {
        "README.md",
        "pipeline.py",
        "modeling_structure_diffusion.py",
        "processing_structure_diffusion.py",
    }
    assert files["pipeline.py"] == source_files["pipeline.py"]
    assert (
        files["modeling_structure_diffusion.py"]
        == source_files["modeling_structure_diffusion.py"]
    )
    assert (
        files["processing_structure_diffusion.py"]
        == source_files["processing_structure_diffusion.py"]
    )


def test_model_card_pins_archive_and_documents_custom_pipeline_prerequisites() -> None:
    files = PUBLISHER.prepare_upload_files(
        pipeline_source=b"pipeline",
        modeling_source=b"modeling",
        processing_source=b"processing",
        readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
        revision=REVISION,
    )
    card = files["README.md"].decode("utf-8")
    source_readme = PUBLISHER.README_SOURCE.read_text(encoding="utf-8")
    archive_pattern = re.compile(r'"structure-diffusion @ (?P<url>https:[^"\s]+)"')

    source_urls = archive_pattern.findall(source_readme)
    card_urls = archive_pattern.findall(card)

    assert source_urls == [
        f"{ARCHIVE_URL}/main.zip#subdirectory=models/structure-diffusion"
    ]
    assert card_urls == [
        f"{ARCHIVE_URL}/{REVISION}.zip#subdirectory=models/structure-diffusion"
    ]
    assert "message=published-code-only&color=brightgreen" in card
    assert "planned-code-only" not in card
    assert 'custom_pipeline="creative-graphic-design/structure-diffusion"' in card
    assert "trust_remote_code=True" in card
    assert "gated" in card.lower()
    assert "CompVis/stable-diffusion-v1-4" in card
    assert "stanza.download('en', processors='tokenize,pos,constituency')" in card


def test_card_generation_requires_a_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character commit SHA"):
        PUBLISHER.prepare_upload_files(
            pipeline_source=b"pipeline",
            modeling_source=b"modeling",
            processing_source=b"processing",
            readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
            revision="main",
        )


def test_publish_calls_hub_readback_callback(
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
    hub_upload_arguments: dict[str, object] = {}

    class ScopedTokenApi:
        def __init__(self, *, token: str) -> None:
            assert token == "fine-grained-token"

        def auth_check(self, **kwargs: str | bool) -> None:
            assert kwargs == {
                "repo_id": PUBLISHER.HUB_REPOSITORY_ID,
                "repo_type": "model",
                "write": True,
            }

        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == PUBLISHER.HUB_REPOSITORY_ID
            assert revision == "main"
            return SimpleNamespace(sha=parent_commit)

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            hub_upload_arguments.update(kwargs)
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

    def upload_and_verify(
        api: object,
        *,
        repo_id: str,
        files: dict[str, bytes],
        allow_patterns: tuple[str, ...],
        parent_commit: str,
        commit_message: str,
        staging_prefix: str,
        upload_folder: Callable[
            [PUBLISHER.hub_publishing.HubUploadRequest],
            PUBLISHER.hub_publishing.HubCommitInfo,
        ],
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
                "readback_bytes": read_file("README.md", uploaded_commit),
            }
        )
        upload_folder(
            PUBLISHER.hub_publishing.HubUploadRequest(
                repo_id=repo_id,
                folder_path=str(tmp_path),
                allow_patterns=allow_patterns,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
        )
        return uploaded_commit

    monkeypatch.setattr(
        PUBLISHER.hub_publishing,
        "upload_and_verify",
        upload_and_verify,
    )

    PUBLISHER.publish(token="fine-grained-token", revision=REVISION)

    assert upload_arguments["repo_id"] == PUBLISHER.HUB_REPOSITORY_ID
    assert upload_arguments["parent_commit"] == parent_commit
    assert upload_arguments["readback_bytes"] == readback_bytes
    assert hub_upload_arguments == {
        "repo_id": PUBLISHER.HUB_REPOSITORY_ID,
        "repo_type": "model",
        "folder_path": str(tmp_path),
        "allow_patterns": list(PUBLISHER.UPLOAD_ALLOW_PATTERNS),
        "commit_message": "Publish Structure Diffusion pipeline at aaaaaaaaaaaa",
        "parent_commit": parent_commit,
        "run_as_future": False,
    }
    assert download_arguments == [
        {
            "repo_id": PUBLISHER.HUB_REPOSITORY_ID,
            "filename": "README.md",
            "repo_type": "model",
            "revision": uploaded_commit,
            "token": "fine-grained-token",
        }
    ]


def test_dry_run_lists_exact_hub_files_without_publishing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PUBLISHER.hub_publishing,
        "verify_source_revision",
        lambda repository_root, revision: None,
    )
    monkeypatch.setattr(
        PUBLISHER,
        "publish",
        lambda **kwargs: pytest.fail("dry-run must not publish"),
    )

    assert PUBLISHER.main(["--revision", REVISION, "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "README.md:" in output
    assert "pipeline.py:" in output
    assert "modeling_structure_diffusion.py:" in output
    assert "processing_structure_diffusion.py:" in output
    assert "Hub write: skipped (--dry-run)" in output
