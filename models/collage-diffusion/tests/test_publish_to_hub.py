from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

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
    assert card.count(f"/archive/{REVISION}.zip") == 2
    assert "custom_revision" not in card
    assert files["pipeline.py"] == b"canonical pipeline"


def test_extracts_the_laygen_archive_install_from_the_published_card() -> None:
    files = PUBLISHER.prepare_upload_files(b"pipeline", README, REVISION)
    card = files["README.md"].decode("utf-8")

    assert PUBLISHER.extract_laygen_requirement(card) == (
        "laygen @ https://github.com/creative-graphic-design/design-generators/"
        f"archive/{REVISION}.zip#subdirectory=lib/laygen"
    )


def test_staging_contains_only_the_explicit_upload_files(tmp_path: Path) -> None:
    files = {"README.md": b"card", "pipeline.py": b"pipeline"}

    PUBLISHER.write_upload_staging(tmp_path, files)

    assert {path.name for path in tmp_path.iterdir()} == {
        "README.md",
        "pipeline.py",
    }
    assert not (tmp_path / "source-assets").exists()


def test_staging_rejects_unapproved_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only README.md and pipeline.py"):
        PUBLISHER.write_upload_staging(
            tmp_path,
            {
                "README.md": b"card",
                "pipeline.py": b"pipeline",
                "source-assets/data.bin": b"",
            },
        )


def test_staging_rejects_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "source-assets").mkdir()

    with pytest.raises(ValueError, match="must be empty"):
        PUBLISHER.write_upload_staging(
            tmp_path,
            {"README.md": b"card", "pipeline.py": b"pipeline"},
        )


def test_card_generation_requires_a_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character commit SHA"):
        PUBLISHER.prepare_upload_files(b"pipeline", README, "main")


def test_card_generation_requires_exactly_one_frontmatter_license() -> None:
    invalid_readme = README.replace("license: apache-2.0\n", "")

    with pytest.raises(ValueError, match="exactly one frontmatter license"):
        PUBLISHER.prepare_upload_files(b"pipeline", invalid_readme, REVISION)


def test_laygen_requirement_rejects_an_unpinned_archive_url() -> None:
    with pytest.raises(ValueError, match="must pin a full commit SHA"):
        PUBLISHER.extract_laygen_requirement(README)
