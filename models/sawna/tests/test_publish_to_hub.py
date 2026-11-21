from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATH = PACKAGE_ROOT / "scripts" / "publish_to_hub.py"
LOADER = SourceFileLoader("sawna_publish_to_hub", str(PUBLISHER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = PUBLISHER
LOADER.exec_module(PUBLISHER)

REVISION = "a" * 40
REPOSITORY_URL = "https://github.com/creative-graphic-design/design-generators"


def test_prepares_only_the_custom_pipeline_entrypoint_and_card() -> None:
    files = PUBLISHER.prepare_upload_files(
        readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
        revision=REVISION,
    )

    assert set(files) == set(PUBLISHER.UPLOAD_ALLOW_PATTERNS)
    assert files["pipeline.py"] == b"from sawna import SAWNAHubPipeline\n"
    card = files["README.md"].decode("utf-8")
    assert card.count(f"{REPOSITORY_URL}/archive/{REVISION}.zip#subdirectory=") == 2
    assert "/archive/main.zip" not in card
    assert 'custom_pipeline="creative-graphic-design/sawna"' in card
    assert 'custom_revision=os.environ["SAWNA_CODE_REVISION"]' in card
    assert "trust_remote_code=True" in card
    assert "fine-grained `HF_TOKEN` must include write access" in card


def test_prepare_upload_files_requires_a_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character commit SHA"):
        PUBLISHER.prepare_upload_files(
            readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
            revision="main",
        )


def test_prepare_upload_files_requires_both_workspace_archive_installs() -> None:
    readme = PUBLISHER.README_SOURCE.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="pin both workspace archive installs"):
        PUBLISHER.prepare_upload_files(
            readme_text=readme.replace("/archive/main.zip", "/archive/pinned.zip", 1),
            revision=REVISION,
        )
