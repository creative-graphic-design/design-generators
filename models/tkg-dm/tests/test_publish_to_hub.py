from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATH = PACKAGE_ROOT / "scripts" / "publish_to_hub.py"
LOADER = SourceFileLoader("tkg_dm_publish_to_hub", str(PUBLISHER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = PUBLISHER
LOADER.exec_module(PUBLISHER)

REVISION = "a" * 40
REPOSITORY_URL = "https://github.com/creative-graphic-design/design-generators"


def test_prepares_only_the_hub_pipeline_dependencies_and_card() -> None:
    files = PUBLISHER.prepare_upload_files(
        helper_source=PUBLISHER.HELPER_SOURCE.read_bytes(),
        readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
        revision=REVISION,
    )

    assert set(files) == set(PUBLISHER.UPLOAD_ALLOW_PATTERNS)
    assert files["pipeline.py"] == PUBLISHER.HUB_PIPELINE_SOURCE
    assert files["pipeline.py"] == b"from .pipeline_tkg_dm import TKGDMHubPipeline\n"
    assert files["pipeline_tkg_dm.py"] == PUBLISHER.HELPER_SOURCE.read_bytes()

    card = files["README.md"].decode("utf-8")
    assert f"{REPOSITORY_URL}/archive/{REVISION}.zip" in card
    assert "/archive/main.zip" not in card
    assert 'custom_revision=os.environ["TKG_DM_CODE_REVISION"]' in card
    assert 'custom_pipeline="creative-graphic-design/tkg-dm"' in card
    assert "trust_remote_code=True" in card


def test_prepare_upload_files_requires_a_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character commit SHA"):
        PUBLISHER.prepare_upload_files(
            helper_source=b"helper",
            readme_text=PUBLISHER.README_SOURCE.read_text(encoding="utf-8"),
            revision="main",
        )


def test_prepare_upload_files_requires_one_archive_install() -> None:
    readme = PUBLISHER.README_SOURCE.read_text(encoding="utf-8")
    multiple_archive_urls = readme.replace(
        "/archive/main.zip",
        "/archive/main.zip /archive/main.zip",
    )

    with pytest.raises(ValueError, match="exactly once"):
        PUBLISHER.prepare_upload_files(
            helper_source=b"helper",
            readme_text=multiple_archive_urls,
            revision=REVISION,
        )
