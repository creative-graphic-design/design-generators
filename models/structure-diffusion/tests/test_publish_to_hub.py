from __future__ import annotations

import importlib.util
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

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
ARCHIVE_URL = "https://github.com/creative-graphic-design/design-generators/archive"


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
