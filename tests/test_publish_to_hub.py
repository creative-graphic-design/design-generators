from pathlib import Path
import sys
from types import SimpleNamespace
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
publisher = SimpleNamespace(**run_path(str(ROOT / "scripts" / "publish_to_hub.py")))

REVISION = "a" * 40


@pytest.mark.parametrize(
    ("package", "filenames", "archive_count"),
    [
        ("collage-diffusion", "README.md pipeline.py", 2),
        (
            "structure-diffusion",
            "README.md pipeline.py modeling_structure_diffusion.py processing_structure_diffusion.py",
            1,
        ),
        ("tkg-dm", "README.md pipeline.py pipeline_tkg_dm.py", 1),
        ("sawna", "README.md pipeline.py", 2),
    ],
)
def test_declared_files_pin_archive_links_and_keep_generated_entrypoints(
    package: str, filenames: str, archive_count: int
) -> None:
    package_directory = ROOT / "models" / package
    config = publisher.load_config(package_directory)
    files = publisher.prepare_upload_files(package_directory, config, REVISION)

    assert set(files) == set(filenames.split())
    assert (
        files["README.md"].count(f"/archive/{REVISION}.zip".encode()) == archive_count
    )
    if package == "sawna":
        assert files["pipeline.py"] == b"from sawna import SAWNAHubPipeline\n"


def test_existing_files_ignore_only_archive_revision_changes() -> None:
    assert publisher.files_match_published_revision(
        {"README.md"},
        {"README.md": b"/archive/" + REVISION.encode() + b".zip"},
        read_file=lambda _: b"/archive/" + b"b" * 40 + b".zip",
    )


def test_collage_preserves_source_asset_file_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_directory = ROOT / "models" / "collage-diffusion"
    config = publisher.load_config(package_directory)
    repo_files = iter([["source-assets/a"], ["source-assets/a", "source-assets/b"]])
    api = SimpleNamespace(
        auth_check=lambda **_: None,
        model_info=lambda *_args, **_: SimpleNamespace(sha=REVISION),
        list_repo_files=lambda *_args, **_: next(repo_files),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda **_: api, hf_hub_download=None),
    )

    monkeypatch.setattr(
        publisher.hub_publishing,
        "upload_and_verify",
        lambda *_args, **_: REVISION,
    )
    with pytest.raises(RuntimeError, match="changed the existing source-assets"):
        publisher.publish(
            config=config,
            files={"pipeline.py": b"content"},
            token="test-token",
            revision=REVISION,
        )
