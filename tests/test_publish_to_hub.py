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
    pipeline = {
        "sawna": b"from sawna import SAWNAHubPipeline\n",
        "tkg-dm": b"from .pipeline_tkg_dm import TKGDMHubPipeline\n",
    }.get(package)
    if pipeline is not None:
        assert files["pipeline.py"] == pipeline


def test_existing_files_ignore_only_generated_archive_revision_changes() -> None:
    template = b"/archive/" + b"!" * 40 + b".zip\n/archive/" + b"c" * 40 + b".zip"
    expected = template.replace(b"!" * 40, REVISION.encode())
    published = template.replace(b"!" * 40, b"b" * 40)

    assert [
        publisher.files_match_published_revision(
            {"README.md"},
            {"README.md": expected},
            lambda _: readme,
            template,
        )
        for readme in (published, published.replace(b"c" * 40, b"d" * 40))
    ] == [True, False]


def test_collage_preserves_source_asset_file_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = publisher.load_config(ROOT / "models" / "collage-diffusion")
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
        publisher.hub_publishing, "upload_and_verify", lambda *_args, **_: REVISION
    )
    with pytest.raises(RuntimeError, match="changed the existing source-assets"):
        publisher.publish(
            config=config,
            files={"pipeline.py": b"content"},
            token="test-token",
            revision=REVISION,
            comparison_template=b"",
        )
