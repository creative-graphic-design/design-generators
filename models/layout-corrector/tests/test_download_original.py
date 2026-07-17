from __future__ import annotations

import importlib.util
import shutil
import zipfile
from pathlib import Path
from typing import Final
from types import ModuleType
from types import SimpleNamespace


ROOT: Final[Path] = Path(__file__).parents[3]


def _load_script() -> ModuleType:
    path = ROOT / "models" / "layout-corrector" / "scripts" / "download_original.py"
    spec = importlib.util.spec_from_file_location(
        "layout_corrector_download_original", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, required_paths: tuple[Path, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zip_file:
        for required_path in required_paths:
            zip_file.writestr(required_path.as_posix(), "ok")


def test_download_replaces_partial_archive_and_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    source_archive = tmp_path / "source.zip"
    _write_archive(source_archive, module.REQUIRED_PATHS)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    (output_dir / "layout_corrector_starter.zip").write_bytes(b"partial")
    stale_dir = output_dir / module.STARTER_DIR
    stale_dir.mkdir()

    calls = []

    def fake_download(*, id: str, output: str, quiet: bool) -> str:
        calls.append((id, quiet))
        shutil.copyfile(source_archive, output)
        return output

    monkeypatch.setattr(module, "gdown", SimpleNamespace(download=fake_download))

    starter_dir = module.ensure_downloaded(output_dir, file_id="file-id")

    assert starter_dir == output_dir / module.STARTER_DIR
    assert calls == [("file-id", False)]
    assert all((output_dir / path).is_file() for path in module.REQUIRED_PATHS)


def test_download_reextracts_complete_archive_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    output_dir = tmp_path / "cache"
    archive = output_dir / "layout_corrector_starter.zip"
    _write_archive(archive, module.REQUIRED_PATHS)

    def fail_download(*args: object, **kwargs: object) -> object:
        raise AssertionError("network should not be used for a complete archive")

    monkeypatch.setattr(module, "gdown", SimpleNamespace(download=fail_download))

    starter_dir = module.ensure_downloaded(output_dir)

    assert starter_dir == output_dir / module.STARTER_DIR
    assert all((output_dir / path).is_file() for path in module.REQUIRED_PATHS)
