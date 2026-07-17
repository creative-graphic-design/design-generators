from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[3]


def _load_script() -> ModuleType:
    path = ROOT / "models" / "layout-dm" / "scripts" / "download_original.py"
    spec = importlib.util.spec_from_file_location("layout_dm_download_original", path)
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


class _Response:
    def __init__(self, archive: Path) -> None:
        self.archive = archive

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        _ = chunk_size
        yield self.archive.read_bytes()


def test_download_replaces_partial_archive_and_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    source_archive = tmp_path / "source.zip"
    _write_archive(source_archive, module.REQUIRED_PATHS)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    (output_dir / "layoutdm_starter.zip").write_bytes(b"partial")
    stale_dir = output_dir / module.STARTER_DIR
    stale_dir.mkdir()

    calls = []

    def fake_get(url: str, *, stream: bool, timeout: int) -> _Response:
        calls.append((url, stream, timeout))
        return _Response(source_archive)

    monkeypatch.setattr(module.requests, "get", fake_get)

    starter_dir = module.ensure_downloaded(
        output_dir, url="https://example.invalid/x.zip"
    )

    assert starter_dir == output_dir / module.STARTER_DIR
    assert calls == [("https://example.invalid/x.zip", True, 60)]
    assert all((output_dir / path).is_file() for path in module.REQUIRED_PATHS)


def test_download_reextracts_complete_archive_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    output_dir = tmp_path / "cache"
    archive = output_dir / "layoutdm_starter.zip"
    _write_archive(archive, module.REQUIRED_PATHS)

    def fail_get(*args: object, **kwargs: object) -> object:
        raise AssertionError("network should not be used for a complete archive")

    monkeypatch.setattr(module.requests, "get", fail_get)

    starter_dir = module.ensure_downloaded(output_dir)

    assert starter_dir == output_dir / module.STARTER_DIR
    assert all((output_dir / path).is_file() for path in module.REQUIRED_PATHS)
