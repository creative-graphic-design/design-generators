"""Download and extract the original LayoutDM starter kit."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests


URL = "https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip"
STARTER_DIR = "download"
REQUIRED_PATHS = (
    Path(STARTER_DIR) / "pretrained_weights" / "layoutdm_rico" / "0" / "config.yaml",
    Path(STARTER_DIR)
    / "pretrained_weights"
    / "layoutdm_publaynet"
    / "0"
    / "config.yaml",
)


def _required_paths_exist(output_dir: Path) -> bool:
    return all((output_dir / path).is_file() for path in REQUIRED_PATHS)


def _archive_is_complete(archive: Path) -> bool:
    if not archive.is_file():
        return False
    try:
        with zipfile.ZipFile(archive) as zip_file:
            if zip_file.testzip() is not None:
                return False
            names = set(zip_file.namelist())
    except zipfile.BadZipFile:
        return False
    return all(path.as_posix() in names for path in REQUIRED_PATHS)


def _remove_incomplete_outputs(output_dir: Path, archive: Path) -> None:
    archive.unlink(missing_ok=True)
    (output_dir / f"{archive.name}.tmp").unlink(missing_ok=True)
    shutil.rmtree(output_dir / STARTER_DIR, ignore_errors=True)


def _download_archive(url: str, archive: Path) -> None:
    tmp_archive = archive.with_name(f"{archive.name}.tmp")
    tmp_archive.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_archive.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_archive.replace(archive)


def ensure_downloaded(output_dir: Path, *, url: str = URL) -> Path:
    """Download and extract a complete LayoutDM starter archive.

    Args:
        output_dir: Directory for the archive and extracted starter tree.
        url: Original LayoutDM starter archive URL.

    Returns:
        The extracted starter directory path.

    Raises:
        RuntimeError: If the downloaded archive or extracted tree lacks required
            checkpoint files.

    Examples:
        >>> ensure_downloaded  # doctest: +ELLIPSIS
        <function...
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "layoutdm_starter.zip"
    if not _archive_is_complete(archive):
        _remove_incomplete_outputs(output_dir, archive)
        _download_archive(url, archive)
    if not _archive_is_complete(archive):
        raise RuntimeError(f"Incomplete LayoutDM starter archive: {archive}")
    if not _required_paths_exist(output_dir):
        shutil.rmtree(output_dir / STARTER_DIR, ignore_errors=True)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(output_dir)
    if not _required_paths_exist(output_dir):
        missing = [
            path.as_posix()
            for path in REQUIRED_PATHS
            if not (output_dir / path).is_file()
        ]
        raise RuntimeError(f"Incomplete LayoutDM starter extraction: {missing}")
    return output_dir / STARTER_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the original LayoutDM starter archive and extract it under "
            "the requested cache directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-dm/original"),
        help="Directory for layoutdm_starter.zip and extracted starter files.",
    )
    parser.add_argument(
        "--url",
        default=URL,
        help="Original LayoutDM starter archive URL.",
    )
    args = parser.parse_args()
    print(ensure_downloaded(args.output_dir, url=args.url))


if __name__ == "__main__":
    main()
