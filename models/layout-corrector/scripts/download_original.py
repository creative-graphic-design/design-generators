"""Download and extract the original Layout-Corrector starter kit."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown


FILE_ID = "1og3l0enR67rDwiAN44K4RchcFYAgsbNq"
STARTER_DIR = "layout_corrector_starter_kit"
REQUIRED_PATHS = (
    Path(STARTER_DIR)
    / "download"
    / "pretrained_weights"
    / "rico25"
    / "layout_corrector"
    / "0"
    / "config.yaml",
    Path(STARTER_DIR)
    / "download"
    / "pretrained_weights"
    / "publaynet"
    / "layout_corrector"
    / "0"
    / "config.yaml",
    Path(STARTER_DIR)
    / "download"
    / "pretrained_weights"
    / "crello-bbox"
    / "layout_corrector"
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


def _download_archive(file_id: str, archive: Path) -> None:
    tmp_archive = archive.with_name(f"{archive.name}.tmp")
    tmp_archive.unlink(missing_ok=True)
    gdown.download(id=file_id, output=str(tmp_archive), quiet=False)
    tmp_archive.replace(archive)


def ensure_downloaded(output_dir: Path, *, file_id: str = FILE_ID) -> Path:
    """Download and extract a complete Layout-Corrector starter archive.

    Args:
        output_dir: Directory for the archive and extracted starter tree.
        file_id: Google Drive file id for the original starter archive.

    Returns:
        The extracted starter-kit directory path.

    Raises:
        RuntimeError: If the downloaded archive or extracted tree lacks required
            checkpoint files.

    Examples:
        >>> ensure_downloaded  # doctest: +ELLIPSIS
        <function...
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "layout_corrector_starter.zip"
    if not _archive_is_complete(archive):
        _remove_incomplete_outputs(output_dir, archive)
        _download_archive(file_id, archive)
    if not _archive_is_complete(archive):
        raise RuntimeError(f"Incomplete Layout-Corrector starter archive: {archive}")
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
        raise RuntimeError(f"Incomplete Layout-Corrector starter extraction: {missing}")
    return output_dir / STARTER_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the original Layout-Corrector Google Drive starter kit and "
            "extract it under the requested cache directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-corrector/original"),
        help=(
            "Directory for layout_corrector_starter.zip and the extracted "
            "layout_corrector_starter_kit directory."
        ),
    )
    args = parser.parse_args()
    print(ensure_downloaded(args.output_dir))


if __name__ == "__main__":
    main()
