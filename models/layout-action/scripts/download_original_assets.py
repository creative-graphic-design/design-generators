"""Download or inventory original LayoutAction assets.

This script writes an ``asset_manifest.json`` outside ``vendor/``. It can either
copy no files and inventory a pre-downloaded source directory, or invoke
``gdown`` for the original Google Drive folder when the download extra is
installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Final

GOOGLE_DRIVE_FOLDER_ID: Final[str] = "1KU9q83gzKD2HGoBduN2CWC0LHUmDcFy0"
EXPECTED_CHECKPOINTS: Final[tuple[str, ...]] = (
    "pretrained_model_resources/Ours/rico.pth",
    "pretrained_model_resources/Ours/publaynet.pth",
)
OPTIONAL_CHECKPOINTS: Final[tuple[str, ...]] = (
    "pretrained_model_resources/Ours/infoppt.pth",
)
OPTIONAL_PROCESSED_DATA: Final[tuple[str, ...]] = (
    "processed_data/LayoutGAN++/rico/processed/test.pt",
    "processed_data/LayoutGAN++/rico/processed/val.pt",
    "processed_data/LayoutGAN++/publaynet/processed/test.pt",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-action/original"),
        help="Directory where assets and asset_manifest.json are stored.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing pre-downloaded asset directory to inventory or copy.",
    )
    parser.add_argument(
        "--google-drive-folder-id",
        default=GOOGLE_DRIVE_FOLDER_ID,
        help="Original LayoutAction Google Drive folder id.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run gdown to download the Google Drive folder into output-dir.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, object]:
    """Build a deterministic asset manifest."""
    files: dict[str, dict[str, object]] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for rel in EXPECTED_CHECKPOINTS + OPTIONAL_CHECKPOINTS + OPTIONAL_PROCESSED_DATA:
        path = root / rel
        if path.exists():
            files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        elif rel in EXPECTED_CHECKPOINTS:
            missing_required.append(rel)
        else:
            missing_optional.append(rel)
    return {
        "source": "LayoutAction Google Drive resources",
        "google_drive_folder_id": GOOGLE_DRIVE_FOLDER_ID,
        "files": files,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def main() -> None:
    """Run the asset inventory workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_dir is not None:
        source = args.source_dir.resolve()
        if source != args.output_dir.resolve():
            shutil.copytree(source, args.output_dir, dirs_exist_ok=True)
    elif args.download:
        subprocess.run(
            [
                "gdown",
                "--folder",
                args.google_drive_folder_id,
                "-O",
                str(args.output_dir),
            ],
            check=True,
        )
    manifest = inventory(args.output_dir)
    with (args.output_dir / "asset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if manifest["missing_required"]:
        raise FileNotFoundError(
            f"Missing required assets: {manifest['missing_required']}"
        )


if __name__ == "__main__":
    main()
