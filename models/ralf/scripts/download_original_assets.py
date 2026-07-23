"""Download or register the original RALF cache bundle.

The original authors distribute a large `cache.zip` outside this repository.
This script either records an existing cache directory or downloads the bundle
with `gdown` when requested.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

GOOGLE_DRIVE_ID = "1b357gVAnCSqMfbP3Cc2ey6LCeoohfYAi"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/ralf/cache"),
        help="Directory containing or receiving the unpacked RALF cache.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(".cache/ralf/cache.zip"),
        help="Path to the downloaded authors' cache.zip.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download cache.zip with gdown before writing the manifest.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unpack cache.zip into --cache-dir before writing the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".cache/ralf/cache_manifest.json"),
        help="Manifest path to write.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cache registration workflow."""
    args = parse_args()
    args.cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.download:
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Install the root download extra or gdown to use --download"
            ) from exc
        gdown.download(id=GOOGLE_DRIVE_ID, output=str(args.zip_path), quiet=False)
    if args.unzip:
        with zipfile.ZipFile(args.zip_path) as archive:
            archive.extractall(args.cache_dir.parent)
        nested_cache = args.cache_dir.parent / "cache"
        if nested_cache.exists() and nested_cache != args.cache_dir:
            args.cache_dir = nested_cache
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        str(path.relative_to(args.cache_dir))
        for path in args.cache_dir.rglob("*")
        if path.is_file()
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "cache_dir": str(args.cache_dir),
                "downloaded_with": "gdown" if args.download else "preexisting",
                "file_count": len(files),
                "sample_files": files[:200],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
