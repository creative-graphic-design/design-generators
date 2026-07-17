"""Download and extract the original LayoutDM starter checkpoint bundle."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


URL = "https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download CyberAgentAILab's LayoutDM starter zip and extract it under "
            "the local cache used by conversion and vendor-parity scripts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-dm/original"),
        help=(
            "Directory that stores layoutdm_starter.zip and the extracted "
            "`download/` tree. From models/layout-dm, use "
            "`../../.cache/layout-dm/original` to write to the repo cache."
        ),
    )
    parser.add_argument(
        "--url",
        default=URL,
        help="Source URL for the original LayoutDM starter checkpoint zip.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "layoutdm_starter.zip"
    if not archive.exists():
        with requests.get(args.url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
