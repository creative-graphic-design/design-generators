from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


URL = "https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/layout-dm/original")
    )
    parser.add_argument("--url", default=URL)
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
