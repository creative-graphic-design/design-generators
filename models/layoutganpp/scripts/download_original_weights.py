"""Download the original LayoutGAN++ checkpoint files released by const-layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

WEIGHTS = {
    "rico": "layoutganpp_rico.pth.tar",
    "publaynet": "layoutganpp_publaynet.pth.tar",
    "magazine": "layoutganpp_magazine.pth.tar",
}
BASE_URL = "https://esslab.jp/~kotaro/files/const_layout"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download original LayoutGAN++ generator checkpoints into the local "
            "cache used by conversion and parity scripts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutganpp/original"),
        help="Directory where downloaded .pth.tar files are stored.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*WEIGHTS, "all"],
        default="all",
        help="Checkpoint dataset to download; 'all' downloads every released file.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = (
        WEIGHTS if args.dataset == "all" else {args.dataset: WEIGHTS[args.dataset]}
    )
    for dataset, filename in datasets.items():
        url = f"{BASE_URL}/{filename}"
        output = args.output_dir / filename
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with output.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"{dataset}: {output}")


if __name__ == "__main__":
    main()
