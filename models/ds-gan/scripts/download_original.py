"""Print manual download instructions for original DS-GAN assets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the local cache directory for original PosterLayout DS-GAN "
            "assets. The upstream dataset requires a signed agreement, and the "
            "weights are hosted on PKU Netdisk/Google Drive, so this script does "
            "not bypass those terms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/ds-gan/original"),
        help="Directory where manually downloaded assets should be placed.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Place DS-GAN-Epoch300.pth under: {args.output_dir}")
    print("Original weights: Google Drive folder linked from vendor README.")
    print("PKU PosterLayout data requires the upstream release agreement.")


if __name__ == "__main__":
    main()
