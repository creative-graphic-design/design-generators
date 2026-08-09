"""Document CGB-DM asset acquisition without downloading large data by default."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".cache/cgb-dm/original-assets",
        help="Directory where manually downloaded assets should be placed.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        default=True,
        help="Print required assets instead of downloading them.",
    )
    return parser.parse_args()


def main() -> None:
    """Print the expected asset location and source notes."""
    args = parse_args()
    print(f"Place the 13 GB Dataset.zip extract under {args.output_dir}/datasets/.")
    print("No generator checkpoint URL is public as of the issue #15 amendment.")


if __name__ == "__main__":
    main()
