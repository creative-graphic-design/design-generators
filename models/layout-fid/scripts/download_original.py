"""Print local asset requirements for layout FID conversion."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor"),
        help="Repository-relative vendor directory to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    """Report expected local assets without downloading large archives."""
    args = parse_args()
    layoutflow = args.vendor_root / "layout-flow" / "pretrained"
    for name in (
        "fid_rico.pth.tar",
        "fid_publaynet.pth.tar",
        "FIDNet_musig_val_rico.pt",
        "FIDNet_musig_test_rico.pt",
        "FIDNet_musig_val_publaynet.pt",
        "FIDNet_musig_test_publaynet.pt",
    ):
        print(layoutflow / name)
    print(args.vendor_root / "layout-dm" / "download" / "fid_weights" / "FIDNetV3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
