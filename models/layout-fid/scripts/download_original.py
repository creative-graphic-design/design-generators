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
    print("LayoutFlow source repository: https://github.com/JulianGuerreiro/LayoutFlow")
    print(
        "LayoutFlow checkpoint host: https://huggingface.co/JulianGuerreiro/LayoutFlow"
    )
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
    print(
        "LayoutDM release archive: https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip"
    )
    print(args.vendor_root / "layout-dm" / "download" / "fid_weights" / "FIDNetV3")
    print("2024 FIDNetV3 cross-check repos:")
    print(
        "https://huggingface.co/creative-graphic-design/layout-fidnet-v3-layoutdm-rico25"
    )
    print(
        "https://huggingface.co/creative-graphic-design/layout-fidnet-v3-layoutdm-publaynet"
    )
    print("https://huggingface.co/creative-graphic-design/layout-fidnet-v3-ralf-pku10")
    print("https://huggingface.co/creative-graphic-design/layout-fidnet-v3-ralf-cgl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
