"""Download original LT-Net Google Drive checkpoints.

The script stores vendor assets outside git and prints the expected checkpoint
locations for COCO and VG-MSDN.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-transformer/vendor"),
        help="Directory where Google Drive assets are downloaded.",
    )
    parser.add_argument(
        "--folder-url",
        default="https://drive.google.com/drive/folders/1pPJxX0ih6pgUpKjeIjIICso6SpOGHoaI?usp=sharing",
        help="Google Drive folder URL from the original README.",
    )
    return parser.parse_args()


def main() -> None:
    """Download the configured Google Drive folder with gdown."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import gdown

    gdown.download_folder(args.folder_url, output=str(args.output_dir), quiet=False)
    print(
        "Expected COCO checkpoint: saved/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth"
    )
    print(
        "Expected VG-MSDN checkpoint: saved/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth"
    )


if __name__ == "__main__":
    main()
