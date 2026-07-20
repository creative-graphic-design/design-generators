"""Convert original LT-Net checkpoints into HF-style local directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from layout_transformer.conversion import convert_original_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["coco", "vg_msdn"], required=True)
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    convert_original_checkpoint(
        checkpoint_path=args.checkpoint_path,
        cfg_path=args.cfg_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        dataset_name=cast(Literal["coco", "vg_msdn"], args.dataset_name),
    )


if __name__ == "__main__":
    main()
