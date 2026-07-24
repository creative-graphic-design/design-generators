"""Convert a DLT ``save_pretrained`` model directory into a pipeline directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt.configuration_dlt import DLTConfig
from dlt.conversion import convert_save_pretrained_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DLT model directory produced by model.save_pretrained "
            "into a Diffusers-compatible pipeline directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["publaynet", "rico13", "magazine"],
        default="publaynet",
    )
    parser.add_argument("--max-num-comp", type=int, default=None)
    parser.add_argument("--categories-num", type=int, default=None)
    return parser


def main() -> None:
    """Run checkpoint conversion."""
    args = build_parser().parse_args()
    config = DLTConfig(
        dataset_name=args.dataset,
        max_num_comp=args.max_num_comp,
        categories_num=args.categories_num,
    )
    convert_save_pretrained_directory(
        args.checkpoint_dir, args.output_dir, config=config
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
