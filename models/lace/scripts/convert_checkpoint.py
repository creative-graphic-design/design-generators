"""Convert an original LACE checkpoint into a Diffusers-style pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from lace.conversion import build_pipeline_from_vendor_checkpoint
from lace.model_card import write_lace_model_card


def _default_checkpoint(dataset: str) -> str:
    return f".cache/lace/original/model/{dataset}_best.pt"


def _default_output(dataset: str) -> str:
    return f".cache/lace/converted/lace-{dataset}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local original LACE checkpoint into a save_pretrained "
            "Diffusers pipeline directory and write its Hub README.md model card."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="publaynet",
        choices=["publaynet", "rico13", "rico25"],
        help="Dataset/checkpoint family to convert.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Original checkpoint path. Defaults to "
            ".cache/lace/original/model/<dataset>_best.pt."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pipeline directory. Defaults to .cache/lace/converted/lace-<dataset>.",
    )
    parser.add_argument(
        "--ddim-num-steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps stored in the converted scheduler.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint or _default_checkpoint(args.dataset)
    output = Path(args.output or _default_output(args.dataset))
    pipe = build_pipeline_from_vendor_checkpoint(
        args.dataset, checkpoint, ddim_num_steps=args.ddim_num_steps
    )
    pipe.save_pretrained(output)
    write_lace_model_card(output, args.dataset)


if __name__ == "__main__":
    main()
