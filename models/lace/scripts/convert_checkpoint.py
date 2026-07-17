from __future__ import annotations

import argparse

from lace.conversion import build_pipeline_from_vendor_checkpoint
from lace.model_card import write_lace_model_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["publaynet", "rico13", "rico25"]
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ddim-num-steps", type=int, default=100)
    args = parser.parse_args()
    pipe = build_pipeline_from_vendor_checkpoint(
        args.dataset, args.checkpoint, ddim_num_steps=args.ddim_num_steps
    )
    pipe.save_pretrained(args.output)
    write_lace_model_card(args.output, args.dataset)


if __name__ == "__main__":
    main()
