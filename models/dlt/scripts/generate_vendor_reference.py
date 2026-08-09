"""Generate DLT reference metadata for gated parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Record the command metadata for regenerating DLT reference outputs. "
            "The tensors are written outside git by the original implementation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--epoch", type=str, required=True)
    parser.add_argument(
        "--condition",
        choices=["all", "whole_box", "loc"],
        default="all",
    )
    parser.add_argument("--output-metadata", type=Path, required=True)
    return parser


def main() -> None:
    """Write reproducibility metadata for a DLT reference run."""
    args = build_parser().parse_args()
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "workdir": args.workdir,
                "epoch": args.epoch,
                "condition": args.condition,
                "command": (
                    "CUDA_VISIBLE_DEVICES=0 python vendor/dlt/dlt/generate_samples.py "
                    f"--config {args.config} --workdir {args.workdir} "
                    f"--epoch {args.epoch} --cond_type {args.condition}"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(args.output_metadata)


if __name__ == "__main__":
    main()
