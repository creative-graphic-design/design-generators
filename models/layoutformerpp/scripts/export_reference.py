"""Generate LayoutFormer++ reference metadata without committing fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write metadata describing an external vendor-reference run."""
    parser = argparse.ArgumentParser(
        description=(
            "Write local metadata for a LayoutFormer++ vendor-reference run. "
            "Generated tensors are intentionally kept out of the repository."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["rico", "publaynet"],
        required=True,
        help="Dataset name for the reference run. Required.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "LayoutFormer++ task name, for example gen_t, gen_ts, gen_r, "
            "refinement, completion, or ugen. Required."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=500,
        help="Random seed recorded for the vendor-reference run. Default: 500.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metadata.json is written. Required.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "seed": args.seed,
        "source": "vendor/ms-layout-generation/LayoutFormer++",
        "note": "Regenerate token/logit fixtures locally; do not commit generated tensors.",
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
