"""Write local metadata for LACE vendor parity reference generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_checkpoint(dataset: str) -> str:
    return f".cache/lace/original/model/{dataset}_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record local metadata for a LACE vendor parity reference run. "
            "The actual golden tensors are local-only because they depend on "
            "the original vendor environment and checkpoint files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="publaynet",
        choices=["publaynet", "rico13", "rico25"],
        help="Dataset/checkpoint family to record.",
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
        "--output-dir",
        default=".cache/lace/reference/publaynet",
        help="Directory where metadata.json is written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reference random seed.")
    args = parser.parse_args()
    checkpoint = args.checkpoint or _default_checkpoint(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "checkpoint": checkpoint,
        "seed": args.seed,
        "note": "Run vendor/lace reference generation in an isolated vendor environment; fixtures are local-only.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
