"""Generate local vendor-reference metadata for Coarse-to-Fine parity tests.

The full vendor reference path requires LayoutFormer++ vendor dependencies and
the downloaded original checkpoint. This script records deterministic metadata
and writes placeholder paths for locally generated tensors; parity tests skip
cleanly when the tensor artifact is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["rico25", "publaynet"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write deterministic reference-generation metadata."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "artifact": "reference.pt",
        "note": "Run vendor inference in this directory to create reference.pt.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
