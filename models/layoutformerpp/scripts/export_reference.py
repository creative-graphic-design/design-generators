"""Generate LayoutFormer++ reference metadata without committing fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write metadata describing an external vendor-reference run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rico", "publaynet"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
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
