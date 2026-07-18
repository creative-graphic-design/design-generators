"""Generate LayoutDiffusion vendor reference metadata.

This script intentionally keeps heavyweight outputs outside git. It validates
inputs and writes only a small metadata file describing how to regenerate
goldens with the original implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from layoutdiffusion.conversion import validate_checkpoint_artifacts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["rico25", "publaynet"])
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ungen", "type", "refine"],
        help="Vendor sampling tasks to run in the original environment.",
    )
    return parser.parse_args()


def main() -> None:
    """Write reference-generation metadata."""
    args = parse_args()
    artifacts = validate_checkpoint_artifacts(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "checkpoint_dir": str(args.checkpoint_dir),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "seed": args.seed,
        "tasks": args.tasks,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote={args.output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
