"""Generate Flex-DM vendor reference metadata for parity tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["crello", "rico"], required=True)
    parser.add_argument("--variant", default="ours-exp-ft")
    parser.add_argument(
        "--asset-dir", type=Path, default=Path(".cache/flex-dm/original")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="elem,pos,attr")
    parser.add_argument("--num-iter", default="1,4")
    return parser.parse_args()


def main() -> None:
    """Write reproducibility metadata for a vendor-reference run.

    The TensorFlow reference execution is intentionally not reimplemented here;
    this script records the exact command metadata and cache locations used by
    the heavyweight vendor parity job.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "variant": args.variant,
        "asset_dir": str(args.asset_dir),
        "seed": args.seed,
        "tasks": args.tasks.split(","),
        "num_iter": [int(item) for item in args.num_iter.split(",")],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.output_dir / "golden_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"wrote {args.output_dir / 'golden_metadata.json'}")


if __name__ == "__main__":
    main()
