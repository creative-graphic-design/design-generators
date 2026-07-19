"""Export LT-Net vendor reference outputs for parity tests.

This script intentionally writes artifacts outside git. It records metadata
needed to regenerate goldens and leaves large tensors in the requested output
directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root", type=Path, default=Path("vendor/layout-transformer")
    )
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["coco", "vg_msdn"], required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Write reproducibility metadata for vendor reference generation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "vendor_root": str(args.vendor_root),
        "cfg_path": str(args.cfg_path),
        "checkpoint_path": str(args.checkpoint_path),
        "dataset_name": args.dataset_name,
        "sample_indices": args.sample_indices,
        "seed": args.seed,
        "cuda_visible_devices": "2",
        "status": "metadata-only; run vendor train.py --eval_only for tensor export",
    }
    with (args.output_dir / "reference_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
