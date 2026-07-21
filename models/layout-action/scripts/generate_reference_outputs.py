"""Generate LayoutAction vendor reference metadata and parity tensors.

The generated ``.pt`` files are cache artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["rico", "publaynet", "infoppt"], required=True
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(".cache/layout-action/original/Resources"),
        help="Directory containing original Resources files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-action/references"),
        help="Directory for generated reference metadata and tensors.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Vendor random seed.")
    parser.add_argument(
        "--eval-command",
        choices=["random_generate", "category_generate", "reconstruction"],
        default="category_generate",
        help="Vendor evaluation command.",
    )
    parser.add_argument(
        "--num-batches", type=int, default=2, help="Reference batch count."
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/layout-action/LayoutAction"),
        help="Vendor LayoutAction code directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Write reproducibility metadata and call the vendor entry point."""
    args = parse_args()
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.asset_dir / "pretrained_model_resources" / "Ours" / f"{args.dataset}.pth"
    )
    meta = {
        "dataset": args.dataset,
        "asset_dir": str(args.asset_dir),
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "eval_command": args.eval_command,
        "num_batches": args.num_batches,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vendor_root": str(args.vendor_root),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    command = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--split",
        "val",
        "--evaluate",
        "--model_path",
        str(checkpoint.resolve()),
        "--eval_command",
        args.eval_command,
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, cwd=args.vendor_root, check=True)


if __name__ == "__main__":
    main()
