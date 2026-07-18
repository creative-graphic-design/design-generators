"""Generate metadata for Parse-Then-Place vendor parity references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    """Record the deterministic vendor reference-generation contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["rico", "web"], required=True)
    parser.add_argument(
        "--stage2-mode", choices=["pretrain", "finetune"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": args.dataset_name,
        "stage2_mode": args.stage2_mode,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
        "vendor_root": "vendor/ms-layout-generation/Parse-Then-Place",
        "goldens_policy": "Generated tensors/text live outside git.",
    }
    (args.output_dir / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
