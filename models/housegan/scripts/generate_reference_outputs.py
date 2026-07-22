"""Generate House-GAN vendor reference metadata and external artifact paths."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch


def main() -> None:
    """Record reproducibility metadata for vendor reference generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-set", default="D")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 6, 42])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "vendor_dir": args.vendor_dir,
        "assets_dir": args.assets_dir,
        "checkpoint": args.checkpoint,
        "target_set": args.target_set,
        "indices": args.indices,
        "seed": args.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "artifacts": {
            "input_graphs": str(output_dir / "input_graphs.pt"),
            "latents": str(output_dir / "latents.pt"),
            "forward_masks": str(output_dir / "forward_masks.pt"),
            "decoded_layouts": str(output_dir / "decoded_layouts.pt"),
        },
    }
    (output_dir / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
