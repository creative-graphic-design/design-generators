from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rico25", "publaynet"], required=True)
    parser.add_argument("--starter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling", default="deterministic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "sampling": args.sampling,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "starter_dir": str(args.starter_dir),
        "fixtures_committed": False,
        "note": "Run in the vendor environment to regenerate local parity tensors.",
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output_dir / "meta.json")


if __name__ == "__main__":
    main()
