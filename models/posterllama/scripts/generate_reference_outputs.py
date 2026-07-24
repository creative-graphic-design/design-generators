"""Run original PosterLlama reference generation with explicit local assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root", type=Path, required=True, help="Original repository root."
    )
    parser.add_argument(
        "--checkpoint-path", type=Path, required=True, help="Raw checkpoint path."
    )
    parser.add_argument(
        "--base-llm-path", type=Path, required=True, help="Backbone directory."
    )
    parser.add_argument(
        "--image-root", type=Path, required=True, help="Input image root."
    )
    parser.add_argument(
        "--jsonl", type=Path, required=True, help="Reference JSONL input."
    )
    parser.add_argument(
        "--output-metadata", type=Path, required=True, help="Metadata JSON to write."
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Single device for reference generation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    parser.add_argument(
        "--do-sample", action="store_true", help="Use sampled generation."
    )
    return parser.parse_args()


def main() -> None:
    """Write reproducibility metadata for a gated original-code run."""
    args = parse_args()
    for path in (
        args.vendor_root,
        args.checkpoint_path,
        args.base_llm_path,
        args.image_root,
        args.jsonl,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(
        json.dumps(
            {
                "vendor_root": str(args.vendor_root),
                "checkpoint_path": str(args.checkpoint_path),
                "base_llm_path": str(args.base_llm_path),
                "image_root": str(args.image_root),
                "jsonl": str(args.jsonl),
                "device": args.device,
                "seed": args.seed,
                "do_sample": args.do_sample,
                "status": "metadata-only; generated texts are not committed",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(args.output_metadata)


if __name__ == "__main__":
    main()
