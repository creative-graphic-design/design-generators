"""Smoke-test LACE local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from lace import LacePipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run a short sample."""
    args = parse_args()
    for path in args.path:
        pipe = LacePipeline.from_pretrained(path).to(args.device)
        out = pipe(batch_size=1, seed=0, num_inference_steps=2)
        in_range = bool((out.bbox >= 0).all() and (out.bbox <= 1).all())
        print(path.name, tuple(out.bbox.shape), tuple(out.labels.shape), in_range)


if __name__ == "__main__":
    main()
