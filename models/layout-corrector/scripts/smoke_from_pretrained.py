"""Smoke-test Layout-Corrector local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from layout_corrector import LayoutCorrectorPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load a converted checkpoint and run one deterministic correction."""
    args = parse_args()
    pipe = LayoutCorrectorPipeline.from_pretrained(args.path)
    out = cast(
        Any, pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
    )
    print(out.sequences.shape)


if __name__ == "__main__":
    main()
