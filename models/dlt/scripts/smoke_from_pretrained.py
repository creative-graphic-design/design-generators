"""Smoke-test local DLT ``from_pretrained`` pipeline directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke script prints dynamic output.

from dlt import DLTPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run a short sample."""
    args = parse_args()
    for path in args.path:
        pipe = DLTPipeline.from_pretrained(path)
        out = cast(
            Any, pipe(batch_size=1, num_elements=2, seed=0, num_inference_steps=2)
        )
        print(path.name, out.bbox.shape, out.labels.shape, type(out).__name__)


if __name__ == "__main__":
    main()
