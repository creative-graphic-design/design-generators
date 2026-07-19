"""Smoke-test LayoutGAN++ local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from layoutganpp import LayoutGANPPPipeline


LABELS: dict[str, list[list[str | int]]] = {
    "rico": [["Toolbar", "Image"]],
    "publaynet": [["text", "figure"]],
    "magazine": [["text", "image"]],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def _dataset_from_path(path: Path) -> str:
    name = path.name
    for dataset in LABELS:
        if dataset in name:
            return dataset
    raise ValueError(f"Cannot infer LayoutGAN++ dataset from {path}")


def main() -> None:
    """Load each converted checkpoint and run one conditioned sample."""
    args = parse_args()
    for path in args.path:
        dataset = _dataset_from_path(path)
        pipe = LayoutGANPPPipeline.from_pretrained(path)
        out = cast(Any, pipe(labels=LABELS[dataset], seed=0))
        print(dataset, tuple(out.bbox.shape))


if __name__ == "__main__":
    main()
