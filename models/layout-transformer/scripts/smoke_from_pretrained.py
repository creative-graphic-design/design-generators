"""Smoke-test LayoutTransformer local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from layout_transformer import LayoutObject, LayoutTransformerPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run one deterministic sample."""
    args = parse_args()
    for path in args.path:
        pipe = LayoutTransformerPipeline.from_pretrained(path, local_files_only=True)
        out = cast(
            Any,
            pipe(objects=[LayoutObject(id="object-1", label=0)], seed=0),
        )
        assert out.bbox.shape[-1] == 4
        assert out.labels.shape == out.mask.shape
        print(path.name, out.bbox.shape, out.labels.shape)


if __name__ == "__main__":
    main()
