"""Smoke-test Coarse-to-Fine local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from coarse_to_fine import (
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run one generation call."""
    args = parse_args()
    for path in args.path:
        model = CoarseToFineForLayoutGeneration.from_pretrained(path)
        processor = CoarseToFineProcessor.from_pretrained(path)
        pipe = CoarseToFinePipeline(model=model, processor=processor)
        out = cast(Any, pipe(batch_size=1, seed=0))
        assert out.bbox.shape == (1, 20, 4)
        assert out.labels.shape == (1, 20)
        assert bool(out.mask.any())
        print(model.config.dataset, out.bbox.shape, out.labels.shape)


if __name__ == "__main__":
    main()
