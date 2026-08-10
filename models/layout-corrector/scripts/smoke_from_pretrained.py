"""Smoke-test Layout-Corrector local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from layout_corrector import LayoutCorrectorPipeline


class _HasShape(Protocol):
    """Tensor-like object with a printable shape."""

    @property
    def shape(self) -> Sequence[int]:
        """Return tensor dimensions."""
        ...


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
        LayoutGenerationOutput,
        pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic"),
    )
    sequences = cast(_HasShape, out.sequences)
    print(sequences.shape)


if __name__ == "__main__":
    main()
