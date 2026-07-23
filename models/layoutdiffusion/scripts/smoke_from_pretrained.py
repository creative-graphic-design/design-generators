"""Smoke-test LayoutDiffusion local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from layoutdiffusion import LayoutDiffusionPipeline
from layoutdiffusion.sampling import LayoutDiffusionSamplingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run one denoising step."""
    args = parse_args()
    for path in args.path:
        pipe = LayoutDiffusionPipeline.from_pretrained(path)
        out = cast(
            Any,
            pipe(
                batch_size=1,
                seed=101,
                sampling=LayoutDiffusionSamplingConfig(num_inference_steps=1),
            ),
        )
        print(path.name, out.bbox.shape, out.labels.shape, out.mask.shape)


if __name__ == "__main__":
    main()
