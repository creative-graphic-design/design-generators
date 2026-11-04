"""Smoke-test local RADM ``from_pretrained`` pipeline directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from PIL import Image

from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from radm import RADMPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each local pipeline and run a tiny image-conditioned sample."""
    args = parse_args()
    image = Image.new("RGB", (32, 32), "white")
    for path in args.path:
        pipe = RADMPipeline.from_pretrained(path)
        out = cast(LayoutGenerationOutput, pipe(image, seed=0, num_inference_steps=2))
        print(path.name, out.bbox.shape, out.labels.shape, type(out).__name__)


if __name__ == "__main__":
    main()
