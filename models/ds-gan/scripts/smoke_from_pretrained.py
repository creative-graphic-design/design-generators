"""Smoke-test local DS-GAN ``from_pretrained`` loading."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from laygen.modeling_outputs import LayoutGenerationOutput

from ds_gan import DSGANPipeline


def main() -> None:
    """Load converted checkpoints and run a minimal local inference."""
    parser = argparse.ArgumentParser(
        description="Run a DS-GAN from_pretrained smoke check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        required=True,
        help="Converted DS-GAN checkpoint directory.",
    )
    args = parser.parse_args()

    for path in args.path:
        pipe = DSGANPipeline.from_pretrained(path, local_files_only=True)
        output = pipe(pixel_values=torch.zeros(1, 4, 350, 240), seed=0)
        output = cast(LayoutGenerationOutput, output)
        assert output.bbox.shape == (1, pipe.config.max_elem, 4)
        assert output.labels.shape == output.mask.shape
        print(
            f"{path}: bbox={tuple(output.bbox.shape)} labels={tuple(output.labels.shape)}"
        )


if __name__ == "__main__":
    main()
