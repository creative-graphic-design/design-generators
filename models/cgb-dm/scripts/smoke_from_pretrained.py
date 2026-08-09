"""Smoke-test loading a converted CGB-DM pipeline."""

from __future__ import annotations

import argparse

import torch

from cgb_dm import CGBDMPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Pipeline directory or Hub id.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    """Load a pipeline and run one content-image call."""
    args = parse_args()
    pipe = CGBDMPipeline.from_pretrained(
        args.path, local_files_only=args.local_files_only
    )
    output = pipe(
        pixel_values=torch.zeros(1, 4, *pipe.processor.image_size),
        num_inference_steps=1,
        seed=1,
    )
    print(tuple(output.bbox.shape))


if __name__ == "__main__":
    main()
