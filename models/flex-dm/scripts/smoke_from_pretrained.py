"""Run a local save/load smoke test for Flex-DM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flex_dm import FlexDmPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Load a converted pipeline and run one completion step."""
    args = parse_args()
    pipe = FlexDmPipeline.from_pretrained(args.checkpoint_dir, local_files_only=True)
    out = pipe(
        bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        labels=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
        feature_group="pos",
        seed=0,
    )
    print(
        {"bbox_shape": tuple(out.bbox.shape), "labels_shape": tuple(out.labels.shape)}
    )


if __name__ == "__main__":
    main()
