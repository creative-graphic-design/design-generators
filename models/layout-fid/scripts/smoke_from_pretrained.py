"""Run a local ``from_pretrained`` smoke test for a layout FID artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from layout_fid import LayoutFIDEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Load an evaluator and extract one feature vector."""
    args = parse_args()
    evaluator = LayoutFIDEvaluator.from_pretrained(args.model_id)
    features = evaluator.extract_features(
        bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        labels=torch.tensor([[0]]),
    )
    print(tuple(features.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
