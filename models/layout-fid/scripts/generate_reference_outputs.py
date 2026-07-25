"""Generate a tiny deterministic layout FID reference batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write a deterministic reference batch for parity debugging."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bbox": torch.tensor(
                [
                    [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]],
                    [[0.4, 0.4, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
            "mask": torch.tensor([[True, True], [True, False]]),
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
