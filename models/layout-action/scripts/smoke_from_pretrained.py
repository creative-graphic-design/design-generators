"""Run a local ``from_pretrained`` smoke test for LayoutAction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from layout_action import LayoutActionPipeline
from laygen.modeling_outputs import LayoutGenerationOutput


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Converted checkpoint root.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    return parser.parse_args()


def main() -> None:
    """Load a pipeline and generate one layout."""
    args = parse_args()
    pipe = LayoutActionPipeline.from_pretrained(args.checkpoint, local_files_only=True)
    output = cast(
        LayoutGenerationOutput,
        pipe(
            batch_size=1,
            condition_type="unconditional",
            seed=args.seed,
            sampling="greedy",
        ),
    )
    print(output.bbox.shape, output.labels.shape, output.mask.shape)


if __name__ == "__main__":
    main()
