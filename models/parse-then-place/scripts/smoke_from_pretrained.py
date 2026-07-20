"""Smoke-test Parse-Then-Place local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from parse_then_place import ParseThenPlacePipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load a converted checkpoint and verify required components."""
    args = parse_args()
    pipeline = ParseThenPlacePipeline.from_pretrained(args.path, local_files_only=True)
    print(
        pipeline.config.dataset_name,
        pipeline.parser is not None,
        pipeline.placement is not None,
    )


if __name__ == "__main__":
    main()
