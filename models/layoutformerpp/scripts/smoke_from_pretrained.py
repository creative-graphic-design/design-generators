"""Smoke-test LayoutFormer++ local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from layoutformerpp import LayoutFormerPPPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and print tokenizer metadata."""
    args = parse_args()
    for path in args.path:
        pipe = LayoutFormerPPPipeline.from_pretrained(path, local_files_only=True)
        print(
            pipe.config.dataset, pipe.config.task, pipe.processor.tokenizer.vocab_size
        )


if __name__ == "__main__":
    main()
