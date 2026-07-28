"""Run a local PosterLlama from_pretrained smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from laygen.modeling_outputs import LayoutGenerationOutput
from posterllama import PosterLlamaPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_dir", type=Path, help="Local pipeline directory.")
    parser.add_argument(
        "--parser-only",
        action="store_true",
        help="Only verify loading when runtime assets are absent.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a local pipeline and optionally run generation."""
    args = parse_args()
    pipe = PosterLlamaPipeline.from_pretrained(args.pipeline_dir, local_files_only=True)
    if args.parser_only:
        print(pipe.config.model_type)
        return
    output = cast(LayoutGenerationOutput, pipe(images=None))
    print(tuple(output.bbox.shape))


if __name__ == "__main__":
    main()
