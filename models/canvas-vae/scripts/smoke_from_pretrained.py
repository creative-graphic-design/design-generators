"""Load a saved CanvasVAE pipeline and check one unconditional generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from laygen.common.testing import assert_layout_output_schema

from canvas_vae import CanvasVAEPipeline


def main() -> None:
    """Run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory written by save_pretrained.")
    parser.add_argument("--batch-size", type=int, default=4, help="Layouts to generate (default: %(default)s).")
    args = parser.parse_args()
    pipe = CanvasVAEPipeline.from_pretrained(args.checkpoint, local_files_only=True)
    out = pipe(batch_size=args.batch_size, seed=0)
    assert_layout_output_schema(out, batch_size=args.batch_size)
    print(tuple(out.bbox.shape), tuple(out.labels.shape), tuple(out.mask.shape))


if __name__ == "__main__":
    main()
