"""Prepare RICO splits, vocabulary counts, and split sizes for CanvasVAE.

Reads the semantic-annotation archive, deduplicates identical files, flattens
each view hierarchy, drops screens with more than 50 elements, assigns splits
by content hash, and writes ``train.jsonl``, ``val.jsonl``, ``test.jsonl``,
``vocabulary.json``, and ``count.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canvas_vae.processing_canvas_vae import prepare_rico_cache


def main() -> None:
    """Write the prepared cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(".cache/canvas-vae/data/semantic_annotations.zip"),
        help="RICO semantic-annotation zip (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/canvas-vae/data/rico"),
        help="Prepared split directory (default: %(default)s).",
    )
    args = parser.parse_args()
    print(json.dumps(prepare_rico_cache(args.archive, args.output_dir)))


if __name__ == "__main__":
    main()
