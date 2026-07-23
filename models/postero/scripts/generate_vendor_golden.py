"""Regenerate deterministic PosterO prompt/parsing parity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from postero.config import PosterOConfig
from postero.exemplars import select_exemplars
from postero.parser import parse_svg_response
from postero.vendor_parity import golden_prompt, fixture_records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/vendor_parity/golden_metadata.json"),
        help="Path to write JSON metadata, relative to models/postero if not absolute.",
    )
    return parser.parse_args()


def main() -> None:
    """Write parity metadata."""
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else package_root / args.output
    config = PosterOConfig(sample_size=1, n_valid_layouts=1, num_return=2)
    query, candidates = fixture_records()
    prompt = golden_prompt()
    selected = select_exemplars(query, candidates, config=config)
    elements, _diagnostics = parse_svg_response(
        '<svg><rect data-label="text_1" x="10" y="20" width="30" height="40"/></svg>',
        config=config,
    )
    metadata = {
        "cases": 4,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "selected_exemplar_ids": [record.id for record in selected],
        "parser_labels": [element.label for element in elements],
        "parser_bbox_ltrb": [list(element.bbox_ltrb) for element in elements],
        "retry_attempts": 2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
