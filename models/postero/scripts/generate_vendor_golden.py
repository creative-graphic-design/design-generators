"""Regenerate deterministic PosterO prompt/parsing parity metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from postero.vendor_parity import (
    implementation_reference,
    implementation_retry_calls,
    parity_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/vendor_parity/golden_metadata.json"),
        help="Path to write JSON metadata, relative to models/postero if not absolute.",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=None,
        help="Original PosterO repository path. Defaults to the checked-out vendor/postero path.",
    )
    return parser.parse_args()


def main() -> None:
    """Write parity metadata."""
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root / "tests" / "vendor_parity"))
    from vendor_reference import vendor_reference

    output = args.output if args.output.is_absolute() else package_root / args.output
    config = parity_config()
    vendor = vendor_reference(args.vendor_root)
    implementation = implementation_reference()
    metadata = {
        "cases": 4,
        "config": {
            "structure": str(config.structure),
            "injection": str(config.injection),
            "pool_strategy": str(config.pool_strategy),
            "rank_strategy": str(config.rank_strategy),
            "sample_size": config.sample_size,
            "n_valid_layouts": config.n_valid_layouts,
            "num_return": config.num_return,
        },
        "prompt_sha256": vendor["prompt_sha256"],
        "implementation_prompt_sha256": implementation["prompt_sha256"],
        "selected_exemplar_ids": vendor["selected_exemplar_ids"],
        "parser_labels": vendor["parser_labels"],
        "parser_bbox_ltrb": vendor["parser_bbox_ltrb"],
        "parser_comparison_count": len(vendor["parser_labels"]),
        "vendor_retry_generate_calls": vendor["retry_generate_calls"],
        "implementation_retry_generate_calls": implementation_retry_calls(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
