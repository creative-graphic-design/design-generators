"""Convert a raw LayoutAction checkpoint to HF-style files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layout_action import (
    LayoutActionConfig,
    convert_layout_action_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["rico", "rico13", "publaynet", "infoppt"], required=True
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Raw vendor .pth checkpoint."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Converted checkpoint directory."
    )
    parser.add_argument("--n-layer", type=int, default=6, help="GPT block count.")
    parser.add_argument(
        "--n-head", type=int, default=8, help="GPT attention head count."
    )
    parser.add_argument("--n-embd", type=int, default=512, help="GPT hidden size.")
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    config = LayoutActionConfig(
        dataset_name=args.dataset,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    report = convert_layout_action_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        config=config,
        strict=True,
    )
    with (args.output_dir / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
