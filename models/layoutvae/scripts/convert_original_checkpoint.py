"""Convert LayoutVAE checkpoint files to Transformers format."""

from __future__ import annotations

import argparse
from pathlib import Path

from layoutvae.conversion import convert_state_dicts, load_original_state_dicts
from layoutvae.model_card import write_layoutvae_model_card


def main() -> None:
    """Run the checkpoint conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert LayoutVAE count and box checkpoints into HF artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("vendor/layout-generation-baselines/LayoutVAE"),
        help="Path to the LayoutVAE source root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutvae/converted/layoutvae-publaynet"),
        help="Directory for converted artifacts.",
    )
    args = parser.parse_args()

    count_state_dict, bbox_state_dict = load_original_state_dicts(args.source_root)
    output_dir = convert_state_dicts(
        count_state_dict=count_state_dict,
        bbox_state_dict=bbox_state_dict,
        output_dir=args.output_dir,
    )
    write_layoutvae_model_card(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
