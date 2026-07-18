"""CLI wrapper for LayouSyn checkpoint conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from layousyn.conversion import convert_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse conversion arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-name", required=True)
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        variant_name=args.variant_name,
    )


if __name__ == "__main__":
    main()
