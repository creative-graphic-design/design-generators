"""Convert a raw BASNet checkpoint into a Transformers directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from basnet import BASNetConfig, convert_original_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    convert_original_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        config=BASNetConfig(input_size=args.input_size),
    )


if __name__ == "__main__":
    main()
