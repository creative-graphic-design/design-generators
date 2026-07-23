"""Convert original House-GAN raw generator checkpoints to HF files."""

from __future__ import annotations

import argparse

from housegan.conversion import convert_original_checkpoint


def main() -> None:
    """Run checkpoint conversion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Original .pth checkpoint")
    parser.add_argument("--target-set", default="D", help="House-GAN target set")
    parser.add_argument("--checkpoint-step", type=int, default=200000)
    parser.add_argument("--output-dir", required=True, help="Output checkpoint root")
    args = parser.parse_args()
    convert_original_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        target_set=args.target_set,
        checkpoint_step=args.checkpoint_step,
    )


if __name__ == "__main__":
    main()
