"""Export a trained PixelVAE encoder from a Lightning checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixel_vae.training import PixelVAETrainingModule


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint and encoder output directory."""
    parser = argparse.ArgumentParser(
        description="Export a trained PixelVAE encoder in save_pretrained format."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load a Lightning checkpoint and save its encoder for inference."""
    args = parse_args()
    module = PixelVAETrainingModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    module.model.encoder.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
