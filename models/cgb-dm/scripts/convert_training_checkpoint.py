"""Convert a CGB-DM training checkpoint into a Diffusers pipeline directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgb_dm.configuration_cgb_dm import cgb_dm_config_for_dataset
from cgb_dm.conversion import build_pipeline_from_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path.")
    parser.add_argument(
        "--dataset", choices=["pku_posterlayout", "cgl"], default="pku_posterlayout"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the converted pipeline."
    )
    return parser.parse_args()


def main() -> None:
    """Convert and save a CGB-DM pipeline."""
    args = parse_args()
    pipe = build_pipeline_from_checkpoint(
        args.checkpoint, config=cgb_dm_config_for_dataset(args.dataset)
    )
    pipe.save_pretrained(args.output_dir)
    print(Path(args.output_dir))


if __name__ == "__main__":
    main()
