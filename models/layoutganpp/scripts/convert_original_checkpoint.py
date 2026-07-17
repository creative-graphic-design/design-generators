"""Convert an original LayoutGAN++ generator checkpoint to Transformers format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from layoutganpp.conversion import config_from_checkpoint_args
from layoutganpp.model_card import write_layoutganpp_model_card
from layoutganpp.modeling_layoutganpp import LayoutGANPPModel
from layoutganpp.processing_layoutganpp import LayoutGANPPProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an original LayoutGAN++ .pth.tar generator checkpoint into a "
            "Transformers-compatible model directory with processor files and a "
            "Hub README model card."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=Path(".cache/layoutganpp/original/layoutganpp_rico.pth.tar"),
        help=(
            "Path to the original checkpoint. The default expects "
            "download_original_weights.py to have downloaded the Rico checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutganpp/converted/layoutganpp-rico"),
        help=(
            "Directory where config, weights, processor files, and README.md are "
            "written."
        ),
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    config = config_from_checkpoint_args(checkpoint["args"])
    model = LayoutGANPPModel(config)
    model.load_state_dict(checkpoint["netG"], strict=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor = LayoutGANPPProcessor(
        dataset_name=config.dataset_name, id2label=config.id2label
    )
    processor.save_pretrained(str(args.output_dir))
    write_layoutganpp_model_card(args.output_dir, config.dataset_name)
    print(args.output_dir)


if __name__ == "__main__":
    main()
