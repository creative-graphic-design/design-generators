"""Convert an original PosterLayout DS-GAN checkpoint to Transformers format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from ds_gan.conversion import convert_vendor_state_dict
from ds_gan.configuration_ds_gan import DSGANConfig
from ds_gan.model_card import write_dsgan_model_card
from ds_gan.modeling_ds_gan import DSGANModel
from ds_gan.pipeline_ds_gan import DSGANPipeline
from ds_gan.processing_ds_gan import DSGANProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert original DS-GAN-Epoch300.pth weights into a pipeline root "
            "with model and processor subfolders."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=Path(".cache/ds-gan/original/DS-GAN-Epoch300.pth"),
        help="Path to the original vendor DS-GAN checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/ds-gan/converted/ds-gan-pku-posterlayout"),
        help="Directory where converted pipeline files are written.",
    )
    args = parser.parse_args()

    config = DSGANConfig()
    model = DSGANModel(config)
    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    model.load_state_dict(convert_vendor_state_dict(checkpoint), strict=True)
    processor = DSGANProcessor(
        dataset_name=config.dataset_name,
        id2label=cast(dict[int | str, str], config.id2label),
        image_size=cast(tuple[int, int], config.image_size),
    )
    pipe = DSGANPipeline(model=model, processor=processor, config=config)
    pipe.save_pretrained(args.output_dir)
    write_dsgan_model_card(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
