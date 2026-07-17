from __future__ import annotations

import argparse
from pathlib import Path

import torch

from const_layout.conversion import config_from_checkpoint_args
from const_layout.modeling_const_layout import ConstLayoutForGeneration
from const_layout.processing_const_layout import ConstLayoutProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    config = config_from_checkpoint_args(checkpoint["args"])
    model = ConstLayoutForGeneration(config)
    model.load_state_dict(checkpoint["netG"], strict=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor = ConstLayoutProcessor(
        dataset_name=config.dataset_name, id2label=config.id2label
    )
    processor.save_pretrained(str(args.output_dir))
    print(args.output_dir)


if __name__ == "__main__":
    main()
