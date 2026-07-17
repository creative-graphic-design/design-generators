"""Convert an original LACE checkpoint into a Diffusers-style pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from laygen.common import DatasetName
from lace.conversion import build_pipeline_from_vendor_checkpoint
from lace.configuration_lace import normalize_dataset
from lace.model_card import write_lace_model_card

DATASET_CHOICES: Final[tuple[str, ...]] = (
    str(DatasetName.publaynet),
    str(DatasetName.rico13),
    str(DatasetName.rico25),
)


def _default_checkpoint(dataset: DatasetName | str) -> str:
    dataset_name = normalize_dataset(dataset)
    return f".cache/lace/original/model/{dataset_name}_best.pt"


def _default_output(dataset: DatasetName | str) -> str:
    dataset_name = normalize_dataset(dataset)
    return f".cache/lace/converted/lace-{dataset_name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local original LACE checkpoint into a save_pretrained "
            "Diffusers pipeline directory and write its Hub README.md model card."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DatasetName.publaynet),
        choices=DATASET_CHOICES,
        help="Dataset/checkpoint family to convert.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Original checkpoint path. Defaults to "
            ".cache/lace/original/model/<dataset>_best.pt."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pipeline directory. Defaults to .cache/lace/converted/lace-<dataset>.",
    )
    parser.add_argument(
        "--ddim-num-steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps stored in the converted scheduler.",
    )
    args = parser.parse_args()
    dataset = normalize_dataset(args.dataset)
    checkpoint = args.checkpoint or _default_checkpoint(dataset)
    output = Path(args.output or _default_output(dataset))
    pipe = build_pipeline_from_vendor_checkpoint(
        dataset, checkpoint, ddim_num_steps=args.ddim_num_steps
    )
    pipe.save_pretrained(output)
    write_lace_model_card(output, dataset)


if __name__ == "__main__":
    main()
