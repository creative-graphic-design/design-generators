"""Convert original LayoutFlow Lightning checkpoints to Diffusers pipeline folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import torch

from laygen.common.labels import DatasetName
from laygen.common.vendor import vendor_root
from layout_flow.configuration_layout_flow import (
    LayoutFlowConfig,
    normalize_dataset_name,
)
from layout_flow.conversion import build_pipeline, convert_lightning_state_dict
from layout_flow.model_card import save_layoutflow_model_card


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_ORIGINAL_DIR: Final[Path] = (
    REPO_ROOT / ".cache" / "layout-flow" / "original" / "checkpoints"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = REPO_ROOT / ".cache" / "layout-flow" / "converted"
CHECKPOINT_NAMES: Final[dict[DatasetName, str]] = {
    DatasetName.publaynet: "checkpoint_PubLayNet_LayoutFlow.ckpt",
    DatasetName.rico25: "checkpoint_RICO_LayoutFlow.ckpt",
}


def default_checkpoint(dataset: DatasetName) -> Path:
    return DEFAULT_ORIGINAL_DIR / CHECKPOINT_NAMES[dataset]


def default_output_dir(dataset: DatasetName) -> Path:
    return DEFAULT_OUTPUT_ROOT / str(dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one original LayoutFlow Lightning checkpoint into a local "
            "Diffusers-compatible pipeline directory with a generated README.md "
            "model card."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(str(dataset) for dataset in CHECKPOINT_NAMES),
        default="publaynet",
        help="Dataset/checkpoint variant to convert.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the original .ckpt file. When omitted, the default cache "
            "path for --dataset is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the converted pipeline. When omitted, a dataset-named "
            "folder under the default conversion cache is used."
        ),
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path("vendor/layout-flow"),
        help="Path to the read-only original LayoutFlow source checkout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset = normalize_dataset_name(args.dataset)
    checkpoint = args.checkpoint or default_checkpoint(dataset)
    output_dir = args.output_dir or default_output_dir(dataset)
    vendor_dir = vendor_root(
        "layout-flow",
        marker=Path("src/models/backbone/layoutdm_backbone.py"),
        path=args.vendor_dir,
    )
    sys.path.insert(0, str(vendor_dir))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = raw.get("state_dict", raw)
    config = LayoutFlowConfig(dataset_name=str(dataset))
    pipe = build_pipeline(config)
    missing, unexpected = pipe.model.load_state_dict(
        convert_lightning_state_dict(state_dict), strict=False
    )
    if unexpected:
        raise RuntimeError(f"Unexpected converted keys: {unexpected}")
    model_missing = [key for key in missing if not key.startswith("_")]
    if model_missing:
        raise RuntimeError(f"Missing model keys: {model_missing}")
    pipe.save_pretrained(output_dir)
    save_layoutflow_model_card(output_dir, dataset=str(dataset))


if __name__ == "__main__":
    main()
