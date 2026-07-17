"""Convert original LayoutFlow Lightning checkpoints to Diffusers pipeline folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from layout_flow.configuration_layout_flow import LayoutFlowConfig
from layout_flow.conversion import build_pipeline, convert_lightning_state_dict
from layout_flow.model_card import save_layoutflow_model_card


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORIGINAL_DIR = REPO_ROOT / ".cache" / "layout-flow" / "original" / "checkpoints"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".cache" / "layout-flow" / "converted"
CHECKPOINT_NAMES = {
    "publaynet": "checkpoint_PubLayNet_LayoutFlow.ckpt",
    "rico25": "checkpoint_RICO_LayoutFlow.ckpt",
}


def default_checkpoint(dataset: str) -> Path:
    return DEFAULT_ORIGINAL_DIR / CHECKPOINT_NAMES[dataset]


def default_output_dir(dataset: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / dataset


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
        choices=sorted(CHECKPOINT_NAMES),
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
        default=REPO_ROOT / "vendor" / "layout-flow",
        help="Path to the read-only original LayoutFlow source checkout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = args.checkpoint or default_checkpoint(args.dataset)
    output_dir = args.output_dir or default_output_dir(args.dataset)
    vendor_dir = args.vendor_dir
    sys.path.insert(0, str(vendor_dir.resolve()))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = raw.get("state_dict", raw)
    config = LayoutFlowConfig(dataset_name=args.dataset)
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
    save_layoutflow_model_card(output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
