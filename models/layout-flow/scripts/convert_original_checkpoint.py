from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from layout_flow.configuration_layout_flow import LayoutFlowConfig
from layout_flow.conversion import build_pipeline, convert_lightning_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["rico25", "publaynet"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path)
    args = parser.parse_args()

    vendor_dir = (
        args.vendor_dir or Path(__file__).resolve().parents[3] / "vendor/layout-flow"
    )
    sys.path.insert(0, str(vendor_dir.resolve()))
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    pipe.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
