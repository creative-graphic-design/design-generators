"""Convert a local RADM checkpoint into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - conversion scripts inspect arbitrary checkpoint payloads.

import torch

from radm.configuration_radm import RADMConfig
from radm.conversion import build_pipeline, convert_original_state_dict
from radm.model_card import write_local_model_card


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a user-supplied RADM checkpoint into a local Diffusers "
            "pipeline directory. No checkpoint is downloaded or redistributed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Local checkpoint path."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Converted output directory."
    )
    parser.add_argument("--dataset-name", default="cgl", help="Dataset metadata name.")
    parser.add_argument(
        "--use-ema", action="store_true", help="Use ema_state when present."
    )
    return parser


def _state_dict(payload: dict[str, Any], *, use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema and isinstance(payload.get("ema_state"), dict):
        return cast(dict[str, torch.Tensor], payload["ema_state"])
    if isinstance(payload.get("model"), dict):
        return cast(dict[str, torch.Tensor], payload["model"])
    if isinstance(payload.get("state_dict"), dict):
        return cast(dict[str, torch.Tensor], payload["state_dict"])
    return cast(dict[str, torch.Tensor], payload)


def main() -> None:
    """Convert and save one local RADM pipeline."""
    args = build_parser().parse_args()
    payload = cast(dict[str, Any], torch.load(args.checkpoint, map_location="cpu"))
    config = RADMConfig(dataset_name=args.dataset_name)
    pipe = build_pipeline(config)
    converted = convert_original_state_dict(_state_dict(payload, use_ema=args.use_ema))
    missing, unexpected = pipe.denoiser.load_state_dict(converted, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected converted keys: {unexpected}")
    missing_public = [key for key in missing if not key.startswith("_")]
    if missing_public:
        raise RuntimeError(f"Missing denoiser keys: {missing_public}")
    pipe.save_pretrained(args.output_dir)
    write_local_model_card(args.output_dir, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
