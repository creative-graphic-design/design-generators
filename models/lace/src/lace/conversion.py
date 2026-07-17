from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .configuration_lace import default_model_config
from .modeling_lace import LaceTransformerModel
from .pipeline_lace import LacePipeline
from .processing_lace import LaceProcessor
from .scheduling_lace import LaceScheduler


def load_vendor_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a state_dict-like checkpoint at {path}")
    return dict(loaded)


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if key != "pos_embed" and key.removeprefix("module.") != "pos_embed"
    }


def build_pipeline_from_vendor_checkpoint(
    dataset: str,
    checkpoint_path: str | Path,
    ddim_num_steps: int = 100,
) -> LacePipeline:
    model = LaceTransformerModel(**default_model_config(dataset))
    converted = convert_state_dict(load_vendor_state_dict(checkpoint_path))
    expected = set(model.state_dict())
    actual = set(converted)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"State dict mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.load_state_dict(converted, strict=True)
    processor = LaceProcessor.from_dataset(dataset)
    scheduler = LaceScheduler(ddim_num_steps=ddim_num_steps)
    return LacePipeline(model=model, scheduler=scheduler, processor=processor)
