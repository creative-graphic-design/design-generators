"""Checkpoint conversion helpers for vendor LACE weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from laygen.common.labels import DatasetName

from .configuration_lace import default_model_config
from .modeling_lace import LaceTransformerModel
from .pipeline_lace import LacePipeline
from .processing_lace import LaceProcessor
from .scheduling_lace import LaceScheduler


def load_vendor_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint as a state dictionary.

    Args:
        path: Path to a vendor checkpoint or state-dict file.

    Returns:
        Mapping from parameter names to tensors.

    Raises:
        TypeError: If the checkpoint does not contain a state-dict-like object.
    """
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a state_dict-like checkpoint at {path}")
    return dict(loaded)


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert vendor parameter names to the Diffusers module names.

    Args:
        state_dict: Vendor state dictionary.

    Returns:
        Converted state dictionary with distributed prefixes and vendor
        positional buffers removed.
    """
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if key != "pos_embed" and key.removeprefix("module.") != "pos_embed"
    }


def build_pipeline_from_vendor_checkpoint(
    dataset: DatasetName | str,
    checkpoint_path: str | Path,
    ddim_num_steps: int = 100,
) -> LacePipeline:
    """Build a LACE pipeline from a vendor checkpoint.

    Args:
        dataset: LACE dataset name or alias.
        checkpoint_path: Path to the vendor checkpoint.
        ddim_num_steps: Number of DDIM inference steps configured on the scheduler.

    Returns:
        Pipeline containing converted model, scheduler, and processor.

    Raises:
        TypeError: If the checkpoint payload is not a state dictionary.
        ValueError: If converted keys do not match the model architecture.
    """
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
