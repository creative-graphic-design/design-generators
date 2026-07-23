"""Conversion helpers for CGB-DM checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from .configuration_cgb_dm import CGBDMConfig
from .modeling_cgb_dm import CGBDMTransformerModel
from .pipeline_cgb_dm import CGBDMPipeline
from .processing_cgb_dm import CGBDMProcessor
from .scheduling_cgb_dm import CGBDMScheduler


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize CGB-DM checkpoint keys for ``CGBDMTransformerModel``.

    Args:
        state_dict: Original or Lightning checkpoint state dictionary.

    Returns:
        Converted state dictionary with common wrapper prefixes stripped.

    Examples:
        >>> convert_state_dict({"model.module.img_encoder.patch.weight": torch.zeros(1)})["img_encoder.patch.weight"].shape
        torch.Size([1])
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key.removeprefix("state_dict.")
        name = name.removeprefix("model.")
        name = name.removeprefix("module.")
        name = name.removeprefix("denoiser.")
        converted[name] = value
    return converted


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state-dict-like checkpoint from disk."""
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, Mapping):
        raw = checkpoint.get("state_dict", checkpoint)
        if isinstance(raw, Mapping):
            return {str(k): cast(torch.Tensor, v) for k, v in raw.items()}
    raise TypeError("Expected a state_dict-like checkpoint")


def build_model_from_config(config: CGBDMConfig) -> CGBDMTransformerModel:
    """Build the CGB-DM denoiser shape described by ``config``."""
    return CGBDMTransformerModel(
        num_labels=config.num_labels,
        max_seq_length=config.max_seq_length,
        image_size=config.image_size,
        dim_model=config.dim_model,
        n_head=config.n_head,
        feature_dim=config.feature_dim,
        num_layers=config.num_layers,
        num_train_timesteps=config.num_train_timesteps,
    )


def build_pipeline_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: CGBDMConfig,
) -> CGBDMPipeline:
    """Build a CGB-DM pipeline from a package-local training checkpoint.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint.
        config: CGB-DM config that matches the training checkpoint.

    Returns:
        Pipeline with converted model weights loaded.

    Raises:
        ValueError: If the checkpoint keys do not match the model.
    """
    model = build_model_from_config(config)
    converted = convert_state_dict(load_state_dict(checkpoint_path))
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected:
        raise ValueError(f"State dict mismatch: unexpected={unexpected}")
    if len(missing) == len(model.state_dict()):
        raise ValueError("State dict mismatch: no CGB-DM model keys matched")
    return CGBDMPipeline(
        model=model,
        scheduler=CGBDMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            ddim_num_steps=config.ddim_num_steps,
            train_beta_schedule=config.train_beta_schedule,
            sampling_beta_schedule=config.sampling_beta_schedule,
        ),
        processor=CGBDMProcessor(
            dataset_name=config.dataset_name,
            id2label=config.id2label,
            num_labels=config.num_labels,
            max_seq_length=config.max_seq_length,
            image_size=config.image_size,
        ),
    )
