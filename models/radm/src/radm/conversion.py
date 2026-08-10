"""Checkpoint conversion helpers for RADM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import torch
from jaxtyping import Float

from .configuration_radm import RADMConfig
from .modeling_radm import RADMDenoiser
from .pipeline_radm import RADMPipeline
from .processing_radm import RADMProcessor
from .scheduling_radm import RADMScheduler

CheckpointPayloadValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | Sequence["CheckpointPayloadValue"]
    | Mapping[str, "CheckpointPayloadValue"]
)


def build_pipeline(config: RADMConfig) -> RADMPipeline:
    """Build a randomly initialized RADM pipeline for a config.

    Args:
        config: RADM configuration.

    Returns:
        Pipeline with denoiser, scheduler, and processor components.

    Examples:
        >>> pipe = build_pipeline(RADMConfig(num_proposals=2, hidden_dim=8, text_feature_dim=4))
        >>> pipe.radm_config.num_proposals
        2
    """
    denoiser = RADMDenoiser(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        text_feature_dim=config.text_feature_dim,
    )
    scheduler = RADMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        num_inference_steps=config.inference_steps,
    )
    return RADMPipeline(
        denoiser=denoiser,
        scheduler=scheduler,
        config=config,
        processor=RADMProcessor(config=config),
    )


def inspect_checkpoint_payload(
    payload: Mapping[
        str,
        CheckpointPayloadValue
        | Float[torch.Tensor, "..."]
        | Mapping[str, Float[torch.Tensor, "..."]],
    ],
) -> dict[str, int | bool | list[str]]:
    """Summarize a Detectron2-style RADM checkpoint payload.

    Args:
        payload: Loaded checkpoint mapping.

    Returns:
        Metadata summary with root keys and tensor counts.
    """
    model_state = _find_state_dict(payload)
    return {
        "root_keys": sorted(str(key) for key in payload),
        "state_dict_keys": len(model_state),
        "has_model": "model" in payload,
        "has_ema_state": "ema_state" in payload,
        "has_optimizer": "optimizer" in payload,
        "has_scheduler": "scheduler" in payload,
    }


def convert_original_state_dict(
    state_dict: Mapping[str, Float[torch.Tensor, "..."]],
) -> dict[str, Float[torch.Tensor, "..."]]:
    """Convert supported RADM checkpoint keys to local denoiser keys.

    Args:
        state_dict: Original checkpoint state dict.

    Returns:
        State dict keyed for ``RADMDenoiser``.

    Raises:
        RuntimeError: If no supported RADM component keys are present.
    """
    converted: dict[str, Float[torch.Tensor, "..."]] = {}
    prefixes = {
        "denoiser.": "",
        "model.denoiser.": "",
        "module.denoiser.": "",
        "radm.denoiser.": "",
    }
    for key, value in state_dict.items():
        for prefix, replacement in prefixes.items():
            if key.startswith(prefix):
                converted[f"{replacement}{key.removeprefix(prefix)}"] = value
                break
    if not converted:
        raise RuntimeError(
            "No RADM denoiser keys were found in the checkpoint. "
            "Pass a checkpoint with denoiser/model.denoiser/module.denoiser keys "
            "or update the conversion mapping after inspecting the original state."
        )
    return converted


def _find_state_dict(
    payload: Mapping[
        str,
        CheckpointPayloadValue
        | Float[torch.Tensor, "..."]
        | Mapping[str, Float[torch.Tensor, "..."]],
    ],
) -> Mapping[str, Float[torch.Tensor, "..."]]:
    for key in ("model", "state_dict", "ema_state"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return cast(Mapping[str, Float[torch.Tensor, "..."]], value)
    return cast(Mapping[str, Float[torch.Tensor, "..."]], payload)
