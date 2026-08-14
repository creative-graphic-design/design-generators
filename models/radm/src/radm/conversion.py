"""Checkpoint conversion helpers for RADM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import torch
from jaxtyping import Float

from .configuration_radm import RADMConfig, RADM_DIFFUSION_STATE_KEYS
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
        >>> config = RADMConfig(num_proposals=2, hidden_dim=8, text_feature_dim=4, backbone_depth=18)
        >>> pipe = build_pipeline(config)
        >>> pipe.radm_config.num_proposals
        2
    """
    denoiser = RADMDenoiser(
        config=config,
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
    unsupported: list[str] = []
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
        else:
            try:
                converted[_reference_key_to_package_key(key)] = value
            except AssertionError:
                unsupported.append(key)
    if not converted:
        raise RuntimeError(
            "No RADM denoiser keys were found in the checkpoint. "
            "Pass a checkpoint with denoiser/model.denoiser/module.denoiser keys "
            "or update the conversion mapping after inspecting the original state."
        )
    if unsupported:
        raise RuntimeError(
            "Unsupported original RADM state keys; refusing a partial conversion: "
            + ", ".join(sorted(unsupported)[:8])
        )
    return converted


def _reference_key_to_package_key(key: str) -> str:
    """Apply the inverse of the reviewed source namespace rules."""
    if key in RADM_DIFFUSION_STATE_KEYS:
        return key
    if key.startswith("head.head_series."):
        return key.replace("head.head_series.", "head.blocks.", 1)
    fpn_rules = (
        ("backbone.fpn_lateral2.", "backbone.body.fpn.inner_blocks.0.0."),
        ("backbone.fpn_lateral3.", "backbone.body.fpn.inner_blocks.1.0."),
        ("backbone.fpn_lateral4.", "backbone.body.fpn.inner_blocks.2.0."),
        ("backbone.fpn_lateral5.", "backbone.body.fpn.inner_blocks.3.0."),
        ("backbone.fpn_output2.", "backbone.body.fpn.layer_blocks.0.0."),
        ("backbone.fpn_output3.", "backbone.body.fpn.layer_blocks.1.0."),
        ("backbone.fpn_output4.", "backbone.body.fpn.layer_blocks.2.0."),
        ("backbone.fpn_output5.", "backbone.body.fpn.layer_blocks.3.0."),
    )
    for reference_prefix, package_prefix in fpn_rules:
        if key.startswith(reference_prefix):
            return package_prefix + key[len(reference_prefix) :]
    if key.startswith("backbone.bottom_up."):
        package = key.replace("backbone.bottom_up.", "backbone.body.body.", 1)
        package = package.replace(
            "backbone.body.body.stem.conv1.norm.", "backbone.body.body.bn1.", 1
        )
        package = package.replace(
            "backbone.body.body.stem.conv1.", "backbone.body.body.conv1.", 1
        )
        for reference_stage, package_layer in (
            ("res2", "layer1"),
            ("res3", "layer2"),
            ("res4", "layer3"),
            ("res5", "layer4"),
        ):
            package = package.replace(
                f"backbone.body.body.{reference_stage}.",
                f"backbone.body.body.{package_layer}.",
                1,
            )
        package = package.replace(".shortcut.norm.", ".downsample.1.")
        package = package.replace(".shortcut.", ".downsample.0.")
        package = package.replace(".conv1.norm.", ".bn1.")
        package = package.replace(".conv2.norm.", ".bn2.")
        package = package.replace(".conv3.norm.", ".bn3.")
        return package
    raise AssertionError(f"Unmapped original RADM topology key: {key}")


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
