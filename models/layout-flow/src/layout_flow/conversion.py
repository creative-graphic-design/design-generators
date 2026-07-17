"""Checkpoint conversion helpers for LayoutFlow."""

from __future__ import annotations

import torch

from .configuration_layout_flow import LayoutFlowConfig
from .modeling_layout_flow import LayoutFlowTransformerModel
from .pipeline_layout_flow import LayoutFlowPipeline
from .scheduling_layout_flow import LayoutFlowEulerScheduler


def build_pipeline(config: LayoutFlowConfig) -> LayoutFlowPipeline:
    """Build a randomly initialized pipeline for a LayoutFlow config.

    Args:
        config: LayoutFlow configuration.

    Returns:
        Pipeline with model and scheduler modules initialized from ``config``.

    Examples:
        >>> pipe = build_pipeline(LayoutFlowConfig(max_length=2, latent_dim=8, d_model=16))
        >>> pipe.layout_flow_config.max_length
        2
    """
    model = LayoutFlowTransformerModel(
        num_labels=config.num_labels,
        latent_dim=config.latent_dim,
        tr_enc_only=config.tr_enc_only,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_pos_enc=config.use_pos_enc,
        attr_encoding=config.attr_encoding,
        seq_type=config.seq_type,
    )
    scheduler = LayoutFlowEulerScheduler(num_inference_steps=config.inference_steps)
    return LayoutFlowPipeline(model=model, scheduler=scheduler, config=config)


def convert_lightning_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert original Lightning checkpoint keys to local model keys.

    Args:
        state_dict: Original LayoutFlow Lightning state dict.

    Returns:
        State dict keyed for ``LayoutFlowTransformerModel``.

    Examples:
        >>> import torch
        >>> out = convert_lightning_state_dict({"model.linear.weight": torch.zeros(1)})
        >>> list(out)
        ['backbone.linear.weight']
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            converted[f"backbone.{key.removeprefix('model.')}"] = value
    return converted
