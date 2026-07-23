"""Transformer denoiser used by converted LACE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from jaxtyping import Bool, Float, Int
from torch import nn
from laygen.nn import (
    ActivationFn,
    ActivationName,
    AdaInsNorm,
    AdaLayerNorm,
    Block,
    SinusoidalPosEmb,
    TimestepEmbeddingType,
    clone_module_list,
    get_activation,
    normalize_activation,
    normalize_timestep_embedding,
)

__all__ = [
    "ActivationFn",
    "ActivationName",
    "AdaInsNorm",
    "AdaLayerNorm",
    "Block",
    "LaceModelOutput",
    "LaceTransformerModel",
    "SinusoidalPosEmb",
    "TimestepEmbeddingType",
    "_activation",
    "_gelu2",
    "_get_clones",
    "normalize_activation",
    "normalize_timestep_embedding",
]


@dataclass
class LaceModelOutput(BaseOutput):
    """Output returned by the LACE transformer.

    Attributes:
        sample: Predicted noise tensor with the same shape as the input sample.
    """

    sample: Float[torch.Tensor, "batch elements channels"]


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return clone_module_list(module, n)


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return get_activation("gelu2")(x)


def _activation(name: ActivationName | str | ActivationFn) -> ActivationFn:
    return get_activation(name)


class LaceTransformerModel(ModelMixin, ConfigMixin):
    """Transformer denoiser for continuous LACE layout tensors.

    Args:
        seq_dim: Number of channels per layout element.
        max_seq_length: Maximum number of layout elements.
        num_layers: Number of transformer blocks.
        dim_transformer: Hidden dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feed-forward hidden dimension.
        diffusion_step: Maximum diffusion timestep.
        timestep_type: Timestep-conditioned normalization variant.
    """

    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        seq_dim: int,
        max_seq_length: int = 25,
        num_layers: int = 4,
        dim_transformer: int = 512,
        nhead: int = 16,
        dim_feedforward: int = 2048,
        diffusion_step: int = 1000,
        timestep_type: TimestepEmbeddingType
        | str
        | None = TimestepEmbeddingType.adalayernorm,
    ) -> None:
        """Initialize a LACE transformer denoiser."""
        super().__init__()
        self.seq_dim = seq_dim
        self.max_seq_length = max_seq_length
        self.pos_encoder = SinusoidalPosEmb(max_seq_length, dim_transformer)
        pos_i = torch.arange(max_seq_length)
        self.register_buffer("pos_embed", self.pos_encoder(pos_i), persistent=False)
        self.layer_in = nn.Linear(seq_dim, dim_transformer)
        encoder_layer = Block(
            d_model=dim_transformer,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
        )
        self.layers = clone_module_list(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.layer_out = nn.Linear(dim_transformer, seq_dim)

    def forward(
        self,
        sample: Float[torch.Tensor, "batch elements channels"],
        timestep: Int[torch.Tensor, "batch"],
        attention_mask: Bool[torch.Tensor, "batch elements"] | None = None,
        return_dict: bool = True,
    ) -> LaceModelOutput | tuple[Float[torch.Tensor, "batch elements channels"]]:
        """Predict denoising residuals for a layout sample.

        Args:
            sample: Noisy layout tensor.
            timestep: Diffusion timestep per sample.
            attention_mask: Optional valid-element mask.
            return_dict: Whether to return ``LaceModelOutput``.

        Returns:
            Output dataclass or a one-item tuple containing the prediction.
        """
        output = F.softplus(self.layer_in(sample))
        pos_i = torch.arange(output.shape[1], device=output.device)
        output = output + self.pos_encoder(pos_i).to(output)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_key_padding_mask=key_padding_mask,
                timestep=timestep,
            )
            if i < self.num_layers - 1:
                output = F.softplus(output)
        output = self.layer_out(output)
        if not return_dict:
            return (output,)
        return LaceModelOutput(sample=output)
