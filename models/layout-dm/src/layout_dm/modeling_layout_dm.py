"""Modeling components for converted LayoutDM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from jaxtyping import Float, Int
from torch import nn

from laygen.nn import (
    AdaLayerNorm,
    Block,
    ElementPositionalEmbedding,
    SinusoidalPosEmb,
    TransformerEncoder,
    clone_module_list,
    get_activation,
)

TimestepType: TypeAlias = Literal["adalayernorm", "adalayernorm_abs"]

__all__ = [
    "AdaLayerNorm",
    "Block",
    "CategoricalTransformer",
    "ElementPositionalEmbedding",
    "LayoutDMDenoiser",
    "LayoutDMDenoiserOutput",
    "SinusoidalPosEmb",
    "TimestepType",
    "TransformerEncoder",
    "_activation",
    "_gelu2",
    "_get_clones",
]


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return clone_module_list(module, n)


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return get_activation("gelu2")(x)


def _activation(
    name: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    return get_activation(name)


class CategoricalTransformer(nn.Module):
    """Token transformer that predicts LayoutDM categorical logits."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_token_length: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        dropout: float = 0.0,
        timestep_type: TimestepType | None = "adalayernorm",
    ) -> None:
        """Initialize the categorical transformer denoiser backbone."""
        super().__init__()
        layer = Block(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            diffusion_step=100,
            timestep_type=timestep_type,
        )
        self.backbone = TransformerEncoder(layer, num_hidden_layers)
        self.cat_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = ElementPositionalEmbedding(
            hidden_size, max_token_length, n_attr_per_elem=5
        )
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, vocab_size, bias=False)
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timestep: Int[torch.Tensor, "batch"] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict logits for flattened LayoutDM token ids."""
        hidden = self.cat_emb(input_ids)
        hidden = self.drop(hidden + self.pos_emb(hidden))
        hidden = self.backbone(hidden, timestep=timestep)
        return {"logits": self.head(hidden)}


@dataclass
class LayoutDMDenoiserOutput(BaseOutput):
    """Denoiser output containing token logits."""

    logits: Float[torch.Tensor, "batch tokens vocab"]


class LayoutDMDenoiser(ModelMixin, ConfigMixin):
    """Diffusers-compatible LayoutDM denoiser.

    Args:
        vocab_size: Size of the LayoutDM tokenizer vocabulary.
        max_token_length: Flattened token sequence length.
        hidden_size: Transformer hidden size.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer layers.
        intermediate_size: Feed-forward hidden size.
        dropout: Dropout probability.
        timestep_type: Timestep-conditioning type.

    Examples:
        >>> model = LayoutDMDenoiser(vocab_size=10, max_token_length=5, hidden_size=8,
        ...     num_attention_heads=2, num_hidden_layers=1, intermediate_size=16)
        >>> model.config.vocab_size
        10
    """

    config_name = "denoiser_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        vocab_size: int,
        max_token_length: int,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: TimestepType | None = "adalayernorm",
    ) -> None:
        """Initialize the categorical transformer denoiser."""
        super().__init__()
        self.transformer = CategoricalTransformer(
            vocab_size=vocab_size,
            max_token_length=max_token_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
            timestep_type=timestep_type,
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> LayoutDMDenoiserOutput:
        """Predict token logits for noised LayoutDM sequences."""
        return LayoutDMDenoiserOutput(
            logits=self.transformer(input_ids, timestep=timesteps)["logits"]
        )

    def predict_start_log_probs(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch tokens vocab"]:
        """Predict log probabilities for the denoised start sequence."""
        logits = self(input_ids=input_ids, timesteps=timesteps).logits[:, :, :-1]
        log_pred = F.log_softmax(logits.double(), dim=-1).float()
        zero_mask = torch.full(
            (*log_pred.shape[:2], 1),
            -70.0,
            device=log_pred.device,
            dtype=log_pred.dtype,
        )
        return torch.cat((log_pred, zero_mask), dim=-1).clamp(-70.0, 0.0)
