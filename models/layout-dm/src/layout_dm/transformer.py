"""Transformer building blocks used by the LayoutDM denoiser."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import torch
from jaxtyping import Int
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
