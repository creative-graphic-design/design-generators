"""Adaptive normalization layers shared by layout-generation models."""

from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn

from .embeddings import SinusoidalPosEmb, TimestepEmbeddingType


class _AdaNorm(nn.Module):
    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType | str = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        super().__init__()
        mode = str(emb_type)
        if "abs" in mode:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in mode:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    """Adaptive layer normalization conditioned on diffusion timestep.

    Args:
        n_embd: Hidden dimension.
        max_timestep: Maximum diffusion timestep.
        emb_type: Timestep embedding variant.
    """

    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType | str = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        """Initialize adaptive layer normalization."""
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Apply timestep-conditioned layer normalization."""
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class AdaInsNorm(_AdaNorm):
    """Adaptive instance normalization conditioned on diffusion timestep.

    Args:
        n_embd: Hidden dimension.
        max_timestep: Maximum diffusion timestep.
        emb_type: Timestep embedding variant.
    """

    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType | str = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        """Initialize adaptive instance normalization."""
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Apply timestep-conditioned instance normalization."""
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
