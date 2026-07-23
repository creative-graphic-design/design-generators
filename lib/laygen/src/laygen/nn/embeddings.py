"""Embedding modules shared by layout-generation models.

``SinusoidalPosEmb`` follows the VQ-Diffusion-derived timestep embedding used
by the LayoutDM and LACE checkpoint backbones. ``ElementPositionalEmbedding`` is a
LayoutDM-specific element/attribute position embedding.
"""

from __future__ import annotations

import math
from enum import StrEnum, auto

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn


class TimestepEmbeddingType(StrEnum):
    """Supported timestep-conditioned normalization variants.

    Origin:
        These names come from VQ-Diffusion-derived adaptive normalization modes
        used by the LayoutDM and LACE checkpoint backbones.
    """

    adalayernorm = auto()
    adainnorm = auto()
    adalayernorm_abs = auto()
    adainnorm_abs = auto()
    adalayernorm_mlp = auto()
    adainnorm_mlp = auto()


def normalize_timestep_embedding(
    timestep_type: TimestepEmbeddingType | str | None,
) -> TimestepEmbeddingType | None:
    """Normalize a timestep embedding mode.

    Origin:
        This normalizes the VQ-Diffusion-derived adaptive normalization mode
        names exposed by LayoutDM and LACE checkpoints.

    Args:
        timestep_type: Embedding enum, string value, or ``None``.

    Returns:
        Canonical embedding enum or ``None``.

    Raises:
        ValueError: If the embedding mode is unsupported.
    """
    if timestep_type is None or isinstance(timestep_type, TimestepEmbeddingType):
        return timestep_type
    try:
        return TimestepEmbeddingType(timestep_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestep_type: {timestep_type}") from exc


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep or position embedding.

    Origin:
        This is the VQ-Diffusion-style sinusoidal timestep embedding carried by
        LayoutDM and LACE. The checkpoint operation order is preserved exactly
        because LACE denoiser parity is bit-sensitive at ``rescale_steps=4000``.

    Args:
        num_steps: Maximum number of positions or timesteps.
        dim: Embedding dimension. Odd dimensions keep the checkpoint truncation
            behavior and return ``2 * floor(dim / 2)`` channels.
        rescale_steps: Rescaling constant used by the released checkpoints.
    """

    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000) -> None:
        """Initialize the embedding parameters."""
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(
        self, x: Int[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch channels"]:
        """Embed integer positions or timesteps.

        Args:
            x: One-dimensional tensor of positions.

        Returns:
            Sinusoidal embedding tensor.
        """
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ElementPositionalEmbedding(nn.Module):
    """Learned element and attribute positional embedding.

    Origin:
        This learned element/attribute positional embedding is specific to
        CyberAgentAILab LayoutDM and is reused by Layout-Corrector.

    Args:
        dim_model: Embedding dimension.
        max_token_length: Maximum flattened token sequence length.
        n_attr_per_elem: Number of attributes per layout element.
    """

    def __init__(
        self, dim_model: int, max_token_length: int, n_attr_per_elem: int = 5
    ) -> None:
        """Initialize element and attribute embedding parameters."""
        super().__init__()
        self.n_elem = max_token_length // n_attr_per_elem
        self.n_attr_per_elem = n_attr_per_elem
        self.elem_emb = nn.Parameter(torch.rand(self.n_elem, dim_model))
        self.attr_emb = nn.Parameter(torch.rand(self.n_attr_per_elem, dim_model))

    def forward(
        self, h: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "batch tokens channels"]:
        """Return positional embeddings matching hidden-state length.

        Args:
            h: Hidden states shaped ``(batch, sequence, dim)``.

        Returns:
            Positional embedding tensor shaped like ``h``.
        """
        batch, seq_len = h.shape[:2]
        elem_emb = repeat(self.elem_emb, "s d -> (s x) d", x=self.n_attr_per_elem)
        attr_emb = repeat(self.attr_emb, "x d -> (s x) d", s=self.n_elem)
        emb = (elem_emb + attr_emb)[:seq_len]
        return repeat(emb, "s d -> b s d", b=batch)

    @property
    def no_decay_param_names(self) -> list[str]:
        """Return parameter names that should skip weight decay."""
        return ["elem_emb", "attr_emb"]
