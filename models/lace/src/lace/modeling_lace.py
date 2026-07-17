"""Transformer denoiser used by converted LACE checkpoints."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Callable

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops.layers.torch import Rearrange
from torch import nn


@dataclass
class LaceModelOutput(BaseOutput):
    """Output returned by the LACE transformer.

    Attributes:
        sample: Predicted noise tensor with the same shape as the input sample.
    """

    sample: torch.FloatTensor


class ActivationName(StrEnum):
    """Supported feed-forward activation names."""

    relu = auto()
    gelu = auto()
    gelu2 = auto()


class TimestepEmbeddingType(StrEnum):
    """Supported timestep-conditioned normalization variants."""

    adalayernorm = auto()
    adainnorm = auto()
    adalayernorm_abs = auto()
    adainnorm_abs = auto()
    adalayernorm_mlp = auto()
    adainnorm_mlp = auto()


ActivationFn = Callable[[torch.Tensor], torch.Tensor]


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return x * F.sigmoid(1.702 * x)


def normalize_activation(
    name: ActivationName | str | ActivationFn,
) -> ActivationName | ActivationFn:
    """Normalize an activation name while preserving custom callables.

    Args:
        name: Activation enum, string value, or callable.

    Returns:
        Canonical activation enum or the original callable.

    Raises:
        ValueError: If the activation name is unsupported.
    """
    if callable(name):
        return name
    try:
        return ActivationName(name)
    except ValueError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def normalize_timestep_embedding(
    timestep_type: TimestepEmbeddingType | str | None,
) -> TimestepEmbeddingType | None:
    """Normalize a timestep embedding mode.

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


def _activation(name: ActivationName | str | ActivationFn) -> ActivationFn:
    canonical = normalize_activation(name)
    if callable(canonical):
        return canonical
    if canonical is ActivationName.relu:
        return F.relu
    if canonical is ActivationName.gelu:
        return F.gelu
    if canonical is ActivationName.gelu2:
        return _gelu2
    raise AssertionError(f"Unhandled activation: {canonical}")


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep or position embedding."""

    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000) -> None:
        """Initialize the embedding table parameters.

        Args:
            num_steps: Maximum number of positions or timesteps.
            dim: Embedding dimension.
            rescale_steps: Vendor rescaling constant.
        """
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class _AdaNorm(nn.Module):
    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
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
    """Adaptive layer normalization conditioned on diffusion timestep."""

    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        """Initialize adaptive layer normalization.

        Args:
            n_embd: Hidden dimension.
            max_timestep: Maximum diffusion timestep.
            emb_type: Timestep embedding variant.
        """
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Apply timestep-conditioned layer normalization."""
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class AdaInsNorm(_AdaNorm):
    """Adaptive instance normalization conditioned on diffusion timestep."""

    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: TimestepEmbeddingType = TimestepEmbeddingType.adalayernorm_abs,
    ) -> None:
        """Initialize adaptive instance normalization.

        Args:
            n_embd: Hidden dimension.
            max_timestep: Maximum diffusion timestep.
            emb_type: Timestep embedding variant.
        """
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


class Block(nn.Module):
    """Transformer block used by the LACE denoiser.

    Args:
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feed-forward hidden dimension.
        dropout: Dropout probability.
        activation: Feed-forward activation name or callable.
        batch_first: Whether inputs use ``(batch, seq, dim)`` order.
        norm_first: Whether to use pre-norm residual blocks.
        diffusion_step: Maximum diffusion timestep.
        timestep_type: Timestep-conditioned normalization variant.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: ActivationName | str | ActivationFn = ActivationName.relu,
        batch_first: bool = True,
        norm_first: bool = True,
        diffusion_step: int = 100,
        timestep_type: TimestepEmbeddingType
        | str
        | None = TimestepEmbeddingType.adalayernorm,
    ) -> None:
        """Initialize a LACE transformer block."""
        super().__init__()
        self.norm_first = norm_first
        self.diffusion_step = diffusion_step
        canonical_timestep = normalize_timestep_embedding(timestep_type)
        self.timestep_type = canonical_timestep
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if canonical_timestep is None:
            self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        elif "adalayernorm" in canonical_timestep:
            self.norm1 = AdaLayerNorm(d_model, diffusion_step, canonical_timestep)
        elif "adainnorm" in canonical_timestep:
            self.norm1 = AdaInsNorm(d_model, diffusion_step, canonical_timestep)
        else:
            raise AssertionError(f"Unhandled timestep_type: {canonical_timestep}")
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention and feed-forward layers.

        Args:
            src: Input sequence.
            src_mask: Optional attention mask.
            src_key_padding_mask: Optional padding mask.
            timestep: Diffusion timestep tensor for adaptive normalization.

        Returns:
            Transformed sequence.
        """
        x = src
        if self.norm_first:
            x = self.norm1(x, timestep) if self.timestep_type else self.norm1(x)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
            return x
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = self.norm1(x, timestep) if self.timestep_type else self.norm1(x)
        return self.norm2(x + self._ff_block(x))

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


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
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.layer_out = nn.Linear(dim_transformer, seq_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> LaceModelOutput | tuple[torch.Tensor]:
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
