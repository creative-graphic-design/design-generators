"""Transformer encoder blocks shared by layout-generation models.

The timestep-aware encoder layer follows Microsoft VQ-Diffusion's
``transformer_utils.Block`` structure as carried by LayoutDM, LACE, and
Layout-Corrector vendor code.
"""

from __future__ import annotations

from typing import assert_never

from jaxtyping import Bool, Float, Int
from torch import nn
from torch import Tensor

from .activations import ActivationFn, ActivationName, get_activation
from .embeddings import TimestepEmbeddingType, normalize_timestep_embedding
from .module_utils import clone_module_list
from .norms import AdaInsNorm, AdaLayerNorm


class TimestepTransformerEncoderLayer(nn.Module):
    """Transformer encoder block with optional adaptive normalization.

    Origin:
        This class follows VQ-Diffusion ``Block`` rather than Diffusers
        ``BasicTransformerBlock`` so checkpoint keys and adaptive norm call
        conventions stay compatible with LayoutDM, LACE, and Layout-Corrector.

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
        """Initialize the transformer encoder block."""
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
        match canonical_timestep:
            case None:
                self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
            case (
                TimestepEmbeddingType.adalayernorm
                | TimestepEmbeddingType.adalayernorm_abs
                | TimestepEmbeddingType.adalayernorm_mlp
            ):
                self.norm1 = AdaLayerNorm(d_model, diffusion_step, canonical_timestep)
            case (
                TimestepEmbeddingType.adainnorm
                | TimestepEmbeddingType.adainnorm_abs
                | TimestepEmbeddingType.adainnorm_mlp
            ):
                self.norm1 = AdaInsNorm(d_model, diffusion_step, canonical_timestep)
            case _:
                assert_never(canonical_timestep)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(
        self,
        src: Float[Tensor, "batch tokens channels"],
        src_mask: Bool[Tensor, "..."] | None = None,
        src_key_padding_mask: Bool[Tensor, "batch tokens"] | None = None,
        timestep: Int[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch tokens channels"]:
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
        x: Float[Tensor, "batch tokens channels"],
        attn_mask: Bool[Tensor, "..."] | None,
        key_padding_mask: Bool[Tensor, "batch tokens"] | None,
    ) -> Float[Tensor, "batch tokens channels"]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(
        self, x: Float[Tensor, "batch tokens channels"]
    ) -> Float[Tensor, "batch tokens channels"]:
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


class TimestepTransformerEncoder(nn.Module):
    """Stack of cloned timestep transformer encoder layers.

    Origin:
        This is the VQ-Diffusion-style cloned ``TransformerEncoder`` wrapper
        used by LayoutDM and Layout-Corrector around the shared ``Block`` layer.

    Args:
        encoder_layer: Layer to clone for the stack.
        num_layers: Number of cloned layers.
        norm: Optional final normalization module.
    """

    def __init__(
        self,
        encoder_layer: TimestepTransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize a transformer encoder stack."""
        super().__init__()
        self.layers = clone_module_list(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Float[Tensor, "batch tokens channels"],
        mask: Bool[Tensor, "..."] | None = None,
        src_key_padding_mask: Bool[Tensor, "batch tokens"] | None = None,
        timestep: Int[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch tokens channels"]:
        """Run hidden states through all encoder layers."""
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
            )
        return self.norm(output) if self.norm is not None else output


Block = TimestepTransformerEncoderLayer
TransformerEncoder = TimestepTransformerEncoder
