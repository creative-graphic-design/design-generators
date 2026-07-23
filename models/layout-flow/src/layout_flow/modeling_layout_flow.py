"""PyTorch modules for the converted LayoutFlow vector-field model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from einops import pack, rearrange, unpack
from jaxtyping import Bool, Float, Int
from torch import nn

from laygen.nn import clone_module_list, get_activation

from .configuration_layout_flow import AttrEncoding, SeqType


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return clone_module_list(module, n)


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return get_activation("gelu2")(x)


def _get_activation_fn(
    activation: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if not isinstance(activation, str):
        return activation
    return get_activation(activation)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the vendor backbone."""

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 10000
    ) -> None:
        """Initialize positional encodings."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "1 tokens channels"]:
        """Return positional encodings matching the sequence length of ``x``."""
        return self.dropout(self.pe[:, : x.shape[1]])


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm conditioned on the integration timestep."""

    def __init__(self, n_embd: int) -> None:
        """Initialize timestep-conditioned normalization."""
        super().__init__()
        self.emb = nn.Sequential(
            nn.Unflatten(0, (-1, 1)),
            nn.Linear(1, n_embd // 2),
            nn.ReLU(),
            nn.Linear(n_embd // 2, n_embd),
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(
        self,
        x: Float[torch.Tensor, "batch tokens channels"],
        timestep: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch tokens channels"]:
        """Normalize ``x`` with scale and shift predicted from ``timestep``."""
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class LayoutFlowBlock(nn.Module):
    """Transformer encoder block matching the original LayoutFlow backbone."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
    ) -> None:
        """Initialize one LayoutFlow transformer block."""
        super().__init__()
        if not norm_first:
            raise ValueError("LayoutFlow transformer expects prenorm blocks")
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: Float[torch.Tensor, "batch tokens channels"],
        src_mask: Bool[torch.Tensor, "..."] | None = None,
        src_key_padding_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        timestep: Float[torch.Tensor, "batch"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens channels"]:
        """Apply self-attention and feed-forward layers."""
        if timestep is None:
            raise ValueError("timestep is required")
        x = self.norm1(src, timestep)
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        return x + self._ff_block(self.norm2(x))

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


class LayoutFlowTransformerEncoder(nn.Module):
    """Stack of LayoutFlow transformer blocks."""

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None
    ) -> None:
        """Clone and stack ``encoder_layer`` ``num_layers`` times."""
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the stacked encoder blocks."""
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
            )
        return self.norm(output) if self.norm is not None else output


class LayoutDMBackbone(nn.Module):
    """Vendor-compatible LayoutFlow backbone module."""

    def __init__(
        self,
        latent_dim: int = 128,
        tr_enc_only: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        dropout: float = 0.1,
        use_pos_enc: bool = False,
        num_cat: int = 6,
        attr_encoding: AttrEncoding = AttrEncoding.continuous,
        seq_type: SeqType = SeqType.stacked,
    ) -> None:
        """Initialize the vendor-compatible backbone."""
        super().__init__()
        self.geom_dim = 4
        self.num_cat = num_cat
        self.attr_encoding = AttrEncoding(attr_encoding)
        self.seq_type = SeqType(seq_type)
        self.use_pose_enc = use_pos_enc
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=200)
        self.cond_enc = nn.Embedding(
            6,
            latent_dim if self.seq_type is SeqType.stacked else 2 * latent_dim,
        )
        attr_dim = (
            int(np.ceil(np.log2(num_cat)))
            if self.attr_encoding is AttrEncoding.analog_bit
            else 1
        )
        if self.attr_encoding is AttrEncoding.discrete:
            self.type_embed = nn.Embedding(num_cat, latent_dim)
            self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        else:
            self.type_embed = nn.Linear(
                attr_dim,
                latent_dim if self.seq_type is SeqType.stacked else 2 * latent_dim,
            )
            if self.seq_type is SeqType.seq:
                self.geom_enc = nn.ModuleList(
                    [nn.Linear(1, 2 * latent_dim) for _ in range(4)]
                )
            else:
                self.geom_embed = nn.Linear(
                    self.geom_dim,
                    latent_dim if self.seq_type is SeqType.stacked else 2 * latent_dim,
                )
        self.elem_embed = nn.Linear(2 * latent_dim, d_model)
        decoder_layer = LayoutFlowBlock(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.tr_enc_only = tr_enc_only
        if tr_enc_only:
            self.transformer = LayoutFlowTransformerEncoder(
                decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
            )
        else:
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
        self.linear = nn.Linear(d_model, self.geom_dim + attr_dim)
        if self.seq_type is not SeqType.stacked:
            k = 2 if self.seq_type is SeqType.seq_cond else 5
            self.to_attrdim = nn.Linear(
                k * (self.geom_dim + attr_dim), self.geom_dim + attr_dim
            )

    def forward(
        self,
        geom: Float[torch.Tensor, "batch elements 4"],
        attr: Float[torch.Tensor, "batch elements bits"],
        cond_flags: Int[torch.Tensor, "batch elements channels"],
        t: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch elements channels"]:
        """Predict vector-field values for geometry and attribute inputs."""
        ps = None
        if self.attr_encoding is AttrEncoding.discrete:
            geom = self.geom_embed(geom)
            attr = self.type_embed(attr.squeeze())
            x = torch.cat([geom, attr], dim=-1)
        elif self.seq_type is SeqType.stacked:
            geom = self.geom_embed(geom) + self.cond_enc(
                cond_flags[:, :, : self.geom_dim].sum(-1)
            )
            attr = self.type_embed(attr) + self.cond_enc(cond_flags[:, :, -1])
            x, _ = pack([geom, attr], "b s *")
        elif self.seq_type is SeqType.seq_cond:
            geom = self.geom_embed(geom) + self.cond_enc(
                cond_flags[:, :, : self.geom_dim].sum(-1)
            )
            attr = self.type_embed(attr) + self.cond_enc(cond_flags[:, :, -1])
            x, ps = pack([geom, attr], "b * d")
        elif self.seq_type is SeqType.seq:
            geom_parts = [
                self.geom_enc[i](geom[:, :, i, None])
                + self.cond_enc(cond_flags[:, :, i])
                for i in range(4)
            ]
            attr = self.type_embed(attr) + self.cond_enc(cond_flags[:, :, -1])
            x, ps = pack(geom_parts + [attr], "b * d")
        else:
            raise ValueError(f"Unsupported seq_type: {self.seq_type}")
        x = self.elem_embed(x)
        if self.use_pose_enc:
            x = x + self.pos_enc(x)
        x = (
            self.transformer(x, timestep=t)
            if self.tr_enc_only
            else self.transformer(x, x)
        )
        x = self.linear(x)
        if self.seq_type is not SeqType.stacked:
            if ps is None:
                raise ValueError(f"Unsupported seq_type: {self.seq_type}")
            x = unpack(x, ps, "b * d")
            x = rearrange(x, "k b s d -> b s (k d)")
            x = self.to_attrdim(x)
        return x


@dataclass
class LayoutFlowModelOutput(BaseOutput):
    """Output of ``LayoutFlowTransformerModel``."""

    sample: Float[torch.Tensor, "batch elements channels"]


class LayoutFlowTransformerModel(ModelMixin, ConfigMixin):
    """Diffusers model wrapper around the LayoutFlow backbone."""

    config_name: str = "layout_flow_model_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        num_labels: int = 6,
        latent_dim: int = 128,
        tr_enc_only: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_pos_enc: bool = False,
        attr_encoding: AttrEncoding = AttrEncoding.analog_bit,
        seq_type: SeqType = SeqType.stacked,
    ) -> None:
        """Initialize the converted LayoutFlow transformer model.

        Args:
            num_labels: Number of dataset labels.
            latent_dim: Vendor latent dimension.
            tr_enc_only: Whether to use the encoder-only path.
            d_model: Transformer hidden size.
            nhead: Number of attention heads.
            dim_feedforward: Feed-forward hidden size.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.
            use_pos_enc: Whether to add positional encodings.
            attr_encoding: Vendor attribute encoding.
            seq_type: Vendor sequence type.
        """
        super().__init__()
        self.geom_dim = 4
        self.attr_dim = (
            int(np.ceil(np.log2(num_labels)))
            if AttrEncoding(attr_encoding) is AttrEncoding.analog_bit
            else 1
        )
        self.backbone = LayoutDMBackbone(
            latent_dim=latent_dim,
            tr_enc_only=tr_enc_only,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            use_pos_enc=use_pos_enc,
            num_cat=num_labels,
            attr_encoding=attr_encoding,
            seq_type=seq_type,
        )

    def forward(
        self,
        sample: Float[torch.Tensor, "batch elements channels"],
        timestep: Float[torch.Tensor, ""],
        cond_mask: Bool[torch.Tensor, "batch elements channels"],
        return_dict: bool = True,
    ) -> LayoutFlowModelOutput | tuple[Float[torch.Tensor, "batch elements channels"]]:
        """Predict the vector field for a model state.

        Args:
            sample: Current model state.
            timestep: Current integration timestep.
            cond_mask: Condition mask.
            return_dict: Whether to return a dataclass output.

        Returns:
            Model output dataclass or single-item tuple.
        """
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        if timestep.ndim == 0:
            timestep = timestep.repeat(sample.shape[0])
        geom = sample[:, :, : self.geom_dim]
        attr = sample[:, :, self.geom_dim :]
        out = self.backbone(geom, attr, cond_mask.to(torch.long), timestep)
        if not return_dict:
            return (out,)
        return LayoutFlowModelOutput(sample=out)
