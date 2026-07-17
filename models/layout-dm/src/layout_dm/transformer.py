from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return x * F.sigmoid(1.702 * x)


def _activation(
    name: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(name):
        return name
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "gelu2":
        return _gelu2
    raise ValueError(f"Unsupported activation: {name}")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLayerNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        self.emb = (
            SinusoidalPosEmb(max_timestep, n_embd)
            if "abs" in emb_type
            else nn.Embedding(max_timestep, n_embd)
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "relu",
        batch_first: bool = True,
        norm_first: bool = True,
        diffusion_step: int = 100,
        timestep_type: str | None = None,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.timestep_type = timestep_type
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = (
            AdaLayerNorm(d_model, diffusion_step, timestep_type or "adalayernorm_abs")
            if timestep_type
            else nn.LayerNorm(d_model, eps=1e-5)
        )
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
        x = src
        if self.norm_first:
            normed = self.norm1(x, timestep) if self.timestep_type else self.norm1(x)
            x = x + self._sa_block(normed, src_mask, src_key_padding_mask)
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


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: Block, num_layers: int, norm: nn.Module | None = None
    ):
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
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
            )
        return self.norm(output) if self.norm is not None else output


class ElementPositionalEmbedding(nn.Module):
    def __init__(self, dim_model: int, max_token_length: int, n_attr_per_elem: int = 5):
        super().__init__()
        self.n_elem = max_token_length // n_attr_per_elem
        self.n_attr_per_elem = n_attr_per_elem
        self.elem_emb = nn.Parameter(torch.rand(self.n_elem, dim_model))
        self.attr_emb = nn.Parameter(torch.rand(self.n_attr_per_elem, dim_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, seq_len = h.shape[:2]
        elem_emb = repeat(self.elem_emb, "s d -> (s x) d", x=self.n_attr_per_elem)
        attr_emb = repeat(self.attr_emb, "x d -> (s x) d", s=self.n_elem)
        emb = (elem_emb + attr_emb)[:seq_len]
        return repeat(emb, "s d -> b s d", b=batch)

    @property
    def no_decay_param_names(self) -> list[str]:
        return ["elem_emb", "attr_emb"]


class CategoricalTransformer(nn.Module):
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
        timestep_type: str | None = "adalayernorm",
    ) -> None:
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
        self, input_ids: torch.LongTensor, timestep: torch.LongTensor | None = None
    ) -> dict[str, torch.Tensor]:
        hidden = self.cat_emb(input_ids)
        hidden = self.drop(hidden + self.pos_emb(hidden))
        hidden = self.backbone(hidden, timestep=timestep)
        return {"logits": self.head(hidden)}
