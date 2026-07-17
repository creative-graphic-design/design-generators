from __future__ import annotations

import copy
import math
from dataclasses import dataclass
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
    sample: torch.FloatTensor


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


def _gelu2(x: torch.Tensor) -> torch.Tensor:
    return x * F.sigmoid(1.702 * x)


def _activation(name: str | Callable[[torch.Tensor], torch.Tensor]):
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


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
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
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "relu",
        batch_first: bool = True,
        norm_first: bool = True,
        diffusion_step: int = 100,
        timestep_type: str | None = "adalayernorm",
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if timestep_type is None:
            self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        elif "adalayernorm" in timestep_type:
            self.norm1 = AdaLayerNorm(d_model, diffusion_step, timestep_type)
        elif "adainnorm" in timestep_type:
            self.norm1 = AdaInsNorm(d_model, diffusion_step, timestep_type)
        else:
            raise ValueError(f"Unsupported timestep_type: {timestep_type}")
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
        timestep_type: str | None = "adalayernorm",
    ) -> None:
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
