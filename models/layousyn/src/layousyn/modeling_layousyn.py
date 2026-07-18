"""PyTorch modules for the converted LayouSyn DiT denoiser."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from torch import nn

from .configuration_layousyn import resolve_model_shape


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN shift and scale."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Small MLP with timm-compatible ``fc1``/``fc2`` parameter names."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int
    ) -> None:
        """Initialize the feed-forward projection."""
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.fc2(self.act(self.fc1(x)))


class ScalarEmbedder(nn.Module):
    """Vendor sinusoidal scalar embedding plus MLP projection."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize the embedder."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def scalar_embedding(
        scalar: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal embeddings for scalar values."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=scalar.device)
        args = scalar[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """Embed scalar values."""
        return self.mlp(self.scalar_embedding(scalar, self.frequency_embedding_size))


class InputEmbedder(nn.Module):
    """Linear layout-coordinate embedder."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize projection."""
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project layout coordinates."""
        return self.proj(x)


class ConceptEmbedder(nn.Module):
    """Project concept embeddings into the DiT hidden width."""

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        """Initialize projection."""
        super().__init__()
        self.proj = Mlp(in_channels, hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project concept embeddings."""
        return self.proj(x)


class CaptionEmbedderIdentity(nn.Module):
    """No-op caption embedder for unconditional checkpoints."""

    def forward(
        self,
        caption: torch.Tensor | None,
        caption_padding_mask: torch.Tensor | None,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return caption inputs unchanged."""
        del train, force_drop_ids
        return caption, caption_padding_mask


class CaptionEmbedder(nn.Module):
    """Project caption embeddings and apply classifier-free label dropout."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float,
        y_null_embedding: torch.Tensor,
        y_null_embedding_mask: torch.Tensor,
    ) -> None:
        """Initialize caption projection and null caption buffers."""
        super().__init__()
        self.proj = Mlp(in_channels, hidden_size, hidden_size)
        self.y_embedding: torch.Tensor
        self.y_padding_mask: torch.Tensor
        self.register_buffer("y_embedding", y_null_embedding.float())
        self.register_buffer("y_padding_mask", y_null_embedding_mask.bool())
        self.uncond_prob = uncond_prob

    def token_drop(
        self,
        caption: torch.Tensor,
        caption_padding_mask: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace selected captions with the learned null caption."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        y_embedding = (
            self.y_embedding.to(caption.device).unsqueeze(0).expand_as(caption)
        )
        y_padding_mask = (
            self.y_padding_mask.to(caption_padding_mask.device)
            .unsqueeze(0)
            .expand_as(caption_padding_mask)
        )
        caption = torch.where(drop_ids[:, None, None], y_embedding, caption)
        caption_padding_mask = torch.where(
            drop_ids[:, None], y_padding_mask, caption_padding_mask
        )
        return caption, caption_padding_mask

    def forward(
        self,
        caption: torch.Tensor,
        caption_padding_mask: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project caption embeddings."""
        if (train and self.uncond_prob > 0) or force_drop_ids is not None:
            caption, caption_padding_mask = self.token_drop(
                caption, caption_padding_mask, force_drop_ids
            )
        return self.proj(caption), caption_padding_mask


class DiTBlock(nn.Module):
    """LayouSyn conditional DiT block with concept and caption attention."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        """Initialize one conditional block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = Mlp(hidden_size, mlp_hidden_dim, hidden_size)
        self.mlp_xenc = Mlp(hidden_size, mlp_hidden_dim, hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_enc: torch.Tensor,
        x_padding_mask: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        y_padding_mask: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one conditional block."""
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
            shift_mlp_xenc,
            scale_mlp_xenc,
            gate_mlp_xenc,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        modulate_sa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                modulate_sa + pos_embed + x_enc,
                modulate_sa + pos_embed + x_enc,
                modulate_sa,
                key_padding_mask=x_padding_mask,
            )[0]
        )
        x_concat = torch.cat([x + pos_embed + x_enc, x_enc + pos_embed], dim=1)
        x_res, x_enc_res = self.cross_attn(
            x_concat, y, y, key_padding_mask=y_padding_mask
        )[0].chunk(2, dim=1)
        x = x + x_res
        x_enc = x_enc + x_enc_res
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(
            modulate(self.norm3(x), shift_mlp_x, scale_mlp_x)
        )
        x_enc = x_enc + gate_mlp_xenc.unsqueeze(1) * self.mlp_xenc(
            modulate(self.norm4(x_enc), shift_mlp_xenc, scale_mlp_xenc)
        )
        return x, x_enc

    def initialize_weights(self) -> None:
        """Zero vendor adaLN and cross-attention output projections."""
        modulation = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)
        nn.init.constant_(modulation.weight, 0)
        nn.init.constant_(modulation.bias, 0)


class DiTUCBlock(nn.Module):
    """LayouSyn unconditional DiT block."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        """Initialize one unconditional block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: torch.Tensor,
        c: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply one unconditional block."""
        del kwargs
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        modulate_sa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                modulate_sa,
                modulate_sa,
                modulate_sa,
                key_padding_mask=x_padding_mask,
            )[0]
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

    def initialize_weights(self) -> None:
        """Zero vendor adaLN projection."""
        modulation = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.constant_(modulation.weight, 0)
        nn.init.constant_(modulation.bias, 0)


class FinalLayer(nn.Module):
    """Vendor final adaLN projection."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        """Initialize final layer."""
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project hidden states to epsilon and variance channels."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def get_1d_sincos_pos_embed(embed_dim: int, max_len: int) -> np.ndarray:
    """Create vendor sine/cosine positional embeddings."""
    grid = np.arange(max_len, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create vendor sine/cosine positional embeddings from positions."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class LayouSynDiTModel(ModelMixin, ConfigMixin):
    """Converted LayouSyn DiT denoiser."""

    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        in_channels: int = 4,
        max_in_len: int = 60,
        concept_in_channels: int = 768,
        y_in_channels: int | None = 768,
        max_y_len: int | None = 120,
        model_name: str = "DiT-S",
        hidden_size: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        is_unconditional: bool = False,
    ) -> None:
        """Initialize the converted DiT model."""
        super().__init__()
        shape = resolve_model_shape(
            model_name,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
        )
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.num_heads = shape["num_heads"]
        self.max_len = max_in_len
        self.is_unconditional = is_unconditional
        self.x_embedder = InputEmbedder(in_channels, shape["hidden_size"])
        self.concept_embedder = ConceptEmbedder(
            concept_in_channels, shape["hidden_size"]
        )
        self.t_embedder = ScalarEmbedder(shape["hidden_size"])
        self.ar_embedder = ScalarEmbedder(shape["hidden_size"])
        if is_unconditional:
            self.y_embedder = CaptionEmbedderIdentity()
        else:
            if y_in_channels is None or max_y_len is None:
                raise ValueError("y_in_channels and max_y_len are required")
            y_null = torch.zeros(max_y_len, y_in_channels)
            y_mask = torch.ones(max_y_len, dtype=torch.bool)
            self.y_embedder = CaptionEmbedder(
                y_in_channels,
                shape["hidden_size"],
                class_dropout_prob,
                y_null,
                y_mask,
            )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_in_len, shape["hidden_size"]), requires_grad=False
        )
        block_cls = DiTUCBlock if is_unconditional else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(shape["hidden_size"], shape["num_heads"], mlp_ratio=mlp_ratio)
                for _ in range(shape["depth"])
            ]
        )
        self.final_layer = FinalLayer(shape["hidden_size"], 2 * in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights with the vendor policy."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.max_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        if not self.is_unconditional and isinstance(self.y_embedder, CaptionEmbedder):
            nn.init.normal_(self.y_embedder.proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.proj.fc2.weight, std=0.02)
        nn.init.normal_(self.concept_embedder.proj.fc1.weight, std=0.02)
        nn.init.normal_(self.concept_embedder.proj.fc2.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            block.initialize_weights()
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        x_padding_mask: torch.Tensor,
        aspect_ratio: torch.Tensor,
        concept_embeds: torch.Tensor,
        caption_embeds: torch.Tensor | None = None,
        caption_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict epsilon and variance channels for one timestep."""
        x = self.x_embedder(sample)
        x_enc = self.concept_embedder(concept_embeds)
        c = self.t_embedder(timestep) + self.ar_embedder(aspect_ratio)
        if self.is_unconditional:
            for block in self.blocks:
                x = block(x, x_padding_mask=x_padding_mask, c=c)
        else:
            if caption_embeds is None or caption_padding_mask is None:
                raise ValueError("caption_embeds and caption_padding_mask are required")
            y, y_padding_mask = self.y_embedder(
                caption_embeds, caption_padding_mask, self.training
            )
            for block in self.blocks:
                x, x_enc = block(
                    x,
                    x_enc,
                    x_padding_mask,
                    c,
                    y=y,
                    y_padding_mask=y_padding_mask,
                    pos_embed=self.pos_embed[:, : x.shape[1]],
                )
        out = self.final_layer(x, c).chunk(2, dim=-1)
        return torch.cat([out[0], out[1]], dim=1)

    def forward_with_cfg(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        x_padding_mask: torch.Tensor,
        aspect_ratio: torch.Tensor,
        concept_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
        caption_padding_mask: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run vendor classifier-free guidance batching."""
        half = sample[: len(sample) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            combined,
            timestep,
            x_padding_mask=x_padding_mask,
            aspect_ratio=aspect_ratio,
            concept_embeds=concept_embeds,
            caption_embeds=caption_embeds,
            caption_padding_mask=caption_padding_mask,
        )
        eps, rest = model_out[:, : sample.shape[1]], model_out[:, sample.shape[1] :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def convert_vendor_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert a vendor DiT state dict to the wrapped model key space."""
    return dict(state_dict)
