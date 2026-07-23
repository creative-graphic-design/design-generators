"""DLT transformer denoiser with original state-dict key compatibility."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import cast

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange
from jaxtyping import Float, Int
from torch import nn


@dataclass
class DLTModelOutput(BaseOutput):
    """Output returned by the DLT denoiser.

    Attributes:
        box: Predicted clean internal-range boxes.
        logits: Category logits.
    """

    box: Float[torch.Tensor, "batch elements 4"]
    logits: Float[torch.Tensor, "batch elements categories"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the original DLT model."""

    def __init__(
        self, d_model: int, dropout: float = 0.05, max_len: int = 5000
    ) -> None:
        """Create the sinusoidal encoding table."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[torch.Tensor, "sequence batch channels"]
    ) -> Float[torch.Tensor, "sequence batch channels"]:
        """Add positional encodings to a sequence tensor."""
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Timestep MLP used by the original DLT model."""

    def __init__(self, latent_dim: int, seq_pos_enc: PositionalEncoding) -> None:
        """Initialize the timestep embedder."""
        super().__init__()
        self.seq_pos_enc = seq_pos_enc
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self, timesteps: Int[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "1 batch channels"]:
        """Embed diffusion timesteps."""
        pe = cast(torch.Tensor, self.seq_pos_enc.pe)
        return self.time_embed(pe[timesteps]).permute(1, 0, 2)


class DLT(ModelMixin, ConfigMixin):
    """Joint continuous/discrete DLT denoiser.

    The module names intentionally match the original implementation so
    original ``model.save_pretrained`` directories can load without key
    rewriting.

    Args:
        categories_num: Original category count including pad and mask/drop ids.
        latent_dim: Transformer latent dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        dropout_r: Dropout probability.
        activation: Transformer activation.
        cond_emb_size: Box-condition embedding size.
        cat_emb_size: Category embedding size.
    """

    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        categories_num: int,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout_r: float = 0.0,
        activation: str = "gelu",
        cond_emb_size: int = 224,
        cat_emb_size: int = 64,
    ) -> None:
        """Initialize the DLT denoiser."""
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.categories_num = categories_num
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)
        self.cat_emb = nn.Parameter(torch.randn(self.categories_num, cat_emb_size))
        self.cond_mask_box_emb = nn.Parameter(torch.randn(2, cond_emb_size))
        self.cond_mask_cat_emb = nn.Parameter(torch.randn(2, cat_emb_size))

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 2,
            dropout=dropout_r,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)
        self.output_process = nn.Sequential(nn.Linear(self.latent_dim, 4))
        self.output_cls = nn.Sequential(nn.Linear(self.latent_dim, categories_num))
        self.size_emb = nn.Sequential(nn.Linear(2, cond_emb_size))
        self.loc_emb = nn.Sequential(nn.Linear(2, cond_emb_size))

    def forward(
        self,
        sample: dict[str, torch.Tensor],
        noisy_sample: dict[str, torch.Tensor],
        timesteps: Int[torch.Tensor, "batch"],
        return_dict: bool = False,
    ) -> (
        DLTModelOutput
        | tuple[
            Float[torch.Tensor, "batch elements 4"],
            Float[torch.Tensor, "batch elements categories"],
        ]
    ):
        """Predict clean boxes and category logits for a noisy layout.

        Args:
            sample: Original-format conditioning batch with ``box_cond``,
                ``cat``, ``mask_box``, and ``mask_cat``.
            noisy_sample: Current noisy ``box`` and ``cat`` tensors.
            timesteps: Continuous diffusion timestep per batch item.
            return_dict: Whether to return ``DLTModelOutput``.

        Returns:
            Either a two-tuple ``(box, logits)`` for original compatibility or a
            dataclass output.
        """
        cat_input = (
            noisy_sample["cat"] * sample["mask_cat"]
            + (1 - sample["mask_cat"]) * sample["cat"]
        )
        cat_input_flat = rearrange(cat_input, "b c -> (b c)")
        sample_tensor = (
            sample["mask_box"] * noisy_sample["box"]
            + (1 - sample["mask_box"]) * sample["box_cond"]
        )

        xy = sample_tensor[:, :, :2]
        wh = sample_tensor[:, :, 2:]

        elem_cat_emb = self.cat_emb[cat_input_flat, :]
        elem_cat_emb = rearrange(
            elem_cat_emb, "(b c) d -> b c d", b=noisy_sample["box"].shape[0]
        )

        def mask_to_emb(
            mask: torch.Tensor, cond_mask_emb: torch.Tensor
        ) -> torch.Tensor:
            mask_flat = rearrange(mask, "b c -> (b c)").long()
            mask_all_emb = cond_mask_emb[mask_flat, :]
            return rearrange(mask_all_emb, "(b c) d -> b c d", b=mask.shape[0])

        emb_mask_wh = mask_to_emb(sample["mask_box"][:, :, 2], self.cond_mask_box_emb)
        emb_mask_xy = mask_to_emb(sample["mask_box"][:, :, 0], self.cond_mask_box_emb)
        emb_mask_cl = mask_to_emb(sample["mask_cat"], self.cond_mask_cat_emb)
        t_emb = self.embed_timestep(timesteps)

        size_emb = self.size_emb(wh) + emb_mask_wh
        loc_emb = self.loc_emb(xy) + emb_mask_xy
        elem_cat_emb = elem_cat_emb + emb_mask_cl

        tokens_emb = torch.cat([size_emb, loc_emb, elem_cat_emb], dim=-1)
        tokens_emb = rearrange(tokens_emb, "b c d -> c b d")
        xseq = torch.cat((t_emb, tokens_emb), dim=0)
        xseq = self.seq_pos_enc(xseq)

        output = self.seqTransEncoder(xseq)[1:]
        output = rearrange(output, "c b d -> b c d")
        output_box = self.output_process(output)
        output_cls = self.output_cls(output)
        if not return_dict:
            return output_box, output_cls
        return DLTModelOutput(box=output_box, logits=output_cls)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        is_main_process: bool = True,
        save_function: Callable[..., object] | None = None,
        safe_serialization: bool = False,
        variant: str | None = None,
        max_shard_size: int | str = "10GB",
        push_to_hub: bool = False,
        use_flashpack: bool = False,
        **kwargs: object,
    ) -> None:
        """Save the model with PyTorch serialization by default.

        DLT keeps the original shared positional-encoding module structure,
        which creates shared buffers that safetensors refuses to flatten.
        """
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            use_flashpack=use_flashpack,
            **kwargs,
        )
