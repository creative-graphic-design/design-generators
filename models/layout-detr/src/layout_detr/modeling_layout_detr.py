"""Transformers-compatible LayoutDETR model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_layout_detr import LayoutDetrConfig


@dataclass
class LayoutDetrModelOutput(ModelOutput):
    """Raw LayoutDETR model output."""

    bbox: Float[torch.Tensor, "batch elements 4"]
    labels: Int[torch.Tensor, "batch elements"] = cast(
        Int[torch.Tensor, "batch elements"], None
    )
    mask: Bool[torch.Tensor, "batch elements"] = cast(
        Bool[torch.Tensor, "batch elements"], None
    )
    latents: Float[torch.Tensor, "batch elements latent"] | None = None
    hidden_states: Float[torch.Tensor, "batch elements hidden"] | None = None


class LayoutDetrForConditionalGeneration(PreTrainedModel):
    """A standard ``PreTrainedModel`` wrapper for LayoutDETR forward inference."""

    config_class = LayoutDetrConfig
    base_model_prefix = "layout_detr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: LayoutDetrConfig) -> None:
        """Initialize lightweight LayoutDETR layers."""
        super().__init__(config)
        self.background_encoder = nn.Sequential(
            nn.Conv2d(config.img_channels, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_z = nn.Linear(config.z_dim, config.bert_f_dim)
        self.emb_label = nn.Embedding(config.num_bbox_labels, config.bert_f_dim)
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.bert_f_dim)
        self.text_len_embeddings = nn.Embedding(
            config.max_text_length, config.bert_f_dim
        )
        self.background_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_in = nn.Sequential(
            nn.Linear(config.bert_f_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, min(8, config.hidden_dim // 8)),
            dim_feedforward=max(config.hidden_dim * 4, 64),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 4),
        )
        self.post_init()

    def forward(
        self,
        *,
        pixel_values: Float[torch.Tensor, "batch channels height width"],
        input_ids: Int[torch.Tensor, "batch elements tokens"],
        text_attention_mask: Bool[torch.Tensor, "batch elements tokens"],
        bbox_labels: Int[torch.Tensor, "batch elements"],
        layout_mask: Bool[torch.Tensor, "batch elements"],
        latents: Float[torch.Tensor, "batch elements latent"],
        return_dict: bool | None = None,
    ) -> LayoutDetrModelOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the LayoutDETR conditional forward pass."""
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if bbox_labels.ndim != 2:
            raise ValueError("bbox_labels must have shape (batch, elements)")
        if (
            latents.shape[:2] != bbox_labels.shape
            or latents.shape[-1] != self.config.z_dim
        ):
            raise ValueError("latents must have shape (batch, elements, z_dim)")
        if input_ids.shape[:2] != bbox_labels.shape:
            raise ValueError("input_ids must have shape (batch, elements, tokens)")
        labels = bbox_labels.to(dtype=torch.long)
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= self.config.num_bbox_labels
        ):
            raise ValueError("bbox_labels contain ids outside config.num_bbox_labels")
        device = labels.device
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        latents = latents.to(device=device, dtype=self.dtype)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        text_attention_mask = text_attention_mask.to(device=device, dtype=torch.bool)
        layout_mask = layout_mask.to(device=device, dtype=torch.bool)

        bg = self.background_proj(self.background_encoder(pixel_values)).unsqueeze(1)
        z = self.fc_z(latents)
        label_features = self.emb_label(labels)
        text_tokens = self.text_embeddings(input_ids)
        token_mask = text_attention_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        denom = token_mask.sum(dim=2).clamp_min(1.0)
        text_features = (text_tokens * token_mask).sum(dim=2) / denom
        lengths = text_attention_mask.sum(dim=-1).clamp_max(
            self.config.max_text_length - 1
        )
        text_len_features = self.text_len_embeddings(lengths)
        hidden = self.fc_in(
            torch.cat([z, label_features, text_features, text_len_features], dim=-1)
        )
        hidden = hidden + bg
        hidden = self.transformer(hidden, src_key_padding_mask=~layout_mask)
        bbox = torch.sigmoid(self.bbox_embed(hidden))
        if not return_dict:
            return bbox, labels, layout_mask
        return LayoutDetrModelOutput(
            bbox=bbox,
            labels=labels,
            mask=layout_mask,
            latents=latents,
            hidden_states=hidden,
        )
