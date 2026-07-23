"""Transformer denoiser used by CGB-DM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from jaxtyping import Float, Int
from torch import nn

from laygen.nn import SinusoidalPosEmb


@dataclass
class CGBDMModelOutput(BaseOutput):
    """Output returned by the CGB-DM denoiser.

    Attributes:
        sample: Predicted epsilon tensor with the same shape as the input layout.
        cgb_weight: Content-graphic balance weight estimated from image tokens.
    """

    sample: Float[torch.Tensor, "batch elements channels"]
    cgb_weight: Float[torch.Tensor, "batch 1 1"] | None = None


class CGBDMImageEncoder(nn.Module):
    """Small ViT-style image encoder preserving CGB-DM component boundaries."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        dim_model: int,
    ) -> None:
        """Initialize patch projection and positional tokens."""
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels, dim_model, patch_size, patch_size)
        patches = (height // patch_size) * (width // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, patches + 1, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(
            dim_model,
            nhead=8,
            dim_feedforward=dim_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a four-channel content image into tokens."""
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        return self.norm(self.transformer(tokens))


class CGBDMLayoutModule(nn.Module):
    """Timestep-conditioned layout encoder or decoder block stack."""

    def __init__(
        self,
        *,
        seq_dim: int,
        dim_model: int,
        n_head: int,
        feature_dim: int,
        num_layers: int,
        num_train_timesteps: int,
        if_encoder: bool,
    ) -> None:
        """Initialize layout projection and attention layers."""
        super().__init__()
        self.if_encoder = if_encoder
        self.layer_in = nn.Linear(seq_dim, dim_model)
        self.time_embedding = SinusoidalPosEmb(num_train_timesteps, dim_model)
        layer = nn.TransformerEncoderLayer(
            dim_model,
            n_head,
            dim_feedforward=feature_dim,
            batch_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.img_cross = nn.MultiheadAttention(dim_model, n_head, batch_first=True)
        self.sal_cross = nn.MultiheadAttention(dim_model, n_head, batch_first=True)
        self.layer_out = nn.Linear(dim_model, seq_dim)
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        sample: torch.Tensor,
        image_tokens: torch.Tensor | None,
        cgb_weight: torch.Tensor | None,
        saliency_tokens: torch.Tensor | None,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run a layout encoder or decoder pass."""
        hidden = (
            sample
            if sample.shape[-1] != self.layer_in.in_features
            else self.layer_in(sample)
        )
        hidden = hidden + self.time_embedding(timestep).unsqueeze(1).to(hidden)
        hidden = self.layers(hidden)
        if image_tokens is not None:
            attended = self.img_cross(
                hidden, image_tokens, image_tokens, need_weights=False
            )[0]
            hidden = hidden + (
                attended if cgb_weight is None else cgb_weight * attended
            )
        if saliency_tokens is not None:
            hidden = (
                hidden
                + self.sal_cross(
                    hidden, saliency_tokens, saliency_tokens, need_weights=False
                )[0]
            )
        hidden = self.norm(hidden)
        if self.if_encoder:
            return hidden
        return self.layer_out(hidden)


class CGBDMQFormer(nn.Module):
    """Estimate a scalar content-graphic balance weight from image tokens."""

    def __init__(self, dim_model: int) -> None:
        """Initialize pooling projection."""
        super().__init__()
        self.scale_emb = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.out = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Linear(dim_model // 2, 1),
            nn.Softplus(),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Return a positive scalar weight per example."""
        pooled = image_tokens.mean(dim=1, keepdim=True) + self.scale_emb
        return self.out(pooled)


class CGBDMMLP(nn.Module):
    """Softplus MLP used for saliency-box embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize three linear layers."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Embed saliency boxes as one-token sequences."""
        hidden = value
        for layer in self.layers[:-1]:
            hidden = torch.nn.functional.softplus(layer(hidden))
        return self.layers[-1](hidden)


class CGBDMTransformerModel(ModelMixin, ConfigMixin):
    """CGB-DM transformer denoiser with image and saliency conditioning.

    Args:
        num_labels: Internal class-channel count including invalid/pad.
        max_seq_length: Maximum number of layout elements.
        image_size: Image tensor size as ``(height, width)``.
        patch_size: Image patch size.
        dim_model: Hidden dimension.
        n_head: Attention head count.
        feature_dim: Feed-forward hidden dimension.
        num_layers: Number of decoder layers.
        num_train_timesteps: Number of training diffusion steps.

    Examples:
        >>> model = CGBDMTransformerModel(num_labels=4, max_seq_length=2, image_size=(32, 32), dim_model=16, n_head=2, feature_dim=32, num_layers=1)
        >>> model.seq_dim
        8
    """

    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        num_labels: int = 4,
        max_seq_length: int = 16,
        image_size: tuple[int, int] | list[int] = (384, 256),
        patch_size: int = 32,
        dim_model: int = 512,
        n_head: int = 8,
        feature_dim: int = 1024,
        num_layers: int = 4,
        num_train_timesteps: int = 1000,
    ) -> None:
        """Initialize the CGB-DM denoising network."""
        super().__init__()
        self.num_labels = int(num_labels)
        self.max_seq_length = int(max_seq_length)
        self.image_size: tuple[int, int] = (int(image_size[0]), int(image_size[1]))
        self.seq_dim = self.num_labels + 4
        self.img_encoder = CGBDMImageEncoder(
            image_size=self.image_size,
            patch_size=patch_size,
            in_channels=4,
            dim_model=dim_model,
        )
        self.layout_encoder = CGBDMLayoutModule(
            seq_dim=self.seq_dim,
            dim_model=dim_model,
            n_head=n_head,
            feature_dim=feature_dim,
            num_layers=max(1, num_layers // 2),
            num_train_timesteps=num_train_timesteps,
            if_encoder=True,
        )
        self.layout_decoder = CGBDMLayoutModule(
            seq_dim=self.seq_dim,
            dim_model=dim_model,
            n_head=n_head,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_train_timesteps=num_train_timesteps,
            if_encoder=False,
        )
        self.cgbwp = CGBDMQFormer(dim_model)
        self.salbox_encoder = CGBDMMLP(4, dim_model, dim_model)

    def forward(
        self,
        sample: Float[torch.Tensor, "batch elements channels"],
        image: Float[torch.Tensor, "batch 4 height width"],
        saliency_box: Float[torch.Tensor, "batch 1 4"],
        timestep: Int[torch.Tensor, "batch"],
        return_dict: bool = True,
    ) -> CGBDMModelOutput | tuple[Float[torch.Tensor, "batch elements channels"]]:
        """Predict epsilon for a noisy layout tensor.

        Args:
            sample: Noisy class-plus-box layout tensor.
            image: Four-channel RGB/saliency tensor in ``[-1, 1]``.
            saliency_box: Saliency box tensor in internal ``[-1, 1]`` center xywh.
            timestep: Per-example diffusion timestep ids.
            return_dict: Whether to return ``CGBDMModelOutput``.

        Returns:
            Output dataclass or one-item tuple containing predicted epsilon.
        """
        image_tokens = self.img_encoder(image)
        saliency_tokens = self.salbox_encoder(saliency_box)
        encoded = self.layout_encoder(sample, None, None, None, timestep)
        cgb_weight = self.cgbwp(image_tokens)
        pred = self.layout_decoder(
            encoded,
            image_tokens,
            cgb_weight,
            saliency_tokens,
            timestep,
        )
        if not return_dict:
            return (pred,)
        return CGBDMModelOutput(sample=pred, cgb_weight=cgb_weight)
