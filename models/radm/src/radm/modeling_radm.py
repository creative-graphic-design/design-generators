"""Tiny CPU-capable RADM denoiser components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from jaxtyping import Bool, Float
from torch import nn


@dataclass
class RADMDenoiserOutput(BaseOutput):
    """Predictions from one RADM denoising step."""

    logits: Float[torch.Tensor, "batch proposals classes"]
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"]
    pred_original_sample: Float[torch.Tensor, "batch proposals 4"]
    pred_noise: Float[torch.Tensor, "batch proposals 4"]


class RADMDenoiser(ModelMixin, ConfigMixin):
    """Proposal-box denoiser used by ``RADMPipeline``.

    Args:
        num_classes: Number of semantic classes.
        hidden_dim: Hidden feature dimension.
        text_feature_dim: Text feature dimension.

    Examples:
        >>> model = RADMDenoiser(num_classes=3, hidden_dim=8, text_feature_dim=4)
        >>> out = model(
        ...     boxes_xyxy=torch.zeros(1, 2, 4),
        ...     text_features=torch.zeros(1, 1, 4),
        ...     text_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        ...     timesteps=torch.tensor([0]),
        ... )
        >>> out.logits.shape
        torch.Size([1, 2, 3])
    """

    config_name: str = "denoiser_config.json"

    @register_to_config
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256,
        text_feature_dim: int = 768,
    ) -> None:
        """Initialize the denoiser."""
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.text_feature_dim = int(text_feature_dim)
        self.box_embed = nn.Linear(4, hidden_dim)
        self.text_embed = nn.Linear(text_feature_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        timesteps: Float[torch.Tensor, "batch"] | Float[torch.Tensor, ""],
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"] | None = None,
        image_features: Float[torch.Tensor, "batch channels"] | None = None,
    ) -> RADMDenoiserOutput:
        """Predict proposal labels and denoised boxes.

        Args:
            boxes_xyxy: Current normalized ``xyxy`` proposal boxes.
            timesteps: Current diffusion timesteps.
            text_features: Text feature tensor.
            text_mask: Optional valid-text mask shaped ``(B, T, 1)``.
            image_features: Optional image-level features.

        Returns:
            Denoiser output containing logits and denoised boxes.
        """
        del image_features
        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(boxes_xyxy.shape[0])
        timestep = timesteps.to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
        timestep = timestep.reshape(-1, 1) / max(float(self.config.num_classes), 1.0)
        context = self._pool_text(text_features, text_mask).to(
            device=boxes_xyxy.device, dtype=boxes_xyxy.dtype
        )
        hidden = (
            self.box_embed(boxes_xyxy)
            + self.text_embed(context).unsqueeze(1)
            + self.time_embed(timestep).unsqueeze(1)
        )
        hidden = self.norm(torch.tanh(hidden))
        delta = torch.tanh(self.delta_head(hidden)) * 0.05
        pred_original = (boxes_xyxy + delta).clamp(0.0, 1.0)
        logits = self.class_head(hidden)
        pred_noise = boxes_xyxy - pred_original
        return RADMDenoiserOutput(
            logits=logits,
            boxes_xyxy=pred_original,
            pred_original_sample=pred_original,
            pred_noise=pred_noise,
        )

    def _pool_text(
        self,
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"] | None,
    ) -> Float[torch.Tensor, "batch text_dim"]:
        if text_features.ndim != 3:
            raise ValueError("text_features must have shape (batch, text, dim)")
        if text_mask is None:
            return text_features.mean(dim=1)
        mask = text_mask.to(device=text_features.device, dtype=text_features.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (text_features * mask).sum(dim=1) / denom
