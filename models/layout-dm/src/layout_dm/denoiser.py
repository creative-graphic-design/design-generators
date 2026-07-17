from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from .transformer import CategoricalTransformer


@dataclass
class LayoutDMDenoiserOutput(BaseOutput):
    logits: torch.Tensor


class LayoutDMDenoiser(ModelMixin, ConfigMixin):
    config_name = "denoiser_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        vocab_size: int,
        max_token_length: int,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: str | None = "adalayernorm",
    ) -> None:
        super().__init__()
        self.transformer = CategoricalTransformer(
            vocab_size=vocab_size,
            max_token_length=max_token_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
            timestep_type=timestep_type,
        )

    def forward(
        self, input_ids: torch.Tensor, timesteps: torch.Tensor
    ) -> LayoutDMDenoiserOutput:
        return LayoutDMDenoiserOutput(
            logits=self.transformer(input_ids, timestep=timesteps)["logits"]
        )

    def predict_start_log_probs(
        self, input_ids: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        logits = self(input_ids=input_ids, timesteps=timesteps).logits[:, :, :-1]
        log_pred = F.log_softmax(logits.double(), dim=-1).float()
        zero_mask = torch.full(
            (*log_pred.shape[:2], 1),
            -70.0,
            device=log_pred.device,
            dtype=log_pred.dtype,
        )
        return torch.cat((log_pred, zero_mask), dim=-1).clamp(-70.0, 0.0)
