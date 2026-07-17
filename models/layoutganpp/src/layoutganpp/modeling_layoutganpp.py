from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from laygen.common.outputs import LayoutGenerationOutput

from .configuration_layoutganpp import LayoutGANPPConfig


@dataclass
class LayoutGANPPModelOutput(ModelOutput):
    bbox: torch.FloatTensor
    labels: torch.LongTensor | None = None
    mask: torch.BoolTensor | None = None
    latents: torch.FloatTensor | None = None


_CONDITION_ALIASES = {
    "label": "label",
    "c": "label",
    "cat_cond": "label",
}


def normalize_condition_type(condition_type: str) -> str:
    key = condition_type.lower().replace("-", "_")
    if key == "unconditional":
        return "unconditional"
    try:
        return _CONDITION_ALIASES[key]
    except KeyError as exc:
        supported = ("label", "c", "cat_cond")
        raise ValueError(
            f"Unsupported condition_type={condition_type}; {supported=}"
        ) from exc


class LayoutGANPPModel(PreTrainedModel):
    config_class = LayoutGANPPConfig
    base_model_prefix = "layoutganpp"
    supports_gradient_checkpointing = False

    def __init__(self, config: LayoutGANPPConfig) -> None:
        super().__init__(config)
        self.fc_z = nn.Linear(config.latent_size, config.d_model // 2)
        self.emb_label = nn.Embedding(config.num_labels, config.d_model // 2)
        self.fc_in = nn.Linear(config.d_model, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model // 2,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.fc_out = nn.Linear(config.d_model, 4)
        self.post_init()

    def forward(
        self,
        latents: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        return_dict: bool = True,
    ) -> (
        LayoutGANPPModelOutput
        | tuple[torch.FloatTensor, torch.LongTensor, torch.BoolTensor]
    ):
        labels = labels.to(dtype=torch.long)
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, sequence)")
        if latents.shape[:2] != labels.shape:
            raise ValueError("latents must have shape (batch, sequence, latent_size)")
        if latents.shape[-1] != self.config.latent_size:
            raise ValueError(
                f"latents last dimension must be {self.config.latent_size}"
            )
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= self.config.num_labels
        ):
            raise ValueError("labels contain ids outside config.num_labels")

        if padding_mask is None:
            if attention_mask is None:
                padding_mask = torch.zeros(
                    labels.shape, dtype=torch.bool, device=labels.device
                )
            else:
                padding_mask = ~attention_mask.to(
                    device=labels.device, dtype=torch.bool
                )
        else:
            padding_mask = padding_mask.to(device=labels.device, dtype=torch.bool)
        latents = latents.to(device=labels.device, dtype=self.dtype)
        z = self.fc_z(latents)
        label_emb = self.emb_label(labels)
        hidden = torch.cat([z, label_emb], dim=-1)
        hidden = torch.relu(self.fc_in(hidden)).permute(1, 0, 2)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        bbox = torch.sigmoid(self.fc_out(hidden.permute(1, 0, 2)))
        mask = ~padding_mask
        if not return_dict:
            return bbox, labels, mask
        return LayoutGANPPModelOutput(
            bbox=bbox, labels=labels, mask=mask, latents=latents
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int = 1,
        condition_type: str = "label",
        bbox: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        mask: torch.BoolTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        num_elements: int | list[int] | torch.LongTensor | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        latents: torch.FloatTensor | None = None,
        **model_kwargs,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        del bbox, num_elements, box_format, normalized, canvas_size, num_inference_steps
        if model_kwargs:
            unknown = ", ".join(sorted(model_kwargs))
            raise ValueError(f"Unsupported generation kwargs: {unknown}")
        canonical = normalize_condition_type(condition_type)
        if canonical == "unconditional":
            raise ValueError(
                "layoutganpp v1 requires labels; unconditional is unsupported"
            )
        if labels is None:
            raise ValueError("labels are required for layoutganpp generation")

        device = next(self.parameters()).device
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        batch_size = labels.shape[0]
        if mask is not None:
            attention_mask = mask
        if attention_mask is None:
            attention_mask = torch.ones(labels.shape, dtype=torch.bool, device=device)
        else:
            attention_mask = torch.as_tensor(
                attention_mask, dtype=torch.bool, device=device
            )
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        if latents is None:
            latents = self._sample_latents(
                (batch_size, labels.shape[1], self.config.latent_size),
                seed=seed,
                generator=generator,
                device=device,
                dtype=self.dtype,
            )
        else:
            latents = torch.as_tensor(latents, dtype=self.dtype, device=device)
        out = self.forward(
            latents=latents,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
        )
        layout = LayoutGenerationOutput(
            bbox=out.bbox.detach().cpu(),
            labels=out.labels.detach().cpu(),
            mask=out.mask.detach().cpu(),
            id2label={int(k): v for k, v in self.config.id2label.items()},
            intermediates={
                "condition_type": canonical,
                "latents": out.latents.detach().cpu() if return_intermediates else None,
            }
            if return_intermediates
            else None,
        )
        if output_type == "dict":
            return dict(layout)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return layout

    def _sample_latents(
        self,
        shape: tuple[int, int, int],
        *,
        seed: int | None,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if generator is None and seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
