"""PyTorch model wrapper for the LayoutGAN++ generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import assert_never

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.common.outputs import LayoutGenerationOutput

from .configuration_layoutganpp import LayoutGANPPConfig


@dataclass
class LayoutGANPPModelOutput(ModelOutput):
    """Raw LayoutGAN++ model output.

    Args:
        bbox: Generated normalized `xywh` boxes with shape `(batch, sequence, 4)`.
        labels: Optional label IDs used for generation.
        mask: Optional valid-element mask.
        latents: Optional latent vectors used by the generator.

    Examples:
        >>> out = LayoutGANPPModelOutput(bbox=torch.zeros(1, 1, 4))
        >>> tuple(out.bbox.shape)
        (1, 1, 4)
    """

    bbox: torch.Tensor
    labels: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    latents: torch.Tensor | None = None


class OutputType(StrEnum):
    """Supported LayoutGAN++ generation output formats."""

    dataclass = auto()
    dict = auto()


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize a public output type value.

    Args:
        output_type: Output type enum or string.

    Returns:
        Normalized output type enum.

    Raises:
        ValueError: If `output_type` is unsupported.

    Examples:
        >>> str(normalize_output_type("dict"))
        'dict'
    """
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


class LayoutGANPPModel(PreTrainedModel):
    """Transformers-compatible LayoutGAN++ generator.

    Args:
        config: LayoutGAN++ model configuration.

    Examples:
        >>> config = LayoutGANPPConfig(num_labels=2, id2label={0: "a", 1: "b"})
        >>> model = LayoutGANPPModel(config)
        >>> model.config.model_type
        'layoutganpp'
    """

    config_class = LayoutGANPPConfig
    base_model_prefix = "layoutganpp"
    supports_gradient_checkpointing = False

    def __init__(self, config: LayoutGANPPConfig) -> None:
        """Initialize the LayoutGAN++ generator layers.

        Args:
            config: LayoutGAN++ model configuration.

        Examples:
            >>> model = LayoutGANPPModel(LayoutGANPPConfig())
            >>> model.base_model_prefix
            'layoutganpp'
        """
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
        latents: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> LayoutGANPPModelOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass from latents and label IDs.

        Args:
            latents: Per-element latent vectors shaped `(batch, sequence, latent_size)`.
            labels: Label IDs shaped `(batch, sequence)`.
            attention_mask: Optional mask where true values mark valid labels.
            padding_mask: Optional mask where true values mark padded labels.
            return_dict: Whether to return a `LayoutGANPPModelOutput`.

        Returns:
            Model output dataclass or tuple containing boxes, labels, and mask.

        Raises:
            ValueError: If labels or latents have invalid shape or label IDs.

        Examples:
            >>> model = LayoutGANPPModel(LayoutGANPPConfig(num_labels=2))
            >>> labels = torch.tensor([[0, 1]])
            >>> latents = torch.zeros(1, 2, model.config.latent_size)
            >>> tuple(model(latents=latents, labels=labels).bbox.shape)
            (1, 2, 4)
        """
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
        condition_type: ConditionType | str = ConditionType.label,
        bbox: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        latents: torch.Tensor | None = None,
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate layouts from label conditions.

        Args:
            batch_size: Requested batch size; label shape determines the final value.
            condition_type: Condition type or alias. LayoutGAN++ supports label conditions.
            bbox: Reserved compatibility argument.
            labels: Required label IDs for generation.
            mask: Optional valid-element mask.
            attention_mask: Optional valid-element mask.
            num_elements: Reserved compatibility argument.
            box_format: Reserved compatibility argument.
            normalized: Reserved compatibility argument.
            canvas_size: Reserved compatibility argument.
            seed: Optional random seed for latent sampling.
            generator: Optional PyTorch random generator.
            num_inference_steps: Reserved compatibility argument.
            output_type: Return format, either `dataclass` or `dict`.
            return_intermediates: Whether to include generation intermediates.
            latents: Optional fixed latent vectors.
            **model_kwargs: Additional keyword arguments, rejected if present.

        Returns:
            A layout generation dataclass or dictionary.

        Raises:
            ValueError: If labels are missing, generation options are unsupported,
                or output type is invalid.

        Examples:
            >>> model = LayoutGANPPModel(LayoutGANPPConfig(num_labels=2))
            >>> out = model.generate(labels=torch.tensor([[0, 1]]), seed=0)
            >>> tuple(out.bbox.shape)
            (1, 2, 4)
        """
        del bbox, num_elements, normalized, canvas_size, num_inference_steps
        normalize_box_format(box_format)
        if model_kwargs:
            unknown = ", ".join(sorted(model_kwargs))
            raise ValueError(f"Unsupported generation kwargs: {unknown}")
        canonical = normalize_condition_type(condition_type)
        if canonical is ConditionType.unconditional:
            raise ValueError(
                "layoutganpp v1 requires labels; unconditional is unsupported"
            )
        if canonical is not ConditionType.label:
            raise ValueError(f"Unsupported condition_type for layoutganpp: {canonical}")
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
        assert isinstance(out, LayoutGANPPModelOutput)
        assert out.labels is not None
        assert out.mask is not None
        assert out.latents is not None
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
        resolved_output_type = normalize_output_type(output_type)
        if resolved_output_type is OutputType.dict:
            return dict(layout)
        if resolved_output_type is OutputType.dataclass:
            return layout
        assert_never(resolved_output_type)

    def _sample_latents(
        self,
        shape: tuple[int, int, int],
        *,
        seed: int | None,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if generator is None and seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
