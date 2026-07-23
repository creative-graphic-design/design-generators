"""Pipeline interface for PosterLayout DS-GAN generation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, cast

import torch
from transformers import PretrainedConfig
from transformers.image_utils import ImageInput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import (
    ConditionType,
    normalize_condition_type as normalize_shared_condition_type,
)
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_ds_gan import DSGANConfig
from .modeling_ds_gan import DSGANModel, DSGANModelOutput, random_initial_layout
from .processing_ds_gan import DSGANProcessor


class OutputType(StrEnum):
    """Supported DS-GAN pipeline output containers."""

    dataclass = auto()
    dict = auto()


_SUPPORTED_CONDITION_TYPES: frozenset[ConditionType] = frozenset(
    {ConditionType.content_image}
)
_DSGAN_CONDITION_ALIASES: dict[str, ConditionType] = {
    "content_aware": ConditionType.content_image,
    "image_saliency": ConditionType.content_image,
}


def normalize_condition_type(
    condition_type: ConditionType | str | None,
) -> ConditionType:
    """Normalize DS-GAN condition aliases.

    Args:
        condition_type: Canonical condition enum, alias, or ``None``.

    Returns:
        Canonical ``content_image`` condition.

    Raises:
        ValueError: If DS-GAN does not support the requested mode.

    Examples:
        >>> str(normalize_condition_type("content"))
        'content_image'
    """
    if condition_type is None:
        canonical = ConditionType.content_image
    elif isinstance(condition_type, ConditionType):
        canonical = condition_type
    else:
        key = condition_type.lower().replace("-", "_")
        canonical = _DSGAN_CONDITION_ALIASES.get(key)
        if canonical is None:
            canonical = normalize_shared_condition_type(condition_type)
    if canonical not in _SUPPORTED_CONDITION_TYPES:
        raise ValueError(f"Unsupported DS-GAN condition_type: {condition_type}")
    return canonical


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize public output type aliases."""
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return DSGANModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return DSGANModel.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return DSGANProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return DSGANProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class DSGANPipeline(LayoutGenerationPipeline):
    """Transformers-side pipeline for content-aware PosterLayout generation.

    Args:
        model: DS-GAN generator.
        processor: Optional processor for images and output decoding.
        config: Optional root pipeline config.
        device: Optional runtime device.

    Examples:
        >>> config = DSGANConfig(backbone="resnet18", max_elem=4, hidden_size=32, num_layers=2, image_size=(64, 64), backbone_feature_size=16)
        >>> pipe = DSGANPipeline(DSGANModel(config))
        >>> pipe.config.model_type
        'ds_gan'
    """

    config_class: ClassVar[type[PretrainedConfig]] = DSGANConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            loader=_load_model_component,
            marker_file="config.json",
            config_subfolder_attribute="model_subfolder",
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            marker_file="processor_config.json",
            save_with_is_main_process=False,
            config_subfolder_attribute="processor_subfolder",
        ),
    }

    config: DSGANConfig
    model: DSGANModel
    processor: DSGANProcessor

    def __init__(
        self,
        model: DSGANModel,
        processor: DSGANProcessor | None = None,
        config: DSGANConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize DS-GAN pipeline."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or DSGANProcessor(
            dataset_name=self.config.dataset_name,
            id2label=cast(dict[int | str, str], self.config.id2label),
            image_size=cast(tuple[int, int], self.config.image_size),
        )
        self.model.eval()
        if device is not None:
            self.to(device)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "DSGANPipeline":
        """Build a pipeline from loaded root components."""
        return cls(
            config=cast(DSGANConfig, config),
            model=cast(DSGANModel, components["model"]),
            processor=cast(DSGANProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: torch.Tensor | list[object] | None = None,
        bbox: torch.Tensor | list[object] | None = None,
        mask: torch.Tensor | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        saliency: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        saliency_pfpnet: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        saliency_basnet: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        initial_layout: torch.Tensor | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate layouts from content images and saliency maps.

        Args:
            images: RGB image or batch. Required unless ``pixel_values`` is given.
            batch_size: Batch size used when ``pixel_values`` is supplied.
            seed: Convenience seed. Ignored when ``generator`` is supplied.
            generator: Explicit torch generator.
            condition_type: Must normalize to ``content_image``.
            labels: Optional public labels used only with ``bbox`` to provide a
                fixed initial layout.
            bbox: Optional public boxes used with ``labels`` for a fixed layout.
            mask: Optional valid-element mask for fixed initial layouts.
            num_elements: Reserved compatibility argument.
            box_format: Format of optional ``bbox``.
            normalized: Whether optional ``bbox`` is normalized.
            canvas_size: Pixel canvas size for optional unnormalized ``bbox``.
            num_inference_steps: Reserved compatibility argument.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to include raw model tensors.
            saliency: Optional single merged saliency map.
            saliency_pfpnet: Optional PFPNet saliency map.
            saliency_basnet: Optional BASNet saliency map.
            pixel_values: Preprocessed ``(B, 4, H, W)`` tensor.
            initial_layout: Optional internal layout ``(B, max_elem, 2, 4)``.

        Returns:
            Shared layout-generation output.

        Raises:
            ValueError: If the condition, image inputs, or fixed layout are invalid.
        """
        del num_elements, num_inference_steps
        canonical = normalize_condition_type(condition_type)
        resolved_output_type = normalize_output_type(output_type)
        device = self.device or next(self.model.parameters()).device
        if pixel_values is None:
            if images is None:
                raise ValueError("images or pixel_values are required for DS-GAN")
            encoded = self.processor(
                images,
                saliency=saliency,
                saliency_pfpnet=saliency_pfpnet,
                saliency_basnet=saliency_basnet,
            )
            pixel_values = cast(torch.Tensor, encoded["pixel_values"])
        pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)
        batch_size = pixel_values.shape[0] if pixel_values is not None else batch_size
        if initial_layout is None and bbox is not None and labels is not None:
            encoded_layout = self.processor.encode_layout(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
                max_elem=self.config.max_elem,
            )
            initial_layout = encoded_layout["layout"]
        if initial_layout is None:
            prepared = self.prepare_generator(
                generator=generator, seed=seed, device=device
            )
            initial_layout = random_initial_layout(
                batch_size,
                self.config.max_elem,
                generator=prepared,
                device=device,
                dtype=self.model.dtype,
            )
        else:
            initial_layout = initial_layout.to(device=device, dtype=self.model.dtype)
        model_output = self.model(
            pixel_values=pixel_values,
            layout=initial_layout,
            return_dict=True,
        )
        assert isinstance(model_output, DSGANModelOutput)
        intermediates = None
        if return_intermediates:
            intermediates = {
                "condition_type": canonical,
                "initial_layout": initial_layout.detach().cpu(),
                "class_probs": model_output.class_probs.detach().cpu(),
            }
        return self.processor.decode(
            class_probs=model_output.class_probs,
            bbox=model_output.bbox,
            output_type=resolved_output_type.value,
            intermediates=intermediates,
        )
