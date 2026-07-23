"""Pipeline interface for LayoutDETR content-image layout generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, cast

import torch
from transformers import PretrainedConfig
from transformers.image_utils import ImageInput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import (
    ConditionType,
    normalize_condition_type as normalize_shared_condition_type,
)
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import (
    LayoutGenerationPipeline,
    PipelineComponentSpec,
    model_processor_component_specs,
)

from .configuration_layout_detr import BackgroundPreprocessing, LayoutDetrConfig
from .modeling_layout_detr import LayoutDetrForConditionalGeneration
from .postprocessing import PostprocessingMode, apply_postprocessing
from .processing_layout_detr import LayoutDetrProcessor


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutDetrForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutDetrForConditionalGeneration.from_pretrained(
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
        return LayoutDetrProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutDetrProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


def normalize_condition_type(condition_type: ConditionType | str) -> ConditionType:
    """Normalize LayoutDETR condition modes."""
    canonical = normalize_shared_condition_type(condition_type)
    if canonical is not ConditionType.content_image:
        raise NotImplementedError(
            "LayoutDETR supports only condition_type='content_image' with image, texts, and labels"
        )
    return canonical


class LayoutDetrPipeline(LayoutGenerationPipeline):
    """Transformers-side LayoutDETR pipeline."""

    config_class: ClassVar[type[PretrainedConfig]] = LayoutDetrConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = (
        model_processor_component_specs(
            model_loader=_load_model_component,
            processor_loader=_load_processor_component,
        )
    )

    config: LayoutDetrConfig
    model: LayoutDetrForConditionalGeneration
    processor: LayoutDetrProcessor

    def __init__(
        self,
        model: LayoutDetrForConditionalGeneration,
        processor: LayoutDetrProcessor | None = None,
        config: LayoutDetrConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a LayoutDETR pipeline."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or LayoutDetrProcessor(config=self.config)
        self.model.eval()
        if device is not None:
            self.to(device)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "LayoutDetrPipeline":
        """Build a pipeline from saved components."""
        return cls(
            config=cast(LayoutDetrConfig, config),
            model=cast(LayoutDetrForConditionalGeneration, components["model"]),
            processor=cast(LayoutDetrProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        images: ImageInput | Sequence[ImageInput] | torch.Tensor | None = None,
        *,
        content: Mapping[str, object] | None = None,
        prompt: str | Sequence[str] | None = None,
        texts: Sequence[Sequence[str]] | Sequence[str] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: torch.Tensor
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: torch.Tensor | None = None,
        mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None = None,
        num_elements: int | Sequence[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        background_preprocessing: BackgroundPreprocessing
        | str = BackgroundPreprocessing.none,
        out_jittering_strength: float = 0.0,
        out_postprocessing: PostprocessingMode | str = PostprocessingMode.none,
        latents: torch.Tensor | None = None,
    ) -> LayoutGenerationOutput:
        """Generate layouts for a background image and per-element text labels."""
        normalize_condition_type(condition_type)
        if bbox is not None:
            raise ValueError("LayoutDETR does not condition on existing bbox")
        if num_elements is not None:
            raise ValueError("LayoutDETR infers num_elements from labels/mask")
        if box_format != BoxFormat.xywh and box_format != "xywh":
            raise ValueError("LayoutDETR outputs normalized xywh boxes only")
        if not normalized:
            raise ValueError("LayoutDETR expects normalized public boxes")
        if num_inference_steps is not None:
            raise ValueError("LayoutDETR is a single forward pass, not iterative")

        encoded = self.processor(
            images=images,
            content=content,
            prompt=prompt,
            texts=texts,
            labels=labels,
            mask=mask,
            condition_type=str(ConditionType.content_image),
            background_preprocessing=background_preprocessing,
            batch_size=batch_size,
            canvas_size=canvas_size,
        )
        device = self.device or next(self.model.parameters()).device
        encoded = encoded.to(device)
        batch, elements = encoded["bbox_labels"].shape
        runtime_generator = self.prepare_generator(
            generator=generator,
            seed=seed,
            device=device,
        )
        if latents is None:
            latents = torch.randn(
                (batch, elements, self.config.z_dim),
                generator=runtime_generator,
                device=device,
            )
        else:
            latents = latents.to(device=device)
        model_output = self.model(
            pixel_values=encoded["pixel_values"],
            input_ids=encoded["input_ids"],
            text_attention_mask=encoded["text_attention_mask"],
            bbox_labels=encoded["bbox_labels"],
            layout_mask=encoded["layout_mask"],
            latents=latents,
            text_lengths=encoded["text_lengths"],
        )
        bbox_out = apply_postprocessing(
            model_output.bbox,
            model_output.mask,
            mode=out_postprocessing,
            jitter_strength=out_jittering_strength,
            generator=runtime_generator,
        )
        intermediates = {
            "latents": latents.detach().cpu(),
            "texts": encoded["texts"],
            "background_preprocessing": str(background_preprocessing),
            "postprocessing": str(out_postprocessing),
        }
        return cast(
            LayoutGenerationOutput,
            self.processor.post_process_layouts(
                bbox_out,
                model_output.labels,
                model_output.mask,
                output_type=output_type,
                return_intermediates=return_intermediates,
                intermediates=intermediates,
            ),
        )
