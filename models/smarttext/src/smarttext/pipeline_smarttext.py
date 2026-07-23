"""Pipeline interface for SmartText content-image text placement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Literal, cast

import torch
from jaxtyping import Float, Int
from PIL import ImageFont
from transformers import PretrainedConfig
from transformers.image_utils import ImageInput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import (
    ConditionType,
    normalize_condition_type as normalize_shared_condition_type,
)
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .candidate_generation import generate_candidates, prepare_scorer_batch
from .color import choose_text_color
from .configuration_smarttext import SmartTextConfig
from .modeling_basnet import SmartTextBASNet
from .modeling_smarttext import SmartTextScorer
from .processing_smarttext import SmartTextProcessor


class OutputType(StrEnum):
    """Supported SmartText pipeline output containers."""

    dataclass = auto()
    dict = auto()


def normalize_condition_type(
    condition_type: ConditionType | str | None,
) -> ConditionType:
    """Normalize SmartText condition aliases.

    Args:
        condition_type: Canonical condition, alias, or ``None``.

    Returns:
        ``ConditionType.content_image``.

    Raises:
        NotImplementedError: If the requested mode is unsupported by SmartText.
    """
    canonical = (
        ConditionType.content_image
        if condition_type is None
        else normalize_shared_condition_type(condition_type)
    )
    if canonical is not ConditionType.content_image:
        raise NotImplementedError(
            "SmartText requires condition_type='content_image' with image/content and text payloads"
        )
    return canonical


def _load_scorer_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return SmartTextScorer.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return SmartTextScorer.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


def _load_saliency_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return SmartTextBASNet.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return SmartTextBASNet.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return SmartTextProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return SmartTextProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


class SmartTextPipeline(LayoutGenerationPipeline):
    """Transformers-side SmartText pipeline.

    Args:
        scorer: Candidate scoring model.
        saliency_model: BASNet saliency model.
        processor: Input/output processor.
        config: Optional root pipeline config.
        device: Optional runtime device.

    Examples:
        >>> config = SmartTextConfig(align_size=3, reduction_dim=4, grid_num=16, max_font_size=20)
        >>> pipe = SmartTextPipeline(SmartTextScorer(config), SmartTextBASNet(config), config=config)
        >>> pipe.config.model_type
        'smarttext'
    """

    config_class: ClassVar[type[PretrainedConfig]] = SmartTextConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "scorer": PipelineComponentSpec(
            attribute_name="scorer",
            loader=_load_scorer_component,
            marker_file="config.json",
            config_subfolder_attribute="scorer_subfolder",
        ),
        "saliency_model": PipelineComponentSpec(
            attribute_name="saliency_model",
            loader=_load_saliency_component,
            marker_file="config.json",
            config_subfolder_attribute="saliency_subfolder",
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            marker_file="processor_config.json",
            save_with_is_main_process=False,
            config_subfolder_attribute="processor_subfolder",
        ),
    }

    config: SmartTextConfig
    scorer: SmartTextScorer
    saliency_model: SmartTextBASNet
    processor: SmartTextProcessor

    def __init__(
        self,
        scorer: SmartTextScorer,
        saliency_model: SmartTextBASNet,
        processor: SmartTextProcessor | None = None,
        config: SmartTextConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize SmartText pipeline."""
        super().__init__(config or scorer.config)
        self.config = config or scorer.config
        self.scorer = scorer
        self.saliency_model = saliency_model
        self.processor = processor or SmartTextProcessor(config=self.config)
        self.scorer.eval()
        self.saliency_model.eval()
        if device is not None:
            self.to(device)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "SmartTextPipeline":
        """Build a SmartText pipeline from loaded components."""
        return cls(
            config=cast(SmartTextConfig, config),
            scorer=cast(SmartTextScorer, components["scorer"]),
            saliency_model=cast(SmartTextBASNet, components["saliency_model"]),
            processor=cast(SmartTextProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        images: ImageInput
        | Sequence[ImageInput]
        | Float[torch.Tensor, "batch channels height width"]
        | None = None,
        *,
        content: Mapping[str, object] | None = None,
        prompt: str | Sequence[str] | None = None,
        text: str | Sequence[str] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: object = None,
        bbox: object = None,
        mask: object = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | Literal["dataclass", "dict"] = OutputType.dataclass,
        return_intermediates: bool = False,
        font: str | Path | ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
        ratio_list: Sequence[float] | None = None,
        text_spacing: int | None = None,
        candi_res: int | None = None,
        saliency: ImageInput
        | Sequence[ImageInput]
        | Float[torch.Tensor, "batch height width"]
        | None = None,
        candidate_boxes: Sequence[Mapping[str, object]]
        | Sequence[Sequence[Mapping[str, object]]]
        | None = None,
        return_text_lines: bool = False,
        score_normalization: Literal["mos", "raw"] = "mos",
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate text placement boxes for content images.

        Args:
            images: RGB image or image batch.
            content: Optional content carrier.
            prompt: Prompt text payload.
            text: Alias for prompt.
            batch_size: Expected batch size for validation.
            seed: Optional seed used only when ``generator`` is absent.
            generator: Explicit torch generator; wins over ``seed``.
            condition_type: Must normalize to ``content_image``.
            labels: Unsupported v1 compatibility argument.
            bbox: Unsupported v1 compatibility argument.
            mask: Unsupported v1 compatibility argument.
            num_elements: Unsupported v1 compatibility argument.
            box_format: Public v1 compatibility argument.
            normalized: Public v1 compatibility argument.
            canvas_size: Optional source canvas size override.
            num_inference_steps: Unused v1 compatibility argument.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to include intermediate payloads.
            font: TrueType font path or PIL font object.
            ratio_list: Optional per-line font ratios.
            text_spacing: Optional text-spacing override.
            candi_res: Optional top-k override.
            saliency: Optional saliency map bypassing BASNet.
            candidate_boxes: Optional reference-style candidates.
            return_text_lines: Return per-line boxes for top candidate.
            score_normalization: ``mos`` or ``raw``.

        Returns:
            Shared layout output.
        """
        del (
            labels,
            bbox,
            mask,
            num_elements,
            box_format,
            normalized,
            canvas_size,
            num_inference_steps,
        )
        normalize_condition_type(condition_type)
        self.prepare_generator(generator=generator, seed=seed, device=self.device)
        if font is None:
            font = ImageFont.load_default()
        effective_config = self.config
        if text_spacing is not None:
            effective_config = copy.copy(self.config)
            effective_config.text_spacing = int(text_spacing)
        encoded = self.processor(
            images,
            content=content,
            prompt=prompt,
            text=text,
            saliency=saliency,
            candidate_boxes=candidate_boxes,
            font=font,
        )
        if len(encoded["images"]) != 1 or batch_size != 1:
            raise ValueError("SmartText currently decodes one image at a time")
        image = encoded["images"][0]
        prompt_text = encoded["prompts"][0]
        resolved_saliency = encoded["saliency"]
        if resolved_saliency is None:
            basnet_values = encoded["basnet_pixel_values"].to(self._runtime_device())
            saliency_out = self.saliency_model(basnet_values)
            resolved_saliency = saliency_out.saliency[0]
        candidates = encoded["candidate_boxes"]
        if candidates is None:
            candidates = generate_candidates(
                image,
                cast(torch.Tensor, resolved_saliency),
                prompt=prompt_text,
                font=font,
                config=effective_config,
                ratio_list=ratio_list,
            )
        pixel_values, boxes, candidates = prepare_scorer_batch(
            image,
            candidates,
            config=effective_config,
        )
        scores = self.scorer(
            pixel_values.to(self._runtime_device()),
            boxes.to(self._runtime_device()),
        ).scores
        top_k = candi_res or self.config.candi_res
        raw_scores = scores.detach().cpu().float().flatten()
        if candidates:
            selected_color_index = max(
                range(len(candidates)), key=lambda index: float(raw_scores[index])
            )
            text_color = choose_text_color(
                image,
                candidates[selected_color_index].bbox_ltrb_px,
                contrast_threshold=self.config.contrast_threshold,
            )
        else:
            text_color = None
        intermediates = None
        if return_intermediates:
            intermediates = {
                "saliency": resolved_saliency,
                "raw_scorer_boxes": boxes.detach().cpu(),
                "prompt": prompt_text,
            }
        return self.processor.decode(
            candidates=candidates,
            scores=scores,
            image_size=image.size,
            output_type=cast(Literal["dataclass", "dict"], str(output_type)),
            return_text_lines=return_text_lines,
            top_k=top_k,
            score_normalization=score_normalization,
            text_color=text_color,
            intermediates=intermediates,
        )

    def _runtime_device(self) -> torch.device:
        return self.device or next(self.scorer.parameters()).device
