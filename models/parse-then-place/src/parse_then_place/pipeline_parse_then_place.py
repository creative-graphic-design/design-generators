"""Pipeline wrapper for Parse-Then-Place."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from transformers import Pipeline
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput

from .modeling_parse_then_place import ParseThenPlaceForConditionalGeneration
from .processing_parse_then_place import ParseThenPlaceProcessor


class ParseThenPlacePipeline(Pipeline):
    """Text-to-layout pipeline for converted Parse-Then-Place checkpoints."""

    model: ParseThenPlaceForConditionalGeneration
    processor: ParseThenPlaceProcessor

    def __init__(
        self,
        model: ParseThenPlaceForConditionalGeneration,
        processor: ParseThenPlaceProcessor,
    ) -> None:
        """Initialize the pipeline with model and processor."""
        super().__init__(
            model=model,
            tokenizer=processor.parser_tokenizer,  # ty: ignore[invalid-argument-type]
        )
        self.processor = processor

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(
        self,
        input_: object,
        **preprocess_parameters: object,
    ) -> dict[str, GenericTensor]:
        """Satisfy the abstract pipeline API; direct calls bypass this path."""
        _ = (input_, preprocess_parameters)
        return {}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        **forward_parameters: object,
    ) -> ModelOutput:
        _ = (input_tensors, forward_parameters)
        raise NotImplementedError("Use ParseThenPlacePipeline.__call__ directly")

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, object],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return model outputs without extra formatting."""
        _ = kwargs
        return model_outputs

    def __call__(
        self,
        *,
        prompt: str | Sequence[str] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.text,
        labels: torch.Tensor | list[object] | None = None,
        bbox: torch.Tensor | list[object] | None = None,
        mask: torch.Tensor | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        num_return_sequences: int | None = None,
        temperature: float | None = None,
        output_candidate: Literal["first", "all", "best"] = "first",
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        layout_text: str | list[str] | list[list[str]] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate a layout from natural-language text."""
        return self.model.generate_layout(
            processor=self.processor,
            prompt=prompt
            if prompt is None or isinstance(prompt, str)
            else list(prompt),
            batch_size=batch_size,
            seed=seed,
            generator=generator,
            condition_type=condition_type,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=str(box_format),
            normalized=normalized,
            canvas_size=canvas_size,
            num_inference_steps=num_inference_steps,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            output_candidate=output_candidate,
            output_type=output_type,
            return_intermediates=return_intermediates,
            layout_text=layout_text,
        )

    generate = __call__
