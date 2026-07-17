"""Pipeline wrapper for LayoutFormer++."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Pipeline

from layout_generation_common.outputs import LayoutGenerationOutput

from .modeling_layoutformerpp import LayoutFormerPPForConditionalGeneration
from .processing_layoutformerpp import LayoutFormerPPProcessor


class LayoutFormerPPPipeline(Pipeline):
    """Small ergonomic pipeline around model `generate_layout`."""

    model: LayoutFormerPPForConditionalGeneration
    processor: LayoutFormerPPProcessor

    def __init__(
        self,
        model: LayoutFormerPPForConditionalGeneration,
        processor: LayoutFormerPPProcessor,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, tokenizer=processor.tokenizer, **kwargs)
        self.processor = processor

    def _sanitize_parameters(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return {}, kwargs, {}

    def preprocess(self, inputs: Any = None, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def _forward(
        self, model_inputs: dict[str, Any], **kwargs: Any
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        return self.model.generate_layout(processor=self.processor, **model_inputs)

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        return model_outputs

    def __call__(
        self,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: str = "unconditional",
        labels: list[list[int | str]] | None = None,
        bbox: Any = None,
        relations: list[list[tuple[int, int, int, int, int]]] | None = None,
        num_elements: int | list[int] | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        output_type: str = "dataclass",
        return_intermediates: bool = False,
        **generate_kwargs: Any,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        return self.model.generate_layout(
            processor=self.processor,
            batch_size=batch_size,
            seed=seed,
            generator=generator,
            condition_type=condition_type,
            labels=labels,
            bbox=bbox,
            relations=relations,
            box_format=box_format,
            output_type=output_type,
            num_elements=num_elements,
            normalized=normalized,
            return_intermediates=return_intermediates,
            **generate_kwargs,
        )
