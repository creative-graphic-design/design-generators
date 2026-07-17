"""Pipeline wrapper for LayoutFormer++."""

from __future__ import annotations

import torch
from transformers import Pipeline
from transformers.utils import ModelOutput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.common.outputs import LayoutGenerationOutput

from .modeling_layoutformerpp import LayoutFormerPPForConditionalGeneration
from .processing_layoutformerpp import LayoutFormerPPProcessor, OutputType


class LayoutFormerPPPipeline(Pipeline):
    """Small ergonomic pipeline around model `generate_layout`."""

    model: LayoutFormerPPForConditionalGeneration
    processor: LayoutFormerPPProcessor

    def __init__(
        self,
        model: LayoutFormerPPForConditionalGeneration,
        processor: LayoutFormerPPProcessor,
        **kwargs: object,
    ) -> None:
        """Initialize the pipeline with a model and matching processor."""
        _ = kwargs
        super().__init__(model=model, tokenizer=processor.tokenizer)
        self.processor = processor

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(  # type: ignore
        self, inputs: object = None, **kwargs: object
    ) -> dict[str, object]:
        """Forward keyword inputs to the model step unchanged."""
        return kwargs

    def _forward(  # type: ignore
        self, model_inputs: dict[str, object], **kwargs: object
    ) -> ModelOutput:
        _ = (model_inputs, kwargs)
        raise NotImplementedError("Use LayoutFormerPPPipeline.__call__ directly")

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, object],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return model outputs without additional formatting."""
        return model_outputs

    def __call__(
        self,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: list[list[int | str]] | None = None,
        bbox: object = None,
        relations: list[list[tuple[int, int, int, int, int]]] | None = None,
        num_elements: int | list[int] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        max_length: int | None = None,
        do_sample: bool | None = None,
        top_k: int = 10,
        temperature: float = 0.7,
        **generate_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate layouts through the wrapped model and processor."""
        _ = generate_kwargs
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
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
        )
