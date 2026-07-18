"""Pipeline wrapper for Coarse-to-Fine."""

from __future__ import annotations

import torch
from transformers import Pipeline
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput

from .modeling_coarse_to_fine import CoarseToFineForLayoutGeneration
from .processing_coarse_to_fine import CoarseToFineProcessor
from .types import OutputType


class CoarseToFinePipeline(Pipeline):
    """Ergonomic pipeline around ``generate_layout``."""

    model: CoarseToFineForLayoutGeneration
    processor: CoarseToFineProcessor

    def __init__(
        self,
        model: CoarseToFineForLayoutGeneration,
        processor: CoarseToFineProcessor,
    ) -> None:
        """Initialize the pipeline with a model and processor."""
        super().__init__(model=model, tokenizer=None)
        self.processor = processor

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(
        self, input_: object, **preprocess_parameters: dict[str, object]
    ) -> dict[str, GenericTensor]:
        """Satisfy the abstract pipeline API; direct calls bypass this path."""
        _ = (input_, preprocess_parameters)
        return {}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        **forward_parameters: dict[str, object],
    ) -> ModelOutput:
        _ = (input_tensors, forward_parameters)
        raise NotImplementedError("Use CoarseToFinePipeline.__call__ directly")

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, object],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return model outputs without additional formatting."""
        _ = kwargs
        return model_outputs

    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: object = None,
        bbox: object = None,
        mask: object = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        latent_z: torch.FloatTensor | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate layouts through the wrapped model and processor."""
        return self.model.generate_layout(
            processor=self.processor,
            batch_size=batch_size,
            seed=seed,
            generator=generator,
            latent_z=latent_z,
            condition_type=condition_type,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_intermediates=return_intermediates,
        )
