"""Pipeline wrapper for Coarse-to-Fine."""

from __future__ import annotations

from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import Pipeline
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .hierarchy import decode_hierarchy_from_logits
from .modeling_coarse_to_fine import CoarseToFineForLayoutGeneration
from .processing_coarse_to_fine import CoarseToFineProcessor
from .types import OutputType, normalize_output_type


class CoarseToFinePipeline(Pipeline):
    """Orchestrate Coarse-to-Fine layout generation."""

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
        labels: Int[torch.Tensor, "batch elements"] | list[object] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"] | list[object] | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | list[object] | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        latent_z: Float[torch.Tensor, "1 batch latent"] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate layouts through model decode and processor post-processing."""
        del labels, bbox, mask, num_elements
        del box_format, normalized, canvas_size, num_inference_steps
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.unconditional:
            raise NotImplementedError(
                "Coarse-to-Fine released checkpoints support only unconditional generation"
            )
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
        if latent_z is None:
            sampled_z = self.model._sample_latent(
                batch_size=batch_size,
                generator=generator,
                device=self.model.device,
            )
        else:
            sampled_z = cast(torch.FloatTensor, latent_z.to(self.model.device))
        raw = self.model._decode_hierarchy(sampled_z)
        hierarchy = decode_hierarchy_from_logits(
            group_bbox_logits=raw["group_bounding_box_logits"],
            group_label_logits=raw["label_in_one_group_logits"],
            grouped_bbox_logits=raw["grouped_bbox_logits"],
            grouped_label_logits=raw["grouped_label_logits"],
            num_labels=self.model.config.num_labels,
            group_eos_index=self.model.config.group_eos_index,
            element_eos_id=self.model.config.element_eos_id,
            discrete_x_grid=self.model.config.discrete_x_grid,
            discrete_y_grid=self.model.config.discrete_y_grid,
        )
        output = cast(
            LayoutGenerationOutput,
            self.processor.post_process_hierarchy(
                hierarchy,
                output_type=OutputType.dataclass,
                return_intermediates=return_intermediates,
            ),
        )
        output.sequences = cast(torch.Tensor, hierarchy.discrete_relative_bbox)
        output.scores = None
        output.trajectory = raw if return_intermediates else None
        if normalize_output_type(output_type) is OutputType.dict:
            return dict(output)
        return output
