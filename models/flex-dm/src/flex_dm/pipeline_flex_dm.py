"""Transformers-side pipeline for Flex-DM infilling."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, model_processor_component_specs

from .configuration_flex_dm import FlexDmConfig
from .masking import apply_token, iterative_decode
from .modeling_flex_dm import FlexDmForMaskedDocumentModeling, FlexDmModelOutput
from .processing_flex_dm import FlexDmProcessor


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return FlexDmForMaskedDocumentModeling.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return FlexDmForMaskedDocumentModeling.from_pretrained(
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
        return FlexDmProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return FlexDmProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


class FlexDmPipeline(LayoutGenerationPipeline):
    """Run Flex-DM completion, refinement, and feature-level content infilling."""

    config_class: ClassVar[type[PretrainedConfig]] = FlexDmConfig
    component_specs: ClassVar = model_processor_component_specs(
        model_loader=_load_model_component,
        processor_loader=_load_processor_component,
    )

    config: FlexDmConfig
    model: FlexDmForMaskedDocumentModeling
    processor: FlexDmProcessor

    def __init__(
        self,
        model: FlexDmForMaskedDocumentModeling,
        processor: FlexDmProcessor | None = None,
        config: FlexDmConfig | None = None,
    ) -> None:
        """Initialize model and processor components."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or FlexDmProcessor.from_config(self.config)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "FlexDmPipeline":
        """Build a pipeline from loaded model and processor components."""
        return cls(
            config=cast(FlexDmConfig, config),
            model=cast(FlexDmForMaskedDocumentModeling, components["model"]),
            processor=cast(FlexDmProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.completion,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | list[object]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        attributes: Mapping[str, object] | None = None,
        content: Mapping[str, object] | None = None,
        feature_group: str | None = None,
        target_indices: Int[torch.Tensor, "..."] | None = None,
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Infills masked Flex-DM document fields.

        Args:
            batch_size: Batch size used when synthetic empty inputs are created.
            seed: Common API compatibility argument. Flex-DM's public inference
                path is deterministic and does not currently consume randomness.
            generator: Common API compatibility argument. When supplied, it
                takes precedence over ``seed``; the deterministic Flex-DM
                inference path does not currently consume it.
            condition_type: Canonical condition or local task alias.
            labels: Public element labels.
            bbox: Public element boxes.
            mask: Public valid-element mask.
            num_elements: Optional element counts for synthetic inputs.
            box_format: Input box coordinate format.
            normalized: Whether input boxes are already normalized.
            canvas_size: Pixel canvas size when ``normalized=False``.
            num_inference_steps: Number of iterative decode steps.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to include logits and masks.
            attributes: Optional non-core document attributes.
            content: Optional Crello image/text embeddings.
            feature_group: Flex-DM task group such as ``pos`` or ``img``.
            target_indices: Optional element indexes for ``elem`` masking.
            model_kwargs: Reserved model keyword arguments.

        Returns:
            Common layout-generation output.

        Raises:
            NotImplementedError: If the requested canonical condition is not
                supported by released Flex-DM MFP checkpoints.
        """
        _ = model_kwargs
        model_device = next(self.model.parameters()).device
        if generator is None and seed is not None:
            generator = torch.Generator(device=model_device).manual_seed(seed)
        encoded = self.processor(
            condition_type=condition_type,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            attributes=attributes,
            content=content,
            feature_group=feature_group,
            target_indices=target_indices,
            batch_size=batch_size,
        )
        canonical = cast(ConditionType, encoded["condition_type"])
        feature = cast(str | None, encoded["feature_group"])
        self._validate_condition(canonical, feature)
        inputs = {
            key: value.to(model_device)
            for key, value in cast(dict[str, torch.Tensor], encoded["inputs"]).items()
        }
        masks = {
            key: value.to(model_device)
            for key, value in cast(dict[str, torch.Tensor], encoded["masks"]).items()
        }
        masked_inputs = self._apply_masks(inputs, masks, generator=generator)
        was_training = self.model.training
        self.model.eval()
        try:
            if num_inference_steps is not None and num_inference_steps > 1:
                outputs = cast(
                    FlexDmModelOutput,
                    iterative_decode(
                        self.model,
                        inputs=masked_inputs,
                        masks=masks,
                        num_iter=num_inference_steps,
                        input_columns=self.config.input_columns,
                        source_inputs=inputs,
                    ),
                )
            else:
                outputs = self.model(
                    inputs=masked_inputs, masks=masks, return_dict=True
                )
        finally:
            self.model.train(was_training)
        return self.processor.post_process_document(
            outputs,
            original_inputs=inputs,
            masks=masks,
            output_type=output_type,
            return_intermediates=return_intermediates,
            refinement_input=inputs if canonical is ConditionType.refinement else None,
        )

    generate = __call__

    def _apply_masks(
        self,
        inputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        *,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        modified = dict(inputs)
        for key, mask in masks.items():
            if key not in self.config.input_columns:
                continue
            column = self.config.input_columns[key]
            if column["is_sequence"] and mask.ndim == 2 and mask.any():
                modified[key] = apply_token(
                    modified[key],
                    column,
                    mask,
                    "masked",
                    generator=generator,
                )
        return modified

    def _validate_condition(
        self,
        condition_type: ConditionType,
        feature_group: str | None,
    ) -> None:
        if condition_type in {ConditionType.completion, ConditionType.refinement}:
            return
        if condition_type is ConditionType.content_image and feature_group in {
            "img",
            "txt",
        }:
            return
        if condition_type is ConditionType.label:
            raise NotImplementedError(
                "Flex-DM has no standalone label-conditioned mode; use "
                'condition_type="completion", feature_group="type".'
            )
        if condition_type is ConditionType.label_size:
            raise NotImplementedError("Flex-DM does not support label_size generation")
        if condition_type is ConditionType.unconditional:
            raise NotImplementedError(
                "Flex-DM released MFP checkpoints require an input document"
            )
        raise NotImplementedError(
            f"Flex-DM does not support condition_type={condition_type}"
        )
