"""Pipeline wrapper for RALF layout generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, cast

import torch
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_ralf import RalfConfig
from .modeling_ralf import RalfForConditionalLayoutGeneration
from .processing_ralf import RalfProcessor
from .retrieval import RalfRetrievalTable


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return RalfForConditionalLayoutGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return RalfForConditionalLayoutGeneration.from_pretrained(
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
        return RalfProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return RalfProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


class RalfPipeline(LayoutGenerationPipeline):
    """Compose a RALF model and processor for content-aware retrieval generation."""

    config_class: ClassVar[type[PretrainedConfig]] = RalfConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            loader=_load_model_component,
            marker_file="config.json",
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            marker_file="processor_config.json",
            save_with_is_main_process=False,
        ),
    }

    config: RalfConfig
    model: RalfForConditionalLayoutGeneration
    processor: RalfProcessor

    def __init__(
        self,
        model: RalfForConditionalLayoutGeneration,
        processor: RalfProcessor | None = None,
        config: RalfConfig | None = None,
    ) -> None:
        """Initialize the RALF pipeline."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or RalfProcessor.from_config(self.config)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "RalfPipeline":
        """Build a pipeline from checkpoint components."""
        return cls(
            config=cast(RalfConfig, config),
            model=cast(RalfForConditionalLayoutGeneration, components["model"]),
            processor=cast(RalfProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        images: object = None,
        saliency: object = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: torch.Tensor
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: torch.Tensor | Sequence[object] | None = None,
        mask: torch.Tensor | Sequence[object] | None = None,
        num_elements: int | Sequence[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        retrieved_layouts: Mapping[str, object] | None = None,
        retrieved_images: object = None,
        retrieved_saliency: object = None,
        retrieved_indexes: torch.Tensor | Sequence[Sequence[int]] | None = None,
        retrieval: Mapping[str, object] | None = None,
        retrieval_table: RalfRetrievalTable | None = None,
        query_ids: Sequence[int | str] | None = None,
        relations: object = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate layouts through the RALF public interface.

        Args:
            images: Poster/content images.
            saliency: Optional saliency maps.
            batch_size: Batch size when images are absent.
            seed: Convenience seed used only when `generator` is absent.
            generator: PyTorch generator; takes precedence over `seed`.
            condition_type: Canonical condition type or alias.
            labels: Optional label constraints.
            bbox: Optional box constraints.
            mask: Optional valid-element mask.
            num_elements: Optional requested element counts.
            box_format: Input box format.
            normalized: Whether boxes are normalized.
            canvas_size: Canvas size for pixel boxes.
            retrieved_layouts: Explicit retrieved layouts.
            retrieved_images: Explicit retrieved images.
            retrieved_saliency: Explicit retrieved saliency maps.
            retrieved_indexes: Explicit retrieved indexes.
            retrieval: Canonical v2 retrieval container.
            retrieval_table: Optional model-side retrieval table.
            query_ids: Query ids used for table lookup when explicit examples are absent.
            relations: Optional relation constraints.
            num_inference_steps: Reserved v1 argument.
            output_type: `dataclass` or `dict`.
            return_intermediates: Whether to return retrieval debug metadata.
            temperature: Sampling temperature.
            top_k: Optional top-k sampling limit.

        Returns:
            LayoutGenerationOutput or dictionary.
        """
        _ = (num_elements, relations, num_inference_steps)
        condition = normalize_condition_type(condition_type)
        if condition not in {
            ConditionType.unconditional,
            ConditionType.label,
            ConditionType.label_size,
            ConditionType.completion,
            ConditionType.refinement,
            ConditionType.relation,
            ConditionType.retrieval,
            ConditionType.content_image,
        }:
            raise NotImplementedError(
                f"RALF does not support condition_type={condition}"
            )
        encoded = self.processor(
            images=images,
            saliency=saliency,
            condition_type=condition,
            labels=labels,
            bbox=bbox,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            retrieved_layouts=retrieved_layouts,
            retrieved_images=retrieved_images,
            retrieved_saliency=retrieved_saliency,
            retrieved_indexes=retrieved_indexes,
            retrieval=retrieval,
            batch_size=batch_size,
        )
        model_device = next(self.model.parameters()).device
        generation_generator = self.prepare_generator(
            generator=generator,
            seed=seed,
            device=model_device,
        )
        intermediates: dict[str, object] = {}
        if "retrieval" in encoded:
            retrieval_batch = encoded["retrieval"]
            if retrieval_batch.indexes is not None:
                intermediates["retrieval"] = {"indexes": retrieval_batch.indexes}
        elif retrieval_table is not None and query_ids is not None:
            intermediates["retrieval"] = {"indexes": retrieval_table.lookup(query_ids)}
        sequences = self.model._generate_sequences(
            encoded["input_ids"].to(model_device),
            pixel_values=encoded["pixel_values"].to(model_device),
            saliency=encoded["saliency"].to(model_device),
            attention_mask=encoded["attention_mask"].to(model_device),
            max_length=self.config.max_token_length,
            temperature=temperature,
            top_k=top_k,
            generator=generation_generator,
            token_mask=self.processor.layout_tokenizer.token_mask(model_device),
        )
        return self.processor.post_process_layouts(
            sequences.cpu(),
            output_type=output_type,
            intermediates=intermediates if return_intermediates else None,
        )

    generate = __call__
