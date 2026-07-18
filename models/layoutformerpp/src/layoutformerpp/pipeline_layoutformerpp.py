"""Pipeline wrapper for LayoutFormer++."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

import torch
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_layoutformerpp import LayoutFormerPPConfig
from .modeling_layoutformerpp import LayoutFormerPPForConditionalGeneration
from .processing_layoutformerpp import LayoutFormerPPProcessor
from .tasks import OutputType


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutFormerPPForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutFormerPPForConditionalGeneration.from_pretrained(
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
        return LayoutFormerPPProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutFormerPPProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class LayoutFormerPPPipeline(LayoutGenerationPipeline):
    """Compose a LayoutFormer++ model and processor for layout generation.

    Args:
        model: Converted LayoutFormer++ model.
        processor: Matching processor/tokenizer.
        config: Optional root pipeline config. Defaults to `model.config`.

    Examples:
        >>> processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
        >>> config = LayoutFormerPPConfig(vocab_size=processor.tokenizer.vocab_size)
        >>> pipe = LayoutFormerPPPipeline(
        ...     model=LayoutFormerPPForConditionalGeneration(config),
        ...     processor=processor,
        ... )
        >>> pipe.config.model_type
        'layoutformerpp'
    """

    config_class: ClassVar[type[PretrainedConfig]] = LayoutFormerPPConfig
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

    config: LayoutFormerPPConfig
    model: LayoutFormerPPForConditionalGeneration
    processor: LayoutFormerPPProcessor

    def __init__(
        self,
        model: LayoutFormerPPForConditionalGeneration,
        processor: LayoutFormerPPProcessor,
        config: LayoutFormerPPConfig | None = None,
    ) -> None:
        """Initialize the pipeline with model and processor components."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "LayoutFormerPPPipeline":
        """Build a pipeline from loaded root components."""
        return cls(
            config=cast(LayoutFormerPPConfig, config),
            model=cast(LayoutFormerPPForConditionalGeneration, components["model"]),
            processor=cast(LayoutFormerPPProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: list[list[int | str]] | None = None,
        bbox: object = None,
        mask: torch.Tensor | None = None,
        relations: list[list[tuple[int, int, int, int, int]]] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        max_length: int | None = None,
        do_sample: bool | None = None,
        top_k: int = 10,
        temperature: float = 0.7,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate layouts by encoding conditions, generating ids, and decoding.

        Args:
            batch_size: Number of layouts to generate when labels are omitted.
            seed: Convenience seed used only when `generator` is absent.
            generator: Optional PyTorch generator; takes precedence over `seed`.
            condition_type: Canonical condition type or supported alias.
            labels: Optional label conditions.
            bbox: Optional layout boxes for size/completion/refinement conditions.
            mask: Reserved public validity mask input.
            relations: Optional relation tuples for relation-conditioned checkpoints.
            num_elements: Reserved v1 interface argument.
            box_format: Input and output bounding-box format.
            normalized: Whether public boxes are normalized.
            canvas_size: Reserved v1 interface argument.
            num_inference_steps: Reserved v1 interface argument.
            output_type: Return `dataclass` or `dict`.
            return_intermediates: Reserved output detail flag.
            max_length: Optional token decode length override.
            do_sample: Optional sampling override.
            top_k: Top-k value used by the vendor sampling loop.
            temperature: Sampling temperature.

        Returns:
            Layout generation output dataclass or dictionary.

        Raises:
            ValueError: If processor inputs are invalid.
        """
        _ = (num_elements, num_inference_steps, return_intermediates)
        encoded = self.processor(
            condition_type=condition_type,
            batch_size=batch_size,
            return_tensors="pt",
            labels=labels,
            bbox=bbox,
            mask=mask,
            relations=relations,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        model_device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        condition = self.processor.normalize_condition_type(condition_type)
        generation_generator = self.prepare_generator(
            generator=generator,
            seed=seed,
            device=model_device,
        )
        default_do_sample = condition in {
            ConditionType.unconditional,
            ConditionType.completion,
        }
        sequences = self.model._generate_sequences(
            input_ids,
            attention_mask,
            max_length=max_length,
            do_sample=default_do_sample if do_sample is None else do_sample,
            top_k=top_k,
            temperature=temperature,
            generator=generation_generator,
        )
        return self.processor.post_process_layouts(
            sequences.cpu(),
            box_format=box_format,
            output_type=output_type,
        )
