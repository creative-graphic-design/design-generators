"""Pipeline wrapper for LayoutAction generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, cast

import numpy as np
import torch
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import (
    LayoutGenerationPipeline,
    PipelineComponentSpec,
    model_processor_component_specs,
)

from .configuration_layout_action import LayoutActionConfig
from .modeling_layout_action import LayoutActionForCausalLM
from .processing_layout_action import LayoutActionProcessor, OutputType


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutActionForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutActionForCausalLM.from_pretrained(
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
        return LayoutActionProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutActionProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class LayoutActionPipeline(LayoutGenerationPipeline):
    """Compose a LayoutAction model and processor for layout generation.

    Args:
        model: Converted LayoutAction causal LM.
        processor: Matching processor/tokenizer.
        config: Optional root pipeline config. Defaults to ``model.config``.

    Examples:
        >>> config = LayoutActionConfig(n_layer=1, n_head=2, n_embd=16, max_elements=1)
        >>> pipe = LayoutActionPipeline(
        ...     model=LayoutActionForCausalLM(config),
        ...     processor=LayoutActionProcessor(LayoutActionTokenizer(config)),
        ...     config=config,
        ... )
        >>> pipe.config.model_type
        'layout-action'
    """

    config_class: ClassVar[type[PretrainedConfig]] = LayoutActionConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = (
        model_processor_component_specs(
            model_loader=_load_model_component,
            processor_loader=_load_processor_component,
        )
    )

    config: LayoutActionConfig
    model: LayoutActionForCausalLM
    processor: LayoutActionProcessor

    def __init__(
        self,
        model: LayoutActionForCausalLM,
        processor: LayoutActionProcessor,
        config: LayoutActionConfig | None = None,
    ) -> None:
        """Initialize the pipeline."""
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
    ) -> "LayoutActionPipeline":
        """Build a pipeline from loaded components."""
        return cls(
            config=cast(LayoutActionConfig, config),
            model=cast(LayoutActionForCausalLM, components["model"]),
            processor=cast(LayoutActionProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
        sampling: Literal["greedy", "multinomial", "top_k"] = "top_k",
        temperature: float = 1.0,
        top_k: int | None = 5,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate a layout through the public LayoutAction interface."""
        encoded = self.processor(
            condition_type=condition_type,
            bbox=bbox,
            labels=labels,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            batch_size=batch_size,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        prepared_generator = self.prepare_generator(
            generator=generator,
            seed=seed,
            device=model_device,
        )
        input_ids = encoded["input_ids"].to(model_device)
        forced_token_ids = encoded.get("forced_token_ids")
        if isinstance(forced_token_ids, torch.Tensor):
            forced_token_ids = forced_token_ids.to(model_device)
        max_new_tokens = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(encoded["max_new_tokens"])
        )
        was_training = self.model.training
        self.model.eval()
        try:
            sequences = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=sampling != "greedy",
                forced_token_ids=forced_token_ids,
                generator=prepared_generator,
            )
        finally:
            self.model.train(was_training)
        return self.processor.post_process_layouts(
            sequences,
            output_type=output_type,
            return_intermediates=return_intermediates,
        )
