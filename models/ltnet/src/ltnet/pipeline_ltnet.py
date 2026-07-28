"""Pipeline wrapper for LT-Net relation-to-layout generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_ltnet import LTNetConfig
from .modeling_ltnet import LTNetForLayoutGeneration
from .processing_ltnet import LTNetProcessor, OutputType
from .relation_schema import LayoutObject, LayoutRelation, SceneGraphInput

SceneGraphMapping = Mapping[
    str,
    str
    | int
    | float
    | Sequence[float]
    | Sequence[Mapping[str, str | int | float | Sequence[float]]]
    | Mapping[int | str, str],
]


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> LTNetForLayoutGeneration:
    if subfolder is not None:
        return LTNetForLayoutGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LTNetForLayoutGeneration.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> LTNetProcessor:
    if subfolder is not None:
        return LTNetProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LTNetProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class LTNetPipeline(LayoutGenerationPipeline):
    """Compose an LT-Net model and processor for scene-graph layout inference.

    Args:
        model: Converted LT-Net model.
        processor: Matching scene-graph processor/tokenizer.
        config: Optional root pipeline config. Defaults to ``model.config``.

    Examples:
        >>> processor = LTNetProcessor.from_config()
        >>> config = LTNetConfig(
        ...     vocab_size=processor.tokenizer.vocab_size,
        ...     hidden_size=32,
        ...     num_hidden_layers=1,
        ...     num_attention_heads=4,
        ... )
        >>> pipe = LTNetPipeline(
        ...     model=LTNetForLayoutGeneration(config),
        ...     processor=processor,
        ... )
        >>> pipe.config.model_type
        'ltnet'
    """

    config_class: ClassVar[type[PretrainedConfig]] = LTNetConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            loader=_load_model_component,
            marker_file="config.json",
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            marker_file="preprocessor_config.json",
            save_with_is_main_process=False,
        ),
    }

    config: LTNetConfig
    model: LTNetForLayoutGeneration
    processor: LTNetProcessor

    def __init__(
        self,
        model: LTNetForLayoutGeneration,
        processor: LTNetProcessor,
        config: LTNetConfig | None = None,
    ) -> None:
        """Initialize the pipeline with model and processor components."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor

    @classmethod
    def _from_pretrained_components(  # ty: ignore[invalid-method-override]
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, LTNetForLayoutGeneration | LTNetProcessor | None],
    ) -> "LTNetPipeline":
        """Build a pipeline from loaded root components."""
        return cls(
            config=cast(LTNetConfig, config),
            model=cast(LTNetForLayoutGeneration, components["model"]),
            processor=cast(LTNetProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.relation,
        labels: Int[torch.Tensor, "batch elements"] | Sequence[int] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Sequence[Sequence[float]]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | Sequence[bool] | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
        scene_graph: SceneGraphInput | SceneGraphMapping | None = None,
        objects: Sequence[LayoutObject] | None = None,
        relations: Sequence[LayoutRelation] | None = None,
    ) -> (
        LayoutGenerationOutput
        | dict[
            str,
            Float[torch.Tensor, "..."]
            | Int[torch.Tensor, "..."]
            | Bool[torch.Tensor, "..."]
            | dict[int, str]
            | dict[str, Float[torch.Tensor, "..."] | None],
        ]
    ):
        """Generate layouts from a public scene graph.

        Args:
            batch_size: Number of layouts to generate from the same graph.
            seed: Convenience seed used only when ``generator`` is absent.
            generator: Optional PyTorch generator; takes precedence over seed.
            condition_type: Must normalize to ``relation``.
            labels: Reserved v1 interface input; LT-Net uses ``scene_graph``.
            bbox: Reserved v1 box constraint input.
            mask: Reserved v1 validity-mask input.
            num_elements: Reserved v1 element-count input.
            box_format: Output box format; LT-Net returns normalized ``xywh``.
            normalized: Whether output boxes should be normalized.
            canvas_size: Reserved denormalization canvas size.
            num_inference_steps: Reserved v1 step count.
            output_type: Return dataclass or dict.
            return_intermediates: Include raw logits/boxes in intermediates.
            scene_graph: Public relation payload.
            objects: Object-node shorthand when ``scene_graph`` is omitted.
            relations: Relation-edge shorthand when ``scene_graph`` is omitted.

        Returns:
            Layout generation output dataclass or dict.

        Raises:
            ValueError: If the condition or graph payload is unsupported.
        """
        _ = (labels, bbox, mask, num_elements, num_inference_steps)
        model_device = next(self.model.parameters()).device
        prepared_generator = self.prepare_generator(
            generator=generator, seed=seed, device=model_device
        )
        encoded = self.processor(
            scene_graph=scene_graph,
            objects=objects,
            relations=relations,
            batch_size=batch_size,
            condition_type=condition_type,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(model_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model._generate_boxes(
                **model_inputs,
                generator=prepared_generator,
            )
        finally:
            self.model.train(was_training)
        return self.processor.post_process_layout_generation(
            output,
            input_token=model_inputs["input_token"],
            input_obj_id=model_inputs["input_obj_id"],
            token_type=model_inputs["token_type"],
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            output_type=output_type,
            return_intermediates=return_intermediates,
        )
