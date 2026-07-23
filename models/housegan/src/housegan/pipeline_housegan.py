"""Pipeline interface for House-GAN relation-conditioned generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import torch
from jaxtyping import Float
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import (
    LayoutGenerationPipeline,
    PipelineComponentSpec,
    model_processor_component_specs,
)

from .configuration_housegan import HouseGanConfig
from .graph_schema import HouseGanSceneGraph
from .modeling_housegan import HouseGanGenerator
from .processing_housegan import HouseGanProcessor, OutputType


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is None:
        return HouseGanGenerator.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return HouseGanGenerator.from_pretrained(
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
        return HouseGanProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
    return HouseGanProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


class HouseGanPipeline(LayoutGenerationPipeline):
    """Transformers-side House-GAN layout generation pipeline."""

    config_class: ClassVar[type[PretrainedConfig]] = HouseGanConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = (
        model_processor_component_specs(
            model_loader=_load_model_component,
            processor_loader=_load_processor_component,
        )
    )

    config: HouseGanConfig
    model: HouseGanGenerator
    processor: HouseGanProcessor

    def __init__(
        self,
        model: HouseGanGenerator,
        processor: HouseGanProcessor | None = None,
        config: HouseGanConfig | None = None,
        device: int | torch.device | None = None,
    ) -> None:
        """Initialize the pipeline."""
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or HouseGanProcessor(
            id2label=self.config.id2label,
            relation_id2label=self.config.relation_id2label,
            canvas_size=cast(tuple[int, int], self.config.canvas_size),
            mask_size=self.config.mask_size,
        )
        if device is not None:
            resolved = (
                torch.device("cpu")
                if isinstance(device, int) and device < 0
                else torch.device(f"cuda:{device}")
                if isinstance(device, int)
                else device
            )
            self.to(resolved)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "HouseGanPipeline":
        """Build a pipeline from loaded model and processor components."""
        return cls(
            config=cast(HouseGanConfig, config),
            model=cast(HouseGanGenerator, components["model"]),
            processor=cast(HouseGanProcessor, components["processor"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.relation,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        scene_graph: HouseGanSceneGraph
        | Mapping[str, object]
        | list[Mapping[str, object]]
        | None = None,
        relations: object | None = None,
        latents: Float[torch.Tensor, "elements latent"] | None = None,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate a floorplan layout from room relation constraints."""
        del num_elements, num_inference_steps
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        graph_batch = _expand_graph_batch(scene_graph, batch_size)
        outputs: list[LayoutGenerationOutput] = []
        torch_generator = self.prepare_generator(
            generator=generator,
            seed=seed,
            device=self.device or next(self.model.parameters()).device,
        )
        for graph_index, graph_item in enumerate(graph_batch):
            condition = self.processor(
                condition_type=condition_type,
                scene_graph=graph_item,
                relations=relations,
                labels=labels,
                bbox=bbox,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
            node_features = condition["node_features"].to(self.model.device)
            edges = condition["edges"].to(self.model.device)
            labels_t = condition["labels"].to(self.model.device)
            room_count = node_features.shape[0]
            graph_latents = latents
            if graph_latents is None:
                graph_latents = torch.randn(
                    room_count,
                    self.model.config.latent_dim,
                    generator=torch_generator,
                    device=self.model.device,
                    dtype=next(self.model.parameters()).dtype,
                )
            elif graph_latents.ndim == 3:
                graph_latents = graph_latents[graph_index]
            model_output = self.model(
                latents=graph_latents.to(self.model.device),
                node_features=node_features,
                edges=edges,
            )
            decoded = self.processor.post_process_masks(
                model_output.masks,
                labels=labels_t,
                edges=edges,
                node_features=node_features,
                scene_graph=condition["scene_graph"],
                output_type="dataclass",
                return_intermediates=return_intermediates,
            )
            outputs.append(cast(LayoutGenerationOutput, decoded))
        merged = _merge_outputs(outputs, output_type=output_type)
        return merged


def _expand_graph_batch(
    scene_graph: HouseGanSceneGraph
    | Mapping[str, object]
    | list[Mapping[str, object]]
    | None,
    batch_size: int,
) -> list[HouseGanSceneGraph | Mapping[str, object] | None]:
    if isinstance(scene_graph, list):
        return cast(list[HouseGanSceneGraph | Mapping[str, object] | None], scene_graph)
    return [scene_graph for _ in range(batch_size)]


def _merge_outputs(
    outputs: list[LayoutGenerationOutput],
    *,
    output_type: OutputType,
) -> LayoutGenerationOutput | dict[str, object]:
    if len(outputs) == 1:
        output = outputs[0]
    else:
        max_elements = max(item.labels.shape[1] for item in outputs)
        bbox_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for item in outputs:
            pad = max_elements - item.labels.shape[1]
            bbox_rows.append(
                torch.nn.functional.pad(cast(torch.Tensor, item.bbox), (0, 0, 0, pad))
            )
            label_rows.append(
                torch.nn.functional.pad(cast(torch.Tensor, item.labels), (0, pad))
            )
            mask_rows.append(
                torch.nn.functional.pad(
                    cast(torch.Tensor, item.mask), (0, pad), value=False
                )
            )
        output = LayoutGenerationOutput(
            bbox=torch.cat(bbox_rows, dim=0),
            labels=torch.cat(label_rows, dim=0),
            mask=torch.cat(mask_rows, dim=0),
            id2label=outputs[0].id2label,
            intermediates=[item.intermediates for item in outputs],
        )
    if output_type == "dict":
        return dict(output)
    return output
