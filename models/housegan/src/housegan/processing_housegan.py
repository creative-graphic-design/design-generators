"""Processor for House-GAN relation graphs and mask decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, Self, cast

import torch
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    ltrb_to_xywh,
    normalize_box_format,
    prepare_layout_tensors,
)
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_housegan import HouseGanConfig
from .graph_schema import (
    HouseGanSceneGraph,
    complete_signed_edges,
    graph_to_node_features,
    normalize_scene_graph,
    relation_from_bboxes,
)

OutputType = Literal["dataclass", "dict"]
Id2LabelMapping = Mapping[int, str] | Mapping[str, str]


class HouseGanProcessor(ProcessorMixin):
    """Normalize House-GAN scene graphs and decode generated masks."""

    attributes: list[str] = []
    config_name = "processor_config.json"

    def __init__(
        self,
        *,
        config: HouseGanConfig,
        default_missing_relation: Literal["not_adjacent", "error"] = "not_adjacent",
    ) -> None:
        """Initialize processor metadata."""
        self.config = config
        self.id2label = {
            int(key): value
            for key, value in cast(Id2LabelMapping, self.config.id2label).items()
        }
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.relation_id2label = {
            int(key): value
            for key, value in cast(
                Id2LabelMapping, self.config.relation_id2label
            ).items()
        }
        self.canvas_size = tuple(self.config.canvas_size)
        self.mask_size = self.config.mask_size
        self.default_missing_relation = default_missing_relation
        self.chat_template = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata."""
        del push_to_hub, kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "processor_class": self.__class__.__name__,
            "id2label": self.id2label,
            "relation_id2label": self.relation_id2label,
            "canvas_size": self.canvas_size,
            "mask_size": self.mask_size,
            "default_missing_relation": self.default_missing_relation,
        }
        (root / self.config_name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        subfolder: str | None = None,
        **kwargs: object,
    ) -> Self:
        """Load processor metadata from ``processor_config.json``."""
        del cache_dir, force_download, local_files_only, token, revision, kwargs
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        payload = json.loads((root / cls.config_name).read_text(encoding="utf-8"))
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise TypeError("processor config payload must be a dictionary")
        return cls(
            config=HouseGanConfig.from_dict(config_payload),
            default_missing_relation=payload.get(
                "default_missing_relation", "not_adjacent"
            ),
        )

    def __call__(
        self,
        *,
        condition_type: ConditionType | str = ConditionType.relation,
        scene_graph: HouseGanSceneGraph | Mapping[str, object] | None = None,
        relations: object | None = None,
        labels: object | None = None,
        bbox: object | None = None,
        mask: object | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode public relation inputs into House-GAN tensors."""
        if return_tensors != "pt":
            raise ValueError("HouseGanProcessor only supports return_tensors='pt'")
        self.normalize_condition_type(condition_type)
        relation_payload = relations
        if relation_payload is None and bbox is not None and labels is not None:
            bbox_t, labels_t, _ = prepare_layout_tensors(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=normalize_box_format(box_format),
                normalized=normalized,
                canvas_size=canvas_size or self.canvas_size,
                clamp_converted_normalized=True,
            )
            relation_payload = relation_from_bboxes(
                _xywh_to_ltrb_list(bbox_t[0]),
            )
            labels = labels_t[0]
        graph = normalize_scene_graph(
            scene_graph,
            labels=labels,
            relations=relation_payload,
            id2label=self.id2label,
        )
        if self.default_missing_relation == "error" and not graph.relations:
            raise ValueError(
                "House-GAN requires relations when missing-pair policy is 'error'"
            )
        node_features = graph_to_node_features(
            graph.nodes,
            label2id=self.label2id,
            num_labels=len(self.id2label),
        )
        edges = complete_signed_edges(
            graph.nodes,
            graph.relations,
            default_adjacent=False,
        )
        label_ids = torch.tensor(
            [
                self.label2id[node.label]
                if isinstance(node.label, str)
                else int(node.label)
                for node in graph.nodes
            ],
            dtype=torch.long,
        )
        return BatchEncoding(
            {
                "node_features": node_features,
                "edges": edges,
                "labels": label_ids,
                "scene_graph": graph,
            }
        )

    def normalize_condition_type(
        self, condition_type: ConditionType | str
    ) -> ConditionType:
        """Normalize and validate the House-GAN condition type."""
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.relation:
            raise NotImplementedError(
                "House-GAN supports only condition_type='relation' and aliases "
                "'scene_graph', 'graph', or 'gen_r'."
            )
        return condition

    def post_process_masks(
        self,
        masks: torch.Tensor,
        *,
        labels: torch.LongTensor,
        edges: torch.LongTensor | None = None,
        node_features: torch.Tensor | None = None,
        scene_graph: object | None = None,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Convert generated room masks to public normalized boxes."""
        bbox_ltrb = mask_to_ltrb(masks, threshold=0.0)
        bbox_xywh = (
            ltrb_to_xywh(bbox_ltrb / float(self.mask_size)).unsqueeze(0).clamp(0.0, 1.0)
        )
        labels_b = labels.to(dtype=torch.long).unsqueeze(0)
        valid = torch.ones(labels_b.shape, dtype=torch.bool, device=labels_b.device)
        intermediates = None
        if return_intermediates:
            intermediates = {
                "room_masks": masks.detach(),
                "bbox_ltrb_32": bbox_ltrb,
                "signed_edges": edges,
                "node_features": node_features,
                "scene_graph": scene_graph,
            }
        if output_type == "dict":
            return {
                "bbox": bbox_xywh,
                "labels": labels_b,
                "mask": valid,
                "id2label": self.id2label,
                "sequences": edges,
                "scores": None,
                "trajectory": None,
                "intermediates": intermediates,
            }
        return LayoutGenerationOutput(
            bbox=bbox_xywh,
            labels=labels_b,
            mask=valid,
            id2label=self.id2label,
            sequences=edges,
            intermediates=intermediates,
        )


def mask_to_ltrb(masks: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
    """Convert thresholded masks to inclusive-exclusive ``ltrb`` boxes."""
    boxes: list[list[float]] = []
    for mask in masks.detach().cpu():
        inds = torch.nonzero(mask > threshold, as_tuple=False)
        if inds.numel() == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        y0 = int(inds[:, 0].min().item())
        x0 = int(inds[:, 1].min().item())
        y1 = int(inds[:, 0].max().item())
        x1 = int(inds[:, 1].max().item())
        boxes.append([float(x0), float(y0), float(x1 + 1), float(y1 + 1)])
    return torch.tensor(boxes, dtype=torch.float32, device=masks.device)


def _xywh_to_ltrb_list(bbox: torch.Tensor) -> list[list[float]]:
    x, y, w, h = bbox.unbind(dim=-1)
    ltrb = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)
    return cast(list[list[float]], ltrb.detach().cpu().tolist())
