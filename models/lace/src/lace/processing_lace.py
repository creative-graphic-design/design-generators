from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config

from layout_generation_common.bbox import ltrb_to_xywh, ltwh_to_xywh, normalize_boxes
from layout_generation_common.outputs import LayoutGenerationOutput

from .configuration_lace import get_dataset_spec


class LaceProcessor(ConfigMixin):
    config_name = "processor_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset: str,
        labels: list[str],
        max_seq_length: int = 25,
    ) -> None:
        self.dataset = dataset
        self.labels = tuple(labels)
        self.max_seq_length = max_seq_length

    @classmethod
    def from_dataset(cls, dataset: str) -> "LaceProcessor":
        spec = get_dataset_spec(dataset)
        return cls(
            dataset=spec.dataset,
            labels=list(spec.labels),
            max_seq_length=spec.max_seq_length,
        )

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def pad_label_id(self) -> int:
        return len(self.labels)

    @property
    def num_classes_with_pad(self) -> int:
        return len(self.labels) + 1

    @property
    def seq_dim(self) -> int:
        return self.num_classes_with_pad + 4

    def __call__(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list,
        labels: torch.Tensor | np.ndarray | list,
        mask: torch.Tensor | np.ndarray | list | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        if labels_t.ndim == 1:
            labels_t = labels_t.unsqueeze(0)
            bbox_t = bbox_t.unsqueeze(0)
        if mask is None:
            mask_t = torch.ones(labels_t.shape, dtype=torch.bool)
        else:
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
        if not normalized:
            if canvas_size is None:
                raise ValueError("canvas_size is required when normalized=False")
            bbox_t = normalize_boxes(
                bbox_t, canvas_size=canvas_size, box_format=box_format
            )
        elif box_format == "ltwh":
            bbox_t = ltwh_to_xywh(bbox_t)
        elif box_format == "ltrb":
            bbox_t = ltrb_to_xywh(bbox_t)
        elif box_format != "xywh":
            raise ValueError(f"Unsupported box_format: {box_format}")
        bbox_t, labels_t, mask_t = self.pad(bbox_t, labels_t, mask_t)
        return {
            "layout": self.encode(bbox_t, labels_t, mask_t),
            "bbox": bbox_t,
            "labels": labels_t,
            "mask": mask_t,
        }

    def pad(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_seq_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max_seq_length or self.max_seq_length
        if bbox.shape[1] > max_len:
            raise ValueError(f"LACE supports at most {max_len} elements")
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
        pad_count = max_len - bbox.shape[1]
        if pad_count:
            bbox_pad = torch.zeros(
                bbox.shape[0], pad_count, 4, dtype=bbox.dtype, device=bbox.device
            )
            label_pad = torch.full(
                (labels.shape[0], pad_count),
                self.pad_label_id,
                dtype=labels.dtype,
                device=labels.device,
            )
            mask_pad = torch.zeros(
                mask.shape[0], pad_count, dtype=torch.bool, device=mask.device
            )
            bbox = torch.cat((bbox, bbox_pad), dim=1)
            labels = torch.cat((labels, label_pad), dim=1)
            mask = torch.cat((mask, mask_pad), dim=1)
        labels = labels.clone()
        labels[~mask] = self.pad_label_id
        return bbox, labels, mask

    def encode(
        self, bbox: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bbox, labels, mask = self.pad(bbox, labels, mask)
        bbox_in = 2 * (bbox.clamp(0.0, 1.0) - 0.5)
        labels = labels.clamp(0, self.pad_label_id)
        labels[~mask] = self.pad_label_id
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes_with_pad
        ).to(dtype=bbox.dtype, device=bbox.device)
        return torch.cat((one_hot, bbox_in), dim=-1)

    def decode(
        self, layout: torch.Tensor, clamp: bool = True
    ) -> LayoutGenerationOutput:
        decoded = layout.clone()
        bbox_latent = decoded[:, :, self.num_classes_with_pad :]
        if clamp:
            bbox_latent = bbox_latent.clamp(-1.0, 1.0)
        bbox = bbox_latent / 2 + 0.5
        labels = decoded[:, :, : self.num_classes_with_pad].argmax(dim=2).long()
        mask = labels != self.pad_label_id
        return LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=self.id2label,
            intermediates={"dataset": self.dataset},
        )

    def save_pretrained(self, save_directory: str) -> None:
        self.save_config(save_directory)

    @classmethod
    def from_pretrained(cls, path: str) -> "LaceProcessor":
        return cls.from_config(cls.load_config(path))
