"""Processor for CGB-DM content images and layout tensors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import IO, Literal, cast

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat, prepare_layout_tensors
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from posgen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)

CGBDM_LAYOUT_KEY = "layout"
CGBDM_BBOX_KEY = "bbox"
CGBDM_LABELS_KEY = "labels"
CGBDM_MASK_KEY = "mask"


class CGBDMProcessor(ProcessorMixin):
    """Prepare RGB/saliency inputs and decode CGB-DM layout tensors.

    Args:
        dataset_name: Poster/content dataset key.
        id2label: Public id-to-label mapping excluding invalid/pad.
        num_labels: Internal class-channel count.
        max_seq_length: Maximum number of elements.
        image_size: Resize target as ``(height, width)``.

    Examples:
        >>> CGBDMProcessor().seq_dim
        8
    """

    config_name = "processor_config.json"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.pku_posterlayout,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        num_labels: int = 4,
        max_seq_length: int = 16,
        image_size: tuple[int, int] | list[int] = (384, 256),
    ) -> None:
        """Initialize processor metadata."""
        super().__init__()
        dataset = normalize_dataset_name(dataset_name)
        labels = id2label_for_dataset(dataset)
        public_labels = {
            key: value for key, value in labels.items() if value != "INVALID"
        }
        self.dataset_name = str(dataset)
        self.id2label = {int(k): v for k, v in (id2label or public_labels).items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = int(num_labels)
        self.max_seq_length = int(max_seq_length)
        self.image_size: tuple[int, int] = (int(image_size[0]), int(image_size[1]))
        self.chat_template = None

    @property
    def seq_dim(self) -> int:
        """Return the internal class-plus-box channel count."""
        return self.num_labels + 4

    def __call__(
        self,
        images: object | None = None,
        *,
        saliency: object | None = None,
        saliency_isnet: object | None = None,
        saliency_basnet: object | None = None,
        saliency_box: torch.Tensor | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode image and saliency inputs into model tensors."""
        if return_tensors != "pt":
            raise ValueError("CGBDMProcessor only supports return_tensors='pt'")
        image_rows = _ensure_batch(images)
        saliency_rows = self._resolve_saliency(
            len(image_rows),
            saliency=saliency,
            saliency_isnet=saliency_isnet,
            saliency_basnet=saliency_basnet,
        )
        pixel_values = []
        boxes = []
        for image, sal in zip(image_rows, saliency_rows, strict=True):
            rgb = _to_rgb_tensor(image, self.image_size)
            sal_tensor = (
                torch.zeros(1, *self.image_size)
                if sal is None
                else _to_l_tensor(sal, self.image_size)
            )
            pixel_values.append(torch.cat((rgb, sal_tensor), dim=0))
            boxes.append(_saliency_box_from_tensor(sal_tensor))
        resolved_box = (
            torch.stack(boxes)
            if saliency_box is None
            else torch.as_tensor(saliency_box, dtype=torch.float32)
        )
        if resolved_box.ndim == 1:
            resolved_box = resolved_box.reshape(1, 1, 4)
        elif resolved_box.ndim == 2:
            resolved_box = resolved_box.unsqueeze(1)
        return BatchEncoding(
            {
                "pixel_values": torch.stack(pixel_values),
                "saliency_box": 2 * (resolved_box.clamp(0.0, 1.0) - 0.5),
            }
        )

    def encode_layout(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[object],
        labels: torch.Tensor | np.ndarray | list[object],
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode public layout tensors into CGB-DM latent layout format."""
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            clamp_converted_normalized=True,
        )
        bbox_t, labels_t, mask_t = self.pad(bbox_t, labels_t, mask_t)
        layout = self.encode(bbox_t, labels_t, mask_t)
        return {
            CGBDM_LAYOUT_KEY: layout,
            CGBDM_BBOX_KEY: bbox_t,
            CGBDM_LABELS_KEY: labels_t,
            CGBDM_MASK_KEY: mask_t,
        }

    def pad(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad public layout tensors to ``max_seq_length``."""
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
        if bbox.shape[1] > self.max_seq_length:
            raise ValueError(f"CGB-DM supports at most {self.max_seq_length} elements")
        pad_count = self.max_seq_length - bbox.shape[1]
        if pad_count:
            bbox = torch.nn.functional.pad(bbox, (0, 0, 0, pad_count))
            labels = torch.nn.functional.pad(labels, (0, pad_count))
            mask = torch.nn.functional.pad(mask, (0, pad_count))
        labels = labels.clone()
        labels[~mask] = 0
        return bbox, labels, mask

    def encode(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode normalized boxes and public labels into internal tensors."""
        bbox, labels, mask = self.pad(bbox, labels, mask)
        internal_labels = labels.clone().clamp_min(0)
        internal_labels[~mask] = 0
        if self.dataset_name == str(DatasetName.pku_posterlayout):
            invalid_id = 3
            internal_labels = torch.where(mask, internal_labels, invalid_id)
        one_hot = torch.nn.functional.one_hot(
            internal_labels.clamp(0, self.num_labels - 1),
            num_classes=self.num_labels,
        ).to(dtype=bbox.dtype, device=bbox.device)
        bbox_in = 2 * (bbox.clamp(0.0, 1.0) - 0.5)
        return torch.cat((one_hot, bbox_in), dim=-1)

    def decode(
        self,
        layout: torch.Tensor,
        *,
        output_type: Literal["dataclass", "dict"] | str = "dataclass",
        scores: torch.Tensor | None = None,
        intermediates: object | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode internal layout tensors into the public schema."""
        bbox = (layout[:, :, self.num_labels :].clamp(-1.0, 1.0) / 2 + 0.5).cpu()
        class_logits = layout[:, :, : self.num_labels]
        class_ids = class_logits.argmax(dim=-1).long().cpu()
        invalid_ids = {0}
        if self.dataset_name == str(DatasetName.pku_posterlayout):
            invalid_ids.add(3)
        mask = torch.ones_like(class_ids, dtype=torch.bool)
        for invalid_id in invalid_ids:
            mask &= class_ids != invalid_id
        labels = class_ids.clamp(0, max(self.id2label)).cpu()
        resolved_scores = (
            scores
            if scores is not None
            else class_logits.softmax(dim=-1).max(dim=-1).values
        )
        output = LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=dict(self.id2label),
            scores=resolved_scores.detach().cpu(),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(output)
        if output_type == "dataclass":
            return output
        raise ValueError(f"Unsupported output_type: {output_type}")

    def _resolve_saliency(
        self,
        batch_size: int,
        *,
        saliency: object | None,
        saliency_isnet: object | None,
        saliency_basnet: object | None,
    ) -> list[object | None]:
        if saliency is not None:
            rows = _ensure_batch(saliency)
            if len(rows) != batch_size:
                raise ValueError("saliency batch size must match images")
            return rows
        if saliency_isnet is None and saliency_basnet is None:
            return [None] * batch_size
        return cast(
            list[object | None],
            [
                _merge_saliency_pair(left, right, self.image_size)
                for left, right in zip(
                    _optional_batch(saliency_isnet, batch_size),
                    _optional_batch(saliency_basnet, batch_size),
                    strict=True,
                )
            ],
        )


def _ensure_batch(value: object | None) -> list[object]:
    if value is None:
        raise ValueError("images or pixel_values are required")
    if isinstance(value, torch.Tensor):
        if value.ndim in {2, 3}:
            return [value]
        if value.ndim == 4:
            return [row for row in value]
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _optional_batch(value: object | None, batch_size: int) -> list[object | None]:
    if value is None:
        return [None] * batch_size
    rows = _ensure_batch(value)
    if len(rows) != batch_size:
        raise ValueError("saliency batch size must match images")
    return rows


def _to_rgb_tensor(value: object, image_size: tuple[int, int]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.float()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("RGB tensor must be CHW or HWC")
        tensor = torch.nn.functional.interpolate(
            tensor, size=image_size, mode="bilinear", align_corners=False
        )[0]
        return _normalize_zero_one_tensor(tensor)
    image = (
        value
        if isinstance(value, Image.Image)
        else Image.open(cast(str | bytes | Path | IO[bytes], value))
    )
    image = image.convert("RGB").resize(
        (image_size[1], image_size[0]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2 - 1


def _to_l_tensor(value: object, image_size: tuple[int, int]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("saliency tensor must be HW or single-channel")
        tensor = torch.nn.functional.interpolate(
            tensor, size=image_size, mode="bilinear", align_corners=False
        )[0]
        return _normalize_zero_one_tensor(tensor)
    image = (
        value
        if isinstance(value, Image.Image)
        else Image.open(cast(str | bytes | Path | IO[bytes], value))
    )
    image = image.convert("L").resize(
        (image_size[1], image_size[0]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0) * 2 - 1


def _normalize_zero_one_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        tensor = tensor / 255.0
    return tensor * 2 - 1


def _saliency_box_from_tensor(saliency: torch.Tensor) -> torch.Tensor:
    binary = saliency[0] > saliency[0].mean()
    if not bool(binary.any()):
        return torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
    ys, xs = torch.where(binary)
    height, width = saliency.shape[-2:]
    left = xs.min().float() / width
    right = (xs.max().float() + 1) / width
    top = ys.min().float() / height
    bottom = (ys.max().float() + 1) / height
    return torch.tensor(
        [[(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]]
    )


def _merge_saliency_pair(
    left: object | None,
    right: object | None,
    image_size: tuple[int, int],
) -> torch.Tensor:
    if left is None:
        return _to_l_tensor(right, image_size)
    if right is None:
        return _to_l_tensor(left, image_size)
    return torch.maximum(
        _to_l_tensor(left, image_size), _to_l_tensor(right, image_size)
    )
