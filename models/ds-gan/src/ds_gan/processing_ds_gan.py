"""Processor for DS-GAN content-image inputs and layout decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat, normalize_boxes, normalize_box_format
from laygen.modeling_outputs import LayoutGenerationOutput
from posgen.common.labels import DatasetName, normalize_dataset_name


class DSGANProcessor(ProcessorMixin):
    """Prepare PosterLayout RGB/saliency inputs and decode DS-GAN outputs.

    Args:
        dataset_name: Dataset key. Only PKU PosterLayout is supported.
        id2label: Public semantic labels excluding vendor ``no object``.
        image_size: Resize target as ``(height, width)``.

    Examples:
        >>> processor = DSGANProcessor()
        >>> processor.id2label[0]
        'text'
    """

    config_name = "processor_config.json"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.pku_posterlayout,
        id2label: dict[int | str, str] | None = None,
        image_size: tuple[int, int] | list[int] = (350, 240),
    ) -> None:
        """Initialize processor metadata."""
        self.chat_template = None
        dataset = normalize_dataset_name(dataset_name)
        if dataset is not DatasetName.pku_posterlayout:
            raise ValueError(f"Unsupported DS-GAN dataset_name: {dataset_name}")
        self.dataset_name = str(dataset)
        default_id2label = {0: "text", 1: "logo", 2: "underlay"}
        raw_id2label = id2label or default_id2label
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        height, width = image_size
        self.image_size: tuple[int, int] = (int(height), int(width))

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | torch.Tensor,
        *,
        saliency: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        saliency_pfpnet: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        saliency_basnet: ImageInput | list[ImageInput] | torch.Tensor | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode content images into ``pixel_values``.

        Args:
            images: RGB image or batch of RGB images.
            saliency: Optional saliency image or batch. If omitted and both
                saliency maps are given, the maps are merged by pixelwise max.
            saliency_pfpnet: Optional PFPNet saliency map.
            saliency_basnet: Optional BASNet saliency map.
            return_tensors: Tensor framework. Only ``pt`` is supported.

        Returns:
            Batch encoding containing ``pixel_values`` shaped ``(B, 4, H, W)``.
        """
        if return_tensors != "pt":
            raise ValueError("DSGANProcessor only supports return_tensors='pt'")
        image_rows = _ensure_batch(images)
        saliency_rows = self._resolve_saliency(
            len(image_rows),
            saliency=saliency,
            saliency_pfpnet=saliency_pfpnet,
            saliency_basnet=saliency_basnet,
        )
        tensors = []
        for image, sal in zip(image_rows, saliency_rows, strict=True):
            rgb = _to_rgb_tensor(image, self.image_size)
            sal_tensor = (
                torch.zeros(1, *self.image_size, dtype=torch.float32)
                if sal is None
                else _to_l_tensor(sal, self.image_size)
            )
            tensors.append(torch.cat((rgb, sal_tensor), dim=0))
        return BatchEncoding({"pixel_values": torch.stack(tensors)})

    def decode(
        self,
        *,
        class_probs: torch.Tensor,
        bbox: torch.Tensor,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        scores: torch.Tensor | None = None,
        intermediates: object | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode raw DS-GAN class probabilities and boxes.

        Args:
            class_probs: Vendor class probabilities shaped ``(B, E, 4)``.
            bbox: Normalized center ``xywh`` boxes shaped ``(B, E, 4)``.
            output_type: Return format.
            scores: Optional per-element class scores.
            intermediates: Optional model-specific intermediate tensors.

        Returns:
            Shared layout output with public labels and mask semantics.
        """
        class_ids = torch.argmax(class_probs, dim=-1)
        mask = class_ids != 0
        public_labels = (class_ids - 1).clamp_min(0).long()
        resolved_scores = scores
        if resolved_scores is None:
            resolved_scores = class_probs.max(dim=-1).values
        output = LayoutGenerationOutput(
            bbox=bbox.detach().cpu().clamp(0.0, 1.0),
            labels=public_labels.detach().cpu(),
            mask=mask.detach().cpu(),
            id2label=dict(self.id2label),
            scores=resolved_scores.detach().cpu(),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(output)
        if output_type == "dataclass":
            return output
        raise ValueError(f"Unsupported output_type: {output_type}")

    def encode_layout(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[object],
        labels: torch.Tensor | np.ndarray | list[object],
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        max_elem: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Encode public boxes/labels into the vendor layout tensor.

        Args:
            bbox: Public boxes.
            labels: Public zero-based semantic labels.
            mask: Optional valid-element mask.
            box_format: Input box format.
            normalized: Whether the boxes are normalized.
            canvas_size: Pixel canvas size used when ``normalized=False``.
            max_elem: Output slot count.

        Returns:
            Dictionary with vendor ``layout``, normalized ``bbox``, labels, and mask.
        """
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
                bbox_t,
                canvas_size=canvas_size,
                box_format=box_format,
            )
        else:
            normalize_box_format(box_format)
            if box_format != BoxFormat.xywh and str(box_format) != "xywh":
                bbox_t = normalize_boxes(
                    bbox_t,
                    canvas_size=(1, 1),
                    box_format=box_format,
                )
        bbox_t, labels_t, mask_t = self.pad(bbox_t, labels_t, mask_t, max_elem=max_elem)
        vendor_labels = torch.zeros_like(labels_t)
        vendor_labels[mask_t] = labels_t[mask_t] + 1
        class_one_hot = torch.nn.functional.one_hot(vendor_labels, num_classes=4).to(
            dtype=bbox_t.dtype
        )
        layout = torch.stack((class_one_hot, bbox_t), dim=2)
        return {"layout": layout, "bbox": bbox_t, "labels": labels_t, "mask": mask_t}

    def pad(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_elem: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad layout tensors to DS-GAN ``max_elem`` slots."""
        if bbox.shape[1] > max_elem:
            raise ValueError(f"DS-GAN supports at most {max_elem} elements")
        pad_count = max_elem - bbox.shape[1]
        if pad_count:
            bbox = torch.cat((bbox, torch.zeros(bbox.shape[0], pad_count, 4)), dim=1)
            labels = torch.cat(
                (labels, torch.zeros(labels.shape[0], pad_count, dtype=torch.long)),
                dim=1,
            )
            mask = torch.cat(
                (mask, torch.zeros(mask.shape[0], pad_count, dtype=torch.bool)),
                dim=1,
            )
        labels = labels.clone()
        labels[~mask] = 0
        return bbox, labels, mask

    def _resolve_saliency(
        self,
        batch_size: int,
        *,
        saliency: ImageInput | list[ImageInput] | torch.Tensor | None,
        saliency_pfpnet: ImageInput | list[ImageInput] | torch.Tensor | None,
        saliency_basnet: ImageInput | list[ImageInput] | torch.Tensor | None,
    ) -> list[object | None]:
        if saliency is not None:
            rows = _ensure_batch(saliency)
            if len(rows) != batch_size:
                raise ValueError("saliency batch size must match images")
            return rows
        if saliency_pfpnet is None and saliency_basnet is None:
            return [None] * batch_size
        first = (
            _ensure_batch(saliency_pfpnet)
            if saliency_pfpnet is not None
            else [None] * batch_size
        )
        second = (
            _ensure_batch(saliency_basnet)
            if saliency_basnet is not None
            else [None] * batch_size
        )
        if len(first) != batch_size or len(second) != batch_size:
            raise ValueError("saliency batch size must match images")
        merged: list[torch.Tensor] = []
        for left, right in zip(first, second, strict=True):
            if left is None:
                merged.append(_to_l_tensor(right, self.image_size))
            elif right is None:
                merged.append(_to_l_tensor(left, self.image_size))
            else:
                merged.append(
                    torch.maximum(
                        _to_l_tensor(left, self.image_size),
                        _to_l_tensor(right, self.image_size),
                    )
                )
        return cast(list[object | None], merged)


def processor_for_dataset(dataset_name: DatasetName | str) -> DSGANProcessor:
    """Create a DS-GAN processor for a supported dataset."""
    return DSGANProcessor(dataset_name=dataset_name)


def _ensure_batch(value: object) -> list[object]:
    if isinstance(value, torch.Tensor):
        if value.ndim in {2, 3}:
            return [value]
        if value.ndim == 4:
            return [row for row in value]
    if isinstance(value, list):
        return list(value)
    return [value]


def _to_rgb_tensor(image: object, image_size: tuple[int, int]) -> torch.Tensor:
    tensor = _to_tensor(image, image_size=image_size, mode="RGB")
    if tensor.shape[0] != 3:
        raise ValueError("RGB image must have three channels")
    return tensor


def _to_l_tensor(image: object | None, image_size: tuple[int, int]) -> torch.Tensor:
    if image is None:
        return torch.zeros(1, *image_size, dtype=torch.float32)
    tensor = _to_tensor(image, image_size=image_size, mode="L")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor[:1]


def _to_tensor(
    image: object,
    *,
    image_size: tuple[int, int],
    mode: Literal["RGB", "L"],
) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] not in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(np.asarray(image))
    pil = pil.convert(mode).resize((image_size[1], image_size[0]))
    array = np.asarray(pil, dtype=np.float32) / 255.0
    if mode == "L":
        return torch.from_numpy(array).unsqueeze(0)
    return torch.from_numpy(array).permute(2, 0, 1)
