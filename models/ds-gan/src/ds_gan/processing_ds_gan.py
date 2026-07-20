"""Processor for DS-GAN content-image inputs and layout decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat, normalize_boxes, prepare_layout_tensors
from laygen.modeling_outputs import LayoutGenerationOutput
from posgen.common.labels import DatasetName, normalize_dataset_name

PKU_DATASET_LABEL2ID: dict[str, int] = {"text": 0, "logo": 1, "underlay": 2}
PKU_VENDOR_LABEL2ID: dict[str, int] = {
    "no_object": 0,
    "text": 1,
    "logo": 2,
    "underlay": 3,
}
_BILINEAR: Final = Image.Resampling.BILINEAR


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
        images: object,
        *,
        saliency: object | None = None,
        saliency_pfpnet: object | None = None,
        saliency_basnet: object | None = None,
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
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            clamp_converted_normalized=True,
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
        saliency: object | None,
        saliency_pfpnet: object | None,
        saliency_basnet: object | None,
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
                merged.append(_merge_saliency_native(left, right, self.image_size))
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
            antialias=False,
        ).squeeze(0)
    if isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(np.asarray(image))
    pil = pil.convert(mode).resize((image_size[1], image_size[0]), _BILINEAR)
    array = np.asarray(pil, dtype=np.float32) / 255.0
    if mode == "L":
        return torch.from_numpy(array).unsqueeze(0)
    return torch.from_numpy(array).permute(2, 0, 1)


def _merge_saliency_native(
    left: object,
    right: object,
    image_size: tuple[int, int],
) -> torch.Tensor:
    left_pil = _to_pil(left).convert("L")
    right_pil = _to_pil(right).convert("L")
    if left_pil.size != right_pil.size:
        right_pil = right_pil.resize(left_pil.size, _BILINEAR)
    merged = Image.fromarray(np.maximum(np.asarray(left_pil), np.asarray(right_pil)))
    return _to_l_tensor(merged, image_size)


def _to_pil(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.max() <= 1.0:
            array = array * 255.0
        return Image.fromarray(array.astype(np.uint8).squeeze())
    return Image.fromarray(np.asarray(image))


def annotations_from_pku_example(
    example: dict[str, object],
    *,
    max_elem: int = 32,
) -> dict[str, torch.Tensor | tuple[int, int]]:
    """Convert a PKU PosterLayout dataset row into public layout tensors.

    The adapter filters ``INVALID`` annotations, converts pixel ``ltrb`` boxes
    to normalized center ``xywh``, derives canvas size from the image columns,
    and applies the vendor ``designSeq.reorder`` ordering policy.
    """
    annotations = cast(dict[str, object], example.get("annotations", example))
    raw_labels = cast(list[object], annotations["cls_elem"])
    raw_boxes = cast(list[object], annotations["box_elem"])
    canvas_size = _canvas_size_from_example(example)
    vendor_labels: list[int] = []
    public_labels: list[int] = []
    boxes: list[list[float]] = []
    for raw_label, raw_box in zip(raw_labels, raw_boxes, strict=True):
        label = str(raw_label)
        if label == "INVALID":
            continue
        public_id = PKU_DATASET_LABEL2ID[label]
        vendor_labels.append(PKU_VENDOR_LABEL2ID[label])
        public_labels.append(public_id)
        boxes.append(_parse_box(raw_box))
    if boxes:
        box_t = torch.tensor(boxes, dtype=torch.float32)
        order = _designseq_reorder(vendor_labels, box_t, max_elem=max_elem)
        box_t = box_t[order]
        labels_t = torch.tensor([public_labels[i] for i in order], dtype=torch.long)
        bbox_t = normalize_boxes(
            box_t.unsqueeze(0), canvas_size=canvas_size, box_format="ltrb"
        ).squeeze(0)
    else:
        bbox_t = torch.zeros(0, 4, dtype=torch.float32)
        labels_t = torch.zeros(0, dtype=torch.long)
    mask_t = torch.ones(labels_t.shape, dtype=torch.bool)
    return {
        "bbox": bbox_t.unsqueeze(0),
        "labels": labels_t.unsqueeze(0),
        "mask": mask_t.unsqueeze(0),
        "canvas_size": canvas_size,
    }


def _parse_box(raw_box: object) -> list[float]:
    if isinstance(raw_box, str):
        import ast

        parsed = ast.literal_eval(raw_box)
    else:
        parsed = raw_box
    values = [float(v) for v in cast(list[int | float | str], parsed)]
    left, top, right, bottom = values
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    return [left, top, right, bottom]


def _canvas_size_from_example(example: dict[str, object]) -> tuple[int, int]:
    for key in ("image", "image_canvas", "canvas", "poster"):
        value = example.get(key)
        if isinstance(value, Image.Image):
            return value.size
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2:
                return int(value.shape[-1]), int(value.shape[-2])
        if value is not None:
            pil = _to_pil(value)
            return pil.size
    width = example.get("width")
    height = example.get("height")
    if width is not None and height is not None:
        width_value = cast(int | float | str, width)
        height_value = cast(int | float | str, height)
        return int(width_value), int(height_value)
    raise ValueError("PKU example must include image/canvas or width and height")


def _designseq_reorder(
    vendor_labels: list[int],
    boxes: torch.Tensor,
    *,
    max_elem: int,
) -> list[int]:
    try:
        from designSeq import reorder
    except ImportError:
        return _fallback_reorder(vendor_labels, boxes, max_elem=max_elem)
    return [int(i) for i in reorder(vendor_labels, boxes, "xyxy", max_elem)]


def _fallback_reorder(
    vendor_labels: list[int],
    boxes: torch.Tensor,
    *,
    max_elem: int,
) -> list[int]:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    indices = list(range(len(vendor_labels)))
    logos = [i for i in indices if vendor_labels[i] == 2]
    texts = sorted(
        [i for i in indices if vendor_labels[i] == 1],
        key=lambda i: float(areas[i]),
        reverse=True,
    )
    underlays = sorted(
        [i for i in indices if vendor_labels[i] == 3],
        key=lambda i: float(areas[i]),
    )
    return (logos + texts + underlays)[:max_elem]
