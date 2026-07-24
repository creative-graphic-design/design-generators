"""Dataset row helpers for CGL-style RADM annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import SupportsFloat, SupportsInt, cast

from jaxtyping import Float, Int
from laygen.common.bbox import ltwh_to_xywh
import torch


@dataclass(frozen=True)
class RADMAnnotation:
    """Normalized RADM annotation row."""

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


def normalize_coco_annotations(
    annotations: Sequence[Mapping[str, object]],
    *,
    canvas_size: tuple[int, int],
) -> tuple[
    Float[torch.Tensor, "batch elements 4"],
    Int[torch.Tensor, "batch elements"],
]:
    """Normalize COCO-style CGL annotations.

    Args:
        annotations: Rows with ``category_id`` and ``bbox`` in pixel ``ltwh``.
        canvas_size: Canvas size as ``(width, height)``.

    Returns:
        Normalized center ``xywh`` boxes and zero-based labels.
    """
    width, height = canvas_size
    boxes = []
    labels = []
    for row in annotations:
        raw_bbox = row["bbox"]
        if not isinstance(raw_bbox, Sequence):
            raise TypeError("annotation bbox must be a sequence")
        numeric_bbox = cast(Sequence[SupportsFloat], raw_bbox)
        left, top, box_width, box_height = (float(value) for value in numeric_bbox)
        boxes.append(
            (left / width, top / height, box_width / width, box_height / height)
        )
        labels.append(int(cast(SupportsInt, row["category_id"])) - 1)
    bbox = ltwh_to_xywh(torch.tensor(boxes, dtype=torch.float32)).unsqueeze(0)
    return bbox, torch.tensor(labels, dtype=torch.long).unsqueeze(0)
