"""Bounding-box conversion helpers for normalized layout tensors."""

from typing import Literal

import torch


def xywh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to ``ltrb`` boxes."""
    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def ltrb_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert ``ltrb`` boxes to center ``xywh`` boxes."""
    left, top, right, bottom = bbox.unbind(dim=-1)
    return torch.stack(
        (
            (left + right) / 2,
            (top + bottom) / 2,
            right - left,
            bottom - top,
        ),
        dim=-1,
    )


def ltwh_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert top-left ``xywh`` boxes to center ``xywh`` boxes."""
    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to top-left ``xywh`` boxes."""
    x, y, width, height = bbox.unbind(dim=-1)
    return torch.stack((x - width / 2, y - height / 2, width, height), dim=-1)


def clamp_boxes(bbox: torch.Tensor) -> torch.Tensor:
    """Clamp normalized boxes into the public ``[0, 1]`` range."""
    return bbox.clamp(0.0, 1.0)


def normalize_boxes(
    bbox: torch.Tensor,
    *,
    canvas_size: tuple[int, int],
    box_format: Literal["xywh", "ltwh", "ltrb"],
) -> torch.Tensor:
    """Normalize pixel boxes to public center ``xywh`` in ``[0, 1]``."""
    width, height = canvas_size
    scale = bbox.new_tensor((width, height, width, height))
    normalized = bbox / scale
    if box_format == "xywh":
        return clamp_boxes(normalized)
    if box_format == "ltwh":
        return clamp_boxes(ltwh_to_xywh(normalized))
    return clamp_boxes(ltrb_to_xywh(normalized))


def denormalize_boxes(
    bbox: torch.Tensor,
    *,
    canvas_size: tuple[int, int],
    box_format: Literal["xywh", "ltwh", "ltrb"],
) -> torch.Tensor:
    """Convert public normalized center ``xywh`` boxes to pixel boxes."""
    width, height = canvas_size
    scale = bbox.new_tensor((width, height, width, height))
    if box_format == "xywh":
        converted = bbox
    elif box_format == "ltwh":
        converted = xywh_to_ltwh(bbox)
    else:
        converted = xywh_to_ltrb(bbox)
    return converted * scale
