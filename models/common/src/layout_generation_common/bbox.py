"""Bounding box conversion helpers."""

from __future__ import annotations

from typing import Literal

import torch


def ltwh_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert top-left ``xywh`` boxes to center ``xywh`` boxes."""
    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to top-left ``xywh`` boxes."""
    center_x, center_y, width, height = bbox.unbind(dim=-1)
    return torch.stack(
        (center_x - width / 2, center_y - height / 2, width, height), dim=-1
    )


def ltrb_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert ``left, top, right, bottom`` boxes to center ``xywh`` boxes."""
    left, top, right, bottom = bbox.unbind(dim=-1)
    width = right - left
    height = bottom - top
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to ``left, top, right, bottom`` boxes."""
    center_x, center_y, width, height = bbox.unbind(dim=-1)
    half_width = width / 2
    half_height = height / 2
    return torch.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        dim=-1,
    )


def normalize_boxes(
    bbox: torch.Tensor,
    *,
    canvas_size: tuple[int, int],
    box_format: Literal["xywh", "ltwh", "ltrb"],
) -> torch.Tensor:
    """Normalize pixel boxes and return center ``xywh`` in ``[0, 1]``."""
    width, height = canvas_size
    scale = bbox.new_tensor((width, height, width, height))
    normalized = bbox.float() / scale
    if box_format == "xywh":
        return normalized
    if box_format == "ltwh":
        return ltwh_to_xywh(normalized)
    if box_format == "ltrb":
        return ltrb_to_xywh(normalized)
    raise ValueError(f"Unsupported box format: {box_format}")


def clamp_boxes(bbox: torch.Tensor) -> torch.Tensor:
    """Clamp normalized boxes to the public ``[0, 1]`` range."""
    return bbox.clamp(0.0, 1.0)
