from __future__ import annotations

from typing import Literal

import torch


BoxFormat = Literal["xywh", "ltwh", "ltrb"]


def xywh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def ltrb_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = bbox.unbind(dim=-1)
    return torch.stack(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dim=-1,
    )


def ltwh_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, w, h), dim=-1)


def clamp_boxes(bbox: torch.Tensor) -> torch.Tensor:
    return bbox.clamp(0.0, 1.0)


def _canvas_tensor(
    canvas_size: tuple[int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    width, height = canvas_size
    return torch.tensor((width, height, width, height), device=device, dtype=dtype)


def normalize_boxes(
    bbox: torch.Tensor,
    *,
    canvas_size: tuple[int, int],
    box_format: BoxFormat,
) -> torch.Tensor:
    bbox = bbox.to(dtype=torch.float32)
    scale = _canvas_tensor(canvas_size, bbox.device, bbox.dtype)
    normalized = bbox / scale
    if box_format == "xywh":
        return clamp_boxes(normalized)
    if box_format == "ltwh":
        return clamp_boxes(ltwh_to_xywh(normalized))
    if box_format == "ltrb":
        return clamp_boxes(ltrb_to_xywh(normalized))
    raise ValueError(f"Unsupported box_format: {box_format}")


def denormalize_boxes(
    bbox: torch.Tensor,
    *,
    canvas_size: tuple[int, int],
    box_format: BoxFormat,
) -> torch.Tensor:
    if box_format == "xywh":
        out = bbox
    elif box_format == "ltwh":
        out = xywh_to_ltwh(bbox)
    elif box_format == "ltrb":
        out = xywh_to_ltrb(bbox)
    else:
        raise ValueError(f"Unsupported box_format: {box_format}")
    scale = _canvas_tensor(canvas_size, out.device, out.dtype)
    return out * scale


def linear_discretize(values: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    delta = 1.0 / num_bins
    values = values.clamp(0.0, 1.0 - delta)
    return (values * num_bins).round().long().clamp(0, num_bins - 1)


def linear_continuize(ids: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    return ids.float().clamp(0, num_bins - 1) / num_bins
