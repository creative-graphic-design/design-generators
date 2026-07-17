"""Bounding box conversion helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

import torch


class BoxFormat(StrEnum):
    """Supported public bounding box coordinate formats.

    Examples:
        >>> BoxFormat.XYWH.value
        'xywh'
    """

    XYWH = "xywh"
    LTWH = "ltwh"
    LTRB = "ltrb"


def normalize_box_format(box_format: BoxFormat | str) -> BoxFormat:
    """Return a box format enum from a public string value.

    Args:
        box_format: Existing enum value or public string.

    Returns:
        The normalized `BoxFormat`.

    Raises:
        ValueError: If `box_format` is unsupported.

    Examples:
        >>> normalize_box_format("ltwh")
        <BoxFormat.LTWH: 'ltwh'>
    """
    try:
        return BoxFormat(box_format)
    except ValueError as exc:
        raise ValueError(f"Unsupported box format: {box_format}") from exc


def ltwh_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert top-left ``xywh`` boxes to center ``xywh`` boxes.

    Args:
        bbox: Tensor whose final dimension is `(left, top, width, height)`.

    Returns:
        Tensor whose final dimension is `(center_x, center_y, width, height)`.

    Examples:
        >>> ltwh_to_xywh(torch.tensor([[0.0, 0.0, 2.0, 4.0]])).tolist()
        [[1.0, 2.0, 2.0, 4.0]]
    """
    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to top-left ``xywh`` boxes.

    Args:
        bbox: Tensor whose final dimension is `(center_x, center_y, width, height)`.

    Returns:
        Tensor whose final dimension is `(left, top, width, height)`.

    Examples:
        >>> xywh_to_ltwh(torch.tensor([[1.0, 2.0, 2.0, 4.0]])).tolist()
        [[0.0, 0.0, 2.0, 4.0]]
    """
    center_x, center_y, width, height = bbox.unbind(dim=-1)
    return torch.stack(
        (center_x - width / 2, center_y - height / 2, width, height), dim=-1
    )


def ltrb_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert ``left, top, right, bottom`` boxes to center ``xywh`` boxes.

    Args:
        bbox: Tensor whose final dimension is `(left, top, right, bottom)`.

    Returns:
        Tensor whose final dimension is `(center_x, center_y, width, height)`.

    Examples:
        >>> ltrb_to_xywh(torch.tensor([[0.0, 0.0, 2.0, 4.0]])).tolist()
        [[1.0, 2.0, 2.0, 4.0]]
    """
    left, top, right, bottom = bbox.unbind(dim=-1)
    width = right - left
    height = bottom - top
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center ``xywh`` boxes to ``left, top, right, bottom`` boxes.

    Args:
        bbox: Tensor whose final dimension is `(center_x, center_y, width, height)`.

    Returns:
        Tensor whose final dimension is `(left, top, right, bottom)`.

    Examples:
        >>> xywh_to_ltrb(torch.tensor([[1.0, 2.0, 2.0, 4.0]])).tolist()
        [[0.0, 0.0, 2.0, 4.0]]
    """
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
    box_format: BoxFormat | str,
) -> torch.Tensor:
    """Normalize pixel boxes and return center ``xywh`` in ``[0, 1]``.

    Args:
        bbox: Pixel-coordinate tensor.
        canvas_size: `(width, height)` canvas size in pixels.
        box_format: Format used by the input tensor.

    Returns:
        Normalized center `xywh` boxes.

    Raises:
        ValueError: If `box_format` is unsupported.

    Examples:
        >>> normalize_boxes(
        ...     torch.tensor([[0.0, 0.0, 10.0, 20.0]]),
        ...     canvas_size=(100, 100),
        ...     box_format="ltwh",
        ... ).tolist()
        [[0.05000000074505806, 0.10000000149011612, 0.10000000149011612, 0.20000000298023224]]
    """
    width, height = canvas_size
    scale = bbox.new_tensor((width, height, width, height))
    normalized = bbox.float() / scale
    normalized_format = normalize_box_format(box_format)
    if normalized_format is BoxFormat.XYWH:
        return normalized
    if normalized_format is BoxFormat.LTWH:
        return ltwh_to_xywh(normalized)
    if normalized_format is BoxFormat.LTRB:
        return ltrb_to_xywh(normalized)
    assert_never(normalized_format)


def clamp_boxes(bbox: torch.Tensor) -> torch.Tensor:
    """Clamp normalized boxes to the public ``[0, 1]`` range.

    Args:
        bbox: Normalized bounding boxes.

    Returns:
        Boxes with all values clamped into `[0, 1]`.

    Examples:
        >>> clamp_boxes(torch.tensor([[-1.0, 0.5, 2.0, 0.25]])).tolist()
        [[0.0, 0.5, 1.0, 0.25]]
    """
    return bbox.clamp(0.0, 1.0)
