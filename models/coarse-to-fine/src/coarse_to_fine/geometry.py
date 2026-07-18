"""Geometry helpers for Coarse-to-Fine discretization and hierarchy math."""

from __future__ import annotations

from typing import cast

import torch

from laygen.common.bbox import (
    BoxFormat,
    clamp_boxes,
    ltwh_to_xywh,
    ltrb_to_xywh,
    normalize_box_format,
    xywh_to_ltrb,
    xywh_to_ltwh,
)


def public_to_ltwh(
    bbox: torch.Tensor,
    *,
    box_format: BoxFormat | str = BoxFormat.xywh,
) -> torch.Tensor:
    """Convert normalized public boxes to normalized left-top ``ltwh``.

    Args:
        bbox: Box tensor with the selected input format.
        box_format: Input box format.

    Returns:
        Normalized ``ltwh`` tensor.

    Raises:
        ValueError: If the box format is unsupported.

    Examples:
        >>> import torch
        >>> public_to_ltwh(torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])).shape
        torch.Size([1, 1, 4])
    """
    fmt = normalize_box_format(box_format)
    boxes = bbox.to(dtype=torch.float32)
    if fmt is BoxFormat.xywh:
        return clamp_boxes(xywh_to_ltwh(boxes))
    if fmt is BoxFormat.ltwh:
        return clamp_boxes(boxes)
    if fmt is BoxFormat.ltrb:
        return clamp_boxes(xywh_to_ltwh(ltrb_to_xywh(boxes)))
    raise ValueError(f"Unsupported box_format: {box_format}")


def ltwh_to_public_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``ltwh`` boxes to public normalized center ``xywh``."""
    return clamp_boxes(ltwh_to_xywh(bbox.to(dtype=torch.float32)))


def discretize_ltwh(
    bbox: torch.Tensor, *, num_x_grid: int, num_y_grid: int
) -> torch.LongTensor:
    """Discretize normalized ``ltwh`` with the vendor ``floor(coord*(grid-1))`` rule.

    Args:
        bbox: Normalized ``ltwh`` tensor.
        num_x_grid: Number of x/width bins.
        num_y_grid: Number of y/height bins.

    Returns:
        Integer ``ltwh`` bin ids.

    Examples:
        >>> import torch
        >>> discretize_ltwh(torch.tensor([[1.0, 0.5, 0.0, 0.5]]), num_x_grid=128, num_y_grid=128)
        tensor([[127,  63,   0,  63]])
    """
    grids = torch.tensor(
        [num_x_grid - 1, num_y_grid - 1, num_x_grid - 1, num_y_grid - 1],
        device=bbox.device,
        dtype=bbox.dtype,
    )
    return cast(torch.LongTensor, torch.floor(bbox.clamp(0.0, 1.0) * grids).long())


def continuize_ltwh(
    ids: torch.Tensor, *, num_x_grid: int, num_y_grid: int
) -> torch.FloatTensor:
    """Convert discrete ``ltwh`` ids back to normalized coordinates.

    Args:
        ids: Integer ``ltwh`` ids.
        num_x_grid: Number of x/width bins.
        num_y_grid: Number of y/height bins.

    Returns:
        Normalized ``ltwh`` tensor.

    Examples:
        >>> import torch
        >>> continuize_ltwh(torch.tensor([[127, 63, 0, 63]]), num_x_grid=128, num_y_grid=128).shape
        torch.Size([1, 4])
    """
    values = ids.to(dtype=torch.float32)
    grids = torch.tensor(
        [num_x_grid - 1, num_y_grid - 1, num_x_grid - 1, num_y_grid - 1],
        device=values.device,
        dtype=values.dtype,
    )
    return cast(torch.FloatTensor, (values / grids).clamp(0.0, 1.0))


def relative_ltwh_to_absolute_ltwh(
    relative_bbox: torch.Tensor, group_bbox: torch.Tensor
) -> torch.Tensor:
    """Convert group-relative ``ltwh`` boxes to absolute normalized ``ltwh``.

    Args:
        relative_bbox: Relative left, top, width, height inside the group.
        group_bbox: Absolute group ``ltwh`` boxes broadcastable to
            ``relative_bbox``.

    Returns:
        Absolute normalized ``ltwh`` boxes.

    Examples:
        >>> import torch
        >>> relative_ltwh_to_absolute_ltwh(
        ...     torch.tensor([[0.5, 0.0, 0.5, 1.0]]),
        ...     torch.tensor([[0.0, 0.0, 0.4, 0.2]]),
        ... )
        tensor([[0.2000, 0.0000, 0.2000, 0.2000]])
    """
    left, top, width, height = relative_bbox.unbind(dim=-1)
    group_left, group_top, group_width, group_height = group_bbox.unbind(dim=-1)
    return clamp_boxes(
        torch.stack(
            (
                group_left + left * group_width,
                group_top + top * group_height,
                width * group_width,
                height * group_height,
            ),
            dim=-1,
        )
    )


def ltwh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    """Convert ``ltwh`` boxes to ``ltrb`` boxes."""
    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left, top, left + width, top + height), dim=-1)


def ltrb_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert ``ltrb`` boxes to ``ltwh`` boxes."""
    left, top, right, bottom = bbox.unbind(dim=-1)
    return torch.stack((left, top, right - left, bottom - top), dim=-1)


def public_to_ltrb(
    bbox: torch.Tensor, *, box_format: BoxFormat | str = BoxFormat.xywh
) -> torch.Tensor:
    """Convert normalized public boxes to normalized ``ltrb``."""
    fmt = normalize_box_format(box_format)
    if fmt is BoxFormat.xywh:
        return clamp_boxes(xywh_to_ltrb(bbox.to(dtype=torch.float32)))
    if fmt is BoxFormat.ltwh:
        return clamp_boxes(ltwh_to_ltrb(bbox.to(dtype=torch.float32)))
    if fmt is BoxFormat.ltrb:
        return clamp_boxes(bbox.to(dtype=torch.float32))
    raise ValueError(f"Unsupported box_format: {box_format}")
