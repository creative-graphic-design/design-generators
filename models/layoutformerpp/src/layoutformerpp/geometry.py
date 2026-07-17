"""Geometry helpers for LayoutFormer++ discrete ltwh boxes."""

from __future__ import annotations

import torch

from laygen.common.bbox import (
    BoxFormat,
    ltwh_to_xywh,
    normalize_box_format,
    xywh_to_ltwh,
)


def discretize_ltwh(
    bbox: torch.Tensor, *, x_grid: int = 128, y_grid: int = 128
) -> torch.Tensor:
    """Convert normalized ltwh values to LayoutFormer++ integer bins."""
    grids = (
        torch.tensor(
            (x_grid, y_grid, x_grid, y_grid), dtype=bbox.dtype, device=bbox.device
        )
        - 1
    )
    return torch.floor(bbox.clamp(0.0, 1.0) * grids).long().clamp_min(0)


def continuize_ltwh(
    ids: torch.Tensor, *, x_grid: int = 128, y_grid: int = 128
) -> torch.Tensor:
    """Convert LayoutFormer++ integer bins back to normalized ltwh."""
    grids = (
        torch.tensor(
            (x_grid, y_grid, x_grid, y_grid), dtype=torch.float32, device=ids.device
        )
        - 1
    )
    return ids.float().clamp_min(0) / grids


def public_to_discrete_ltwh(
    bbox: torch.Tensor,
    *,
    box_format: BoxFormat | str = BoxFormat.xywh,
    x_grid: int = 128,
    y_grid: int = 128,
) -> torch.Tensor:
    """Convert public normalized boxes to internal discrete ltwh tokens."""
    fmt = normalize_box_format(box_format)
    ltwh = xywh_to_ltwh(bbox.float()) if fmt is BoxFormat.xywh else bbox.float()
    if fmt is not BoxFormat.xywh and fmt is not BoxFormat.ltwh:
        raise ValueError(f"Unsupported box_format: {box_format}")
    return discretize_ltwh(ltwh, x_grid=x_grid, y_grid=y_grid)


def discrete_ltwh_to_public(
    ids: torch.Tensor,
    *,
    box_format: BoxFormat | str = BoxFormat.xywh,
    x_grid: int = 128,
    y_grid: int = 128,
) -> torch.Tensor:
    """Convert internal discrete ltwh tokens to public normalized boxes."""
    ltwh = continuize_ltwh(ids, x_grid=x_grid, y_grid=y_grid)
    fmt = normalize_box_format(box_format)
    if fmt is BoxFormat.xywh:
        return ltwh_to_xywh(ltwh).clamp(0.0, 1.0)
    if fmt is BoxFormat.ltwh:
        return ltwh.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported box_format: {box_format}")
