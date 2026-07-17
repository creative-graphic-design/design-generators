from __future__ import annotations

from typing import Literal

import torch


BoxFormat = Literal["xywh", "ltwh", "ltrb"]


def xywh_to_ltrb(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center-size boxes to corner boxes.

    Args:
        bbox: Tensor whose final dimension is `(x, y, width, height)`.

    Returns:
        Tensor with final dimension `(left, top, right, bottom)`.

    Raises:
        ValueError: This function does not raise directly; PyTorch raises if the
            final dimension cannot be unpacked into four values.

    Examples:
        >>> import torch
        >>> xywh_to_ltrb(torch.tensor([[0.5, 0.5, 0.2, 0.4]])).shape
        torch.Size([1, 4])
    """

    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def ltrb_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert corner boxes to center-size boxes.

    Args:
        bbox: Tensor whose final dimension is `(left, top, right, bottom)`.

    Returns:
        Tensor with final dimension `(x, y, width, height)`.

    Raises:
        ValueError: This function does not raise directly; PyTorch raises if the
            final dimension cannot be unpacked into four values.

    Examples:
        >>> import torch
        >>> ltrb_to_xywh(torch.tensor([[0.4, 0.3, 0.6, 0.7]])).shape
        torch.Size([1, 4])
    """

    left, top, right, bottom = bbox.unbind(dim=-1)
    return torch.stack(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dim=-1,
    )


def ltwh_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert left-top-size boxes to center-size boxes.

    Args:
        bbox: Tensor whose final dimension is `(left, top, width, height)`.

    Returns:
        Tensor with final dimension `(x, y, width, height)`.

    Raises:
        ValueError: This function does not raise directly; PyTorch raises if the
            final dimension cannot be unpacked into four values.

    Examples:
        >>> import torch
        >>> ltwh_to_xywh(torch.tensor([[0.4, 0.3, 0.2, 0.4]])).shape
        torch.Size([1, 4])
    """

    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert center-size boxes to left-top-size boxes.

    Args:
        bbox: Tensor whose final dimension is `(x, y, width, height)`.

    Returns:
        Tensor with final dimension `(left, top, width, height)`.

    Raises:
        ValueError: This function does not raise directly; PyTorch raises if the
            final dimension cannot be unpacked into four values.

    Examples:
        >>> import torch
        >>> xywh_to_ltwh(torch.tensor([[0.5, 0.5, 0.2, 0.4]])).shape
        torch.Size([1, 4])
    """

    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, w, h), dim=-1)


def clamp_boxes(bbox: torch.Tensor) -> torch.Tensor:
    """Clamp normalized box coordinates into `[0, 1]`.

    Args:
        bbox: Tensor of box coordinates.

    Returns:
        Tensor with all values clipped to the valid normalized range.

    Raises:
        ValueError: This function does not raise.

    Examples:
        >>> import torch
        >>> clamp_boxes(torch.tensor([-0.5, 0.5, 1.5])).tolist()
        [0.0, 0.5, 1.0]
    """

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
    """Normalize pixel boxes to center-size coordinates in `[0, 1]`.

    Args:
        bbox: Pixel-space boxes.
        canvas_size: `(width, height)` canvas used to scale coordinates.
        box_format: Input coordinate format.

    Returns:
        Normalized `xywh` boxes with values clamped to `[0, 1]`.

    Raises:
        ValueError: If `box_format` is unsupported.

    Examples:
        >>> import torch
        >>> normalize_boxes(
        ...     torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
        ...     canvas_size=(100, 100),
        ...     box_format="ltrb",
        ... ).shape
        torch.Size([1, 4])
    """

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
    """Convert normalized center-size boxes back to pixel coordinates.

    Args:
        bbox: Normalized `xywh` boxes.
        canvas_size: `(width, height)` canvas used to scale coordinates.
        box_format: Output coordinate format.

    Returns:
        Pixel-space boxes in the requested format.

    Raises:
        ValueError: If `box_format` is unsupported.

    Examples:
        >>> import torch
        >>> denormalize_boxes(
        ...     torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
        ...     canvas_size=(100, 100),
        ...     box_format="ltrb",
        ... ).shape
        torch.Size([1, 4])
    """

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
    """Map normalized continuous values to uniform integer bins.

    Args:
        values: Tensor of normalized values.
        num_bins: Number of bins covering `[0, 1]`.

    Returns:
        Integer tensor with values in `[0, num_bins - 1]`.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> linear_discretize(torch.tensor([0.0, 0.25, 0.99]), num_bins=4).tolist()
        [0, 1, 3]
    """

    delta = 1.0 / num_bins
    values = values.clamp(0.0, 1.0 - delta)
    return (values * num_bins).round().long().clamp(0, num_bins - 1)


def linear_continuize(ids: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    """Map uniform integer bins back to normalized lower-edge values.

    Args:
        ids: Integer bin tensor.
        num_bins: Number of bins covering `[0, 1]`.

    Returns:
        Floating tensor with values clipped to the representable bin range.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> linear_continuize(torch.tensor([0, 1, 3]), num_bins=4).tolist()
        [0.0, 0.25, 0.75]
    """

    return ids.float().clamp(0, num_bins - 1) / num_bins
