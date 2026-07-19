"""Bounding-box conversion and quantization helpers for layout packages."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, TypeAlias
from typing import assert_never

if TYPE_CHECKING:
    import torch
    from laygen.common.torch_typing import TorchBoxes, TorchTensor
else:
    try:
        import torch
        from laygen.common.torch_typing import TorchBoxes, TorchTensor
    except ImportError:
        TorchBoxes: TypeAlias = object
        TorchTensor: TypeAlias = object


class BoxFormat(StrEnum):
    """Supported bounding-box coordinate formats."""

    xywh = auto()
    ltwh = auto()
    ltrb = auto()


def normalize_box_format(box_format: BoxFormat | str) -> BoxFormat:
    """Convert a public box-format value to ``BoxFormat``.

    Args:
        box_format: Box format enum or its string value.

    Returns:
        Normalized ``BoxFormat`` enum.

    Raises:
        ValueError: If ``box_format`` is not supported.
    """
    if isinstance(box_format, BoxFormat):
        return box_format
    try:
        return BoxFormat(box_format)
    except ValueError as exc:
        raise ValueError(f"Unsupported box_format: {box_format}") from exc


def xywh_to_ltrb(bbox: TorchBoxes) -> TorchBoxes:
    """Convert normalized center ``xywh`` boxes to ``ltrb`` boxes.

    Args:
        bbox: Tensor with the last dimension ordered as center x, center y,
            width, and height.

    Returns:
        Tensor with the same leading shape and last dimension ordered as left,
        top, right, and bottom.

    Examples:
        >>> import torch
        >>> xywh_to_ltrb(torch.tensor([[0.5, 0.5, 0.2, 0.4]])).shape
        torch.Size([1, 4])
    """
    import torch

    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def ltrb_to_xywh(bbox: TorchBoxes) -> TorchBoxes:
    """Convert ``ltrb`` boxes to normalized center ``xywh`` boxes."""
    import torch

    left, top, right, bottom = bbox.unbind(dim=-1)
    return torch.stack(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dim=-1,
    )


def ltwh_to_xywh(bbox: TorchBoxes) -> TorchBoxes:
    """Convert left-top-width-height boxes to center ``xywh`` boxes."""
    import torch

    left, top, width, height = bbox.unbind(dim=-1)
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def xywh_to_ltwh(bbox: TorchBoxes) -> TorchBoxes:
    """Convert center ``xywh`` boxes to left-top-width-height boxes."""
    import torch

    x, y, w, h = bbox.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, w, h), dim=-1)


def clamp_boxes(bbox: TorchBoxes) -> TorchBoxes:
    """Clamp normalized box coordinates into the inclusive ``[0, 1]`` range."""
    return bbox.clamp(0.0, 1.0)


def _canvas_tensor(
    canvas_size: tuple[int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    import torch

    width, height = canvas_size
    return torch.tensor((width, height, width, height), device=device, dtype=dtype)


def normalize_boxes(
    bbox: TorchBoxes,
    *,
    canvas_size: tuple[int, int],
    box_format: BoxFormat | str,
) -> TorchBoxes:
    """Normalize pixel boxes to center ``xywh`` coordinates.

    Args:
        bbox: Tensor containing pixel-space boxes.
        canvas_size: Canvas size as ``(width, height)``.
        box_format: Input box format.

    Returns:
        Tensor containing normalized center ``xywh`` boxes.

    Raises:
        ValueError: If ``box_format`` is unsupported.

    Examples:
        >>> import torch
        >>> normalize_boxes(
        ...     torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]),
        ...     canvas_size=(100, 100),
        ...     box_format="ltrb",
        ... ).shape
        torch.Size([1, 1, 4])
    """
    import torch

    bbox = bbox.to(dtype=torch.float32)
    scale = _canvas_tensor(canvas_size, bbox.device, bbox.dtype)
    normalized = bbox / scale
    fmt = normalize_box_format(box_format)
    if fmt is BoxFormat.xywh:
        return clamp_boxes(normalized)
    if fmt is BoxFormat.ltwh:
        return clamp_boxes(ltwh_to_xywh(normalized))
    if fmt is BoxFormat.ltrb:
        return clamp_boxes(ltrb_to_xywh(normalized))
    assert_never(fmt)


def denormalize_boxes(
    bbox: TorchBoxes,
    *,
    canvas_size: tuple[int, int],
    box_format: BoxFormat | str,
) -> TorchBoxes:
    """Convert normalized center ``xywh`` boxes to pixel-space boxes.

    Args:
        bbox: Normalized center ``xywh`` tensor.
        canvas_size: Canvas size as ``(width, height)``.
        box_format: Requested output box format.

    Returns:
        Tensor in the requested pixel-space format.

    Raises:
        ValueError: If ``box_format`` is unsupported.
    """
    fmt = normalize_box_format(box_format)
    if fmt is BoxFormat.xywh:
        out = bbox
    elif fmt is BoxFormat.ltwh:
        out = xywh_to_ltwh(bbox)
    elif fmt is BoxFormat.ltrb:
        out = xywh_to_ltrb(bbox)
    else:
        assert_never(fmt)
    scale = _canvas_tensor(canvas_size, out.device, out.dtype)
    return out * scale


def linear_discretize(values: TorchTensor, *, num_bins: int) -> TorchTensor:
    """Map normalized continuous values to evenly spaced integer bins."""
    delta = 1.0 / num_bins
    values = values.clamp(0.0, 1.0 - delta)
    return (values * num_bins).round().long().clamp(0, num_bins - 1)


def linear_continuize(ids: TorchTensor, *, num_bins: int) -> TorchTensor:
    """Map evenly spaced integer bins back to normalized continuous values."""
    return ids.float().clamp(0, num_bins - 1) / num_bins
