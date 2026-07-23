"""Pure tensor LayoutDETR postprocessing helpers."""

from __future__ import annotations

from enum import StrEnum, auto

import torch
from jaxtyping import Bool, Float

from laygen.common.bbox import clamp_boxes, xywh_to_ltrb


class PostprocessingMode(StrEnum):
    """Supported LayoutDETR postprocessing modes."""

    none = auto()
    horizontal_center_aligned = auto()
    horizontal_left_aligned = auto()


def normalize_postprocessing_mode(
    mode: PostprocessingMode | str,
) -> PostprocessingMode:
    """Normalize a public postprocessing mode value."""
    if isinstance(mode, PostprocessingMode):
        return mode
    try:
        return PostprocessingMode(mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported out_postprocessing: {mode}") from exc


def jitter_boxes(
    bbox: Float[torch.Tensor, "batch elements 4"],
    *,
    strength: float,
    generator: torch.Generator | None,
) -> Float[torch.Tensor, "batch elements 4"]:
    """Apply vendor-style multiplicative jitter to generated boxes."""
    if strength == 0.0:
        return bbox
    if strength < 0.0 or strength >= 1.0:
        raise ValueError("strength must be in [0, 1)")
    low = torch.log(bbox.new_tensor(1.0 - strength))
    high = torch.log(bbox.new_tensor(1.0 + strength))
    noise = torch.rand(
        bbox.shape,
        generator=generator,
        device=bbox.device,
        dtype=bbox.dtype,
    )
    return bbox * torch.exp(low + (high - low) * noise)


def horizontal_center_aligned(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 4"]:
    """Align valid boxes to the mean center-x coordinate."""
    out = bbox.clone()
    for batch in range(out.shape[0]):
        valid = mask[batch]
        if valid.any():
            out[batch, valid, 0] = out[batch, valid, 0].mean()
    return out


def horizontal_left_aligned(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 4"]:
    """Align valid boxes to the mean left edge."""
    out = bbox.clone()
    left_edges = xywh_to_ltrb(out)[..., 0]
    for batch in range(out.shape[0]):
        valid = mask[batch]
        if valid.any():
            target_left = left_edges[batch, valid].mean()
            out[batch, valid, 0] -= left_edges[batch, valid] - target_left
    return out


def de_overlap(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 4"]:
    """Reduce vertical overlaps with the deterministic vendor arithmetic."""
    out = bbox.clone()
    for batch in range(out.shape[0]):
        indexes = torch.nonzero(mask[batch], as_tuple=False).flatten().tolist()
        for i in indexes:
            for j in indexes:
                if i == j:
                    continue
                yc1, h1 = out[batch, i, 1], out[batch, i, 3]
                yc2, h2 = out[batch, j, 1], out[batch, j, 3]
                overlap = h1 / 2 + h2 / 2 - torch.abs(yc2 - yc1)
                if overlap > 0:
                    if yc1 < yc2:
                        out[batch, i, 1] -= overlap / 2
                        out[batch, j, 1] += overlap / 2
                    else:
                        out[batch, i, 1] += overlap / 2
                        out[batch, j, 1] -= overlap / 2
    return out


def apply_postprocessing(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
    *,
    mode: PostprocessingMode | str = PostprocessingMode.none,
    jitter_strength: float = 0.0,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "batch elements 4"]:
    """Apply LayoutDETR jitter/alignment/de-overlap without rendering."""
    out = jitter_boxes(bbox, strength=jitter_strength, generator=generator)
    normalized_mode = normalize_postprocessing_mode(mode)
    # Vendor generate.py contains a random-postprocessing assignment typo; the
    # public port keeps deterministic explicit modes only.
    if normalized_mode is PostprocessingMode.horizontal_center_aligned:
        out = horizontal_center_aligned(out, mask)
    elif normalized_mode is PostprocessingMode.horizontal_left_aligned:
        out = horizontal_left_aligned(out, mask)
    elif normalized_mode is not PostprocessingMode.none:
        raise ValueError(f"Unsupported out_postprocessing: {mode}")
    return clamp_boxes(de_overlap(out, mask))
