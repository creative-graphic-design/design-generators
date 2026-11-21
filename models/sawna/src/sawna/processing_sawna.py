"""Space-aware initial-noise processing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float
from tkg_dm import shift_tkg_noise_channels

BoundingBox: TypeAlias = tuple[float, float, float, float]
DEFAULT_TARGET_SHIFT: Final[float] = 0.07


def apply_sawna_noise(
    latents: Float[torch.Tensor, "batch channel height width"],
    bounding_boxes: Sequence[BoundingBox],
    *,
    target_shift: float = DEFAULT_TARGET_SHIFT,
    blur_sigma: float | None = None,
) -> Float[torch.Tensor, "batch channel height width"]:
    """Blend channel-shifted noise into blurred reserved regions.

    Args:
        latents: Four-dimensional initial noise with at least three channels.
        bounding_boxes: Normalized ``(x1, y1, x2, y2)`` reserved regions.
        target_shift: Increase in positive-value ratio for TKG-DM channels 1 and 2.
        blur_sigma: Gaussian standard deviation in latent-grid pixels, or ``None``
            for two percent of the shorter latent-grid side.

    Returns:
        Initial noise with TKG-DM channel shifts blended into the reserved regions.

    Raises:
        ValueError: If ``target_shift``, ``blur_sigma``, or a bounding box is invalid.
        TypeError: If ``latents`` is not floating-point.

    Examples:
        No boxes preserve the caller's initial noise:

        >>> noise = torch.zeros((1, 4, 8, 8))
        >>> apply_sawna_noise(noise, []) is noise
        True
    """
    _validate_controls(target_shift=target_shift, blur_sigma=blur_sigma)

    if not bounding_boxes:
        return latents

    if latents.ndim != 4:
        raise ValueError("latents must have batch, channel, height, and width axes")

    if not latents.is_floating_point():
        raise TypeError("latents must use a floating-point dtype")

    if latents.shape[0] < 1 or latents.shape[1] < 3:
        raise ValueError("latents must contain a batch and at least three channels")

    height, width = latents.shape[-2:]
    if height < 8 or width < 8:
        raise ValueError("latent height and width must each be at least 8")

    mask = _create_blurred_bbox_mask(
        height,
        width,
        bounding_boxes,
        blur_sigma=blur_sigma,
        device=latents.device,
    )
    shifted_noise = shift_tkg_noise_channels(latents, target_shift=target_shift)

    # The reference casts the latent-grid mask to the noise dtype before blending.
    mask = mask.to(dtype=latents.dtype).expand_as(latents)
    return latents + mask * (shifted_noise - latents)


def _validate_controls(*, target_shift: float, blur_sigma: float | None) -> None:
    if not math.isfinite(target_shift) or target_shift < 0:
        raise ValueError("target_shift must be finite and non-negative")

    if blur_sigma is not None and (not math.isfinite(blur_sigma) or blur_sigma <= 0):
        raise ValueError("blur_sigma must be finite and positive when supplied")


def _create_blurred_bbox_mask(
    height: int,
    width: int,
    bounding_boxes: Sequence[BoundingBox],
    *,
    blur_sigma: float | None,
    device: torch.device,
) -> Float[torch.Tensor, "1 1 height width"]:
    # The reference creates and blurs the occupancy map at initial-latent size.
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)

    for box in bounding_boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("each bounding box must contain four finite coordinates")

        x1, y1, x2, y2 = box
        px1, py1 = int(x1 * width), int(y1 * height)
        px2, py2 = int(x2 * width), int(y2 * height)
        px1, px2 = sorted((max(0, min(width, px1)), max(0, min(width, px2))))
        py1, py2 = sorted((max(0, min(height, py1)), max(0, min(height, py2))))
        mask[py1:py2, px1:px2] = 1.0

    sigma = blur_sigma
    if sigma is None:
        sigma = min(height, width) * 0.02

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    coordinates = torch.arange(kernel_size, dtype=mask.dtype, device=device)
    coordinates = coordinates - kernel_size // 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 1, -1)
    mask = mask[None, None, :, :]

    # The reference uses separable zero-padded convolutions for the Gaussian blur.
    padding = kernel_size // 2
    mask = F.conv2d(mask, kernel, padding=(0, padding))
    mask = F.conv2d(mask, kernel.transpose(-1, -2), padding=(padding, 0))

    return mask
