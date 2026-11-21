from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
import torch
from tkg_dm import shift_tkg_noise_channels

from sawna.processing_sawna import (
    BoundingBox,
    _create_blurred_bbox_mask,
    apply_sawna_noise,
)


def test_no_boxes_preserve_the_initial_noise() -> None:
    latents = torch.zeros((1, 4, 64, 64))

    result = apply_sawna_noise(latents, [])

    assert result is latents


def test_reserved_box_blends_shifted_noise_only_through_its_mask() -> None:
    channel = torch.linspace(-1, 1, 64 * 64).reshape(1, 1, 64, 64)
    latents = channel.expand(1, 4, -1, -1).clone()
    boxes = [(0.25, 0.25, 0.75, 0.75)]
    mask = _create_blurred_bbox_mask(
        64,
        64,
        boxes,
        blur_sigma=None,
        device=latents.device,
    ).expand_as(latents)
    shifted = shift_tkg_noise_channels(latents, target_shift=0.07)

    result = apply_sawna_noise(latents, boxes)

    assert result.shape == latents.shape
    assert result.dtype == latents.dtype
    assert torch.equal(result, latents + mask * (shifted - latents))
    assert torch.equal(result[:, :, 0, 0], latents[:, :, 0, 0])
    assert torch.equal(result[:, :, 32, 32], shifted[:, :, 32, 32])


def test_blurred_bbox_mask_uses_latent_grid_dimensions() -> None:
    mask = _create_blurred_bbox_mask(
        64,
        80,
        [(0.25, 0.25, 0.75, 0.75)],
        blur_sigma=None,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 64, 80)
    assert mask.dtype == torch.float32
    assert mask[0, 0, 32, 40] == 1
    assert mask[0, 0, 0, 0] == 0
    assert 0 < mask[0, 0, 16, 20] < 1


@pytest.mark.parametrize(
    ("target_shift", "blur_sigma", "message"),
    [
        (-0.1, None, "target_shift must be finite and non-negative"),
        (math.nan, None, "target_shift must be finite and non-negative"),
        (0.07, 0.0, "blur_sigma must be finite and positive"),
        (0.07, math.inf, "blur_sigma must be finite and positive"),
    ],
)
def test_invalid_noise_controls_are_rejected(
    target_shift: float,
    blur_sigma: float | None,
    message: str,
) -> None:
    latents = torch.zeros((1, 4, 64, 64))

    with pytest.raises(ValueError, match=message):
        apply_sawna_noise(
            latents,
            [(0.3, 0.3, 0.7, 0.7)],
            target_shift=target_shift,
            blur_sigma=blur_sigma,
        )


def test_invalid_latent_shapes_and_dtypes_are_rejected() -> None:
    with pytest.raises(ValueError, match="batch, channel, height, and width"):
        apply_sawna_noise(torch.zeros((4, 64, 64)), [(0.3, 0.3, 0.7, 0.7)])

    with pytest.raises(TypeError, match="floating-point"):
        apply_sawna_noise(
            torch.zeros((1, 4, 64, 64), dtype=torch.int64),
            [(0.3, 0.3, 0.7, 0.7)],
        )

    with pytest.raises(ValueError, match="at least three channels"):
        apply_sawna_noise(torch.zeros((1, 2, 64, 64)), [(0.3, 0.3, 0.7, 0.7)])


@pytest.mark.parametrize(
    "bounding_boxes",
    [[(0.1, 0.2, 0.3)], [(0.1, 0.2, 0.3, math.nan)]],
)
def test_invalid_bounding_boxes_are_rejected(
    bounding_boxes: list[tuple[float, ...]],
) -> None:
    with pytest.raises(ValueError, match="four finite coordinates"):
        apply_sawna_noise(
            torch.zeros((1, 4, 64, 64)),
            cast(Sequence[BoundingBox], bounding_boxes),
        )
