from __future__ import annotations

import pytest
import torch
from tkg_dm import apply_tkg_noise


def test_apply_tkg_noise_shifts_only_color_channels() -> None:
    latents = torch.full((1, 4, 16, 16), -1.0)

    transformed = apply_tkg_noise(latents, target_shift=0.07)

    assert transformed.shape == latents.shape
    assert transformed.dtype == latents.dtype
    assert torch.equal(transformed[:, 0], latents[:, 0])
    assert torch.equal(transformed[:, 3], latents[:, 3])
    assert torch.all(transformed[:, 1] > latents[:, 1])
    assert torch.all(transformed[:, 2] > latents[:, 2])


def test_zero_shift_returns_the_same_noise() -> None:
    latents = torch.arange(-128, 128, dtype=torch.float32).reshape(1, 4, 8, 8)

    transformed = apply_tkg_noise(latents, target_shift=0.0)

    torch.testing.assert_close(transformed, latents, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("latents", "target_shift", "sigma", "message"),
    [
        (torch.zeros((4, 8, 8)), 0.07, 0.5, "batch, channel, height, and width"),
        (torch.zeros((1, 2, 8, 8)), 0.07, 0.5, "at least three channels"),
        (torch.zeros((1, 4, 7, 8)), 0.07, 0.5, "at least 8"),
        (torch.zeros((1, 4, 8, 8), dtype=torch.int64), 0.07, 0.5, "floating-point"),
        (torch.ones((1, 4, 8, 8)), 0.1, 0.5, "positive ratio above 1"),
        (torch.zeros((1, 4, 8, 8)), -0.1, 0.5, "finite and non-negative"),
        (torch.zeros((1, 4, 8, 8)), 0.07, 0.0, "sigma must be finite and positive"),
    ],
)
def test_apply_tkg_noise_rejects_invalid_inputs(
    latents: torch.Tensor,
    target_shift: float,
    sigma: float,
    message: str,
) -> None:
    error_type = TypeError if not latents.is_floating_point() else ValueError

    with pytest.raises(error_type, match=message):
        apply_tkg_noise(latents, target_shift=target_shift, sigma=sigma)
