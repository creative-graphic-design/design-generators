from __future__ import annotations

import pytest
import torch
from tkg_dm import apply_tkg_noise, shift_tkg_noise_channels


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


def test_shift_tkg_noise_channels_returns_unmasked_channel_shift() -> None:
    latents = torch.linspace(-1, 1, 16 * 16).reshape(1, 1, 16, 16).expand(1, 4, -1, -1)

    shifted = shift_tkg_noise_channels(latents, target_shift=0.07)

    assert shifted.shape == latents.shape
    assert shifted.dtype == latents.dtype
    assert torch.equal(shifted[:, 0], latents[:, 0])
    assert torch.equal(shifted[:, 3], latents[:, 3])
    assert torch.all(shifted[:, 1] > latents[:, 1])
    assert torch.all(shifted[:, 2] > latents[:, 2])
    for channel in (1, 2):
        channel_shift = shifted[:, channel] - latents[:, channel]
        expected_shift = channel_shift.flatten()[0].expand_as(channel_shift)
        torch.testing.assert_close(channel_shift, expected_shift, rtol=0, atol=1e-6)


def test_zero_channel_shift_returns_the_original_noise() -> None:
    latents = torch.arange(-128, 128, dtype=torch.float32).reshape(1, 4, 8, 8)

    shifted = shift_tkg_noise_channels(latents, target_shift=0.0)

    assert shifted is latents


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
