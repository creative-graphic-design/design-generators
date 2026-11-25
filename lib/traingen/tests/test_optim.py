from __future__ import annotations

import pytest
import torch
from torch import nn

from traingen.optim import KerasAdam, clip_gradients_by_norm


def test_clip_gradients_by_norm_clips_each_tensor_independently() -> None:
    clipped = nn.Parameter(torch.zeros(2))
    under_limit = nn.Parameter(torch.zeros(2))
    no_gradient = nn.Parameter(torch.zeros(1))
    clipped.grad = torch.tensor([3.0, 4.0])
    under_limit.grad = torch.tensor([0.3, 0.4])

    clip_gradients_by_norm([clipped, under_limit, no_gradient], max_norm=1.0)

    torch.testing.assert_close(clipped.grad, torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(under_limit.grad, torch.tensor([0.3, 0.4]))
    assert no_gradient.grad is None


def test_clip_gradients_by_norm_handles_zero_and_rejects_invalid_limits() -> None:
    zero = nn.Parameter(torch.zeros(2))
    nonzero = nn.Parameter(torch.zeros(2))
    zero.grad = torch.zeros(2)
    nonzero.grad = torch.ones(2)

    clip_gradients_by_norm([zero, nonzero], max_norm=0.0)

    torch.testing.assert_close(zero.grad, torch.zeros(2))
    torch.testing.assert_close(nonzero.grad, torch.zeros(2))
    for max_norm in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            clip_gradients_by_norm([nonzero], max_norm=max_norm)


def test_keras_adam_matches_first_and_second_steps() -> None:
    weight = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = KerasAdam([weight], lr=0.1)
    weight.grad = torch.tensor([1.0, -1.0])
    optimizer.step()
    torch.testing.assert_close(
        weight.detach(), torch.tensor([0.9, -1.9]), rtol=1e-6, atol=2e-6
    )

    weight.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    torch.testing.assert_close(
        weight.detach(),
        torch.tensor([0.80678266, -1.8169408]),
        rtol=1e-6,
        atol=2e-6,
    )


def test_keras_adam_skips_missing_gradients_and_returns_closure_loss() -> None:
    weight = nn.Parameter(torch.ones(1))
    untouched = nn.Parameter(torch.ones(1))
    optimizer = KerasAdam([weight, untouched], lr=0.1)
    weight.grad = torch.ones_like(weight)
    result = optimizer.step(lambda: weight.sum())

    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.item() == 1.0
    assert untouched.item() == 1.0
