from __future__ import annotations

import torch
from torch import nn

from traingen.optim import KerasAdam


def test_keras_adam_matches_first_and_second_steps() -> None:
    weight = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = KerasAdam([weight], lr=0.1)
    weight.grad = torch.tensor([1.0, -1.0])
    optimizer.step()
    torch.testing.assert_close(
        weight.detach(), torch.tensor([0.9, -1.9]), rtol=0, atol=2e-7
    )

    weight.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    torch.testing.assert_close(
        weight.detach(), torch.tensor([0.80678266, -1.8169408]), rtol=0, atol=2e-7
    )


def test_keras_adam_skips_missing_gradients_and_returns_closure_loss() -> None:
    weight = nn.Parameter(torch.ones(1))
    untouched = nn.Parameter(torch.ones(1))
    optimizer = KerasAdam([weight, untouched], lr=0.1)
    weight.grad = torch.ones_like(weight)
    result = optimizer.step(lambda: weight.sum())

    assert result is not None
    assert result.item() == 1.0
    assert untouched.item() == 1.0
