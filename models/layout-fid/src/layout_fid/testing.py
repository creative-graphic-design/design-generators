"""Test helpers for layout FID packages."""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float


def assert_feature_close(
    actual: Float[torch.Tensor, "batch channels"],
    expected: Float[torch.Tensor, "batch channels"],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert layout FID feature parity."""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_statistics_shape(
    mu: Float[np.ndarray, "channels"],
    sigma: Float[np.ndarray, "channels channels"],
) -> None:
    """Assert reference statistics have compatible shapes."""
    if mu.ndim != 1:
        raise AssertionError("mu must be one-dimensional")

    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise AssertionError("sigma must be square with feature_dim rows")
