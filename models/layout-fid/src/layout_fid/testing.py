"""Test helpers for layout FID packages."""

from __future__ import annotations

import numpy as np
import torch


def assert_feature_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert layout FID feature parity."""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_statistics_shape(mu: np.ndarray, sigma: np.ndarray) -> None:
    """Assert reference statistics have compatible shapes."""
    if mu.ndim != 1:
        raise AssertionError("mu must be one-dimensional")
    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise AssertionError("sigma must be square with feature_dim rows")
