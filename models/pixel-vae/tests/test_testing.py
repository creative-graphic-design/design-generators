from __future__ import annotations

import numpy as np
import pytest

from pixel_vae.testing import (
    assert_within_limits,
    calibrate_limits,
    ceil_two_significant_figures,
    max_absolute_difference,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (0.001845, 0.0019), (0.0675, 0.068), (123.01, 130.0)],
)
def test_ceil_two_significant_figures(value: float, expected: float) -> None:
    assert ceil_two_significant_figures(value) == expected


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_ceil_two_significant_figures_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ceil_two_significant_figures(value)


def test_calibrate_limits_uses_three_repeat_max_and_margin() -> None:
    limits = calibrate_limits(
        [
            {"mean": 0.001, "loss": 0.01},
            {"mean": 0.0012, "loss": 0.02},
            {"mean": 0.0008, "loss": 0.015},
        ]
    )
    assert limits == {"loss": 0.03, "mean": 0.0018}


@pytest.mark.parametrize(
    "repeats",
    [
        [],
        [{"mean": 1.0}],
        [{"mean": 1.0}, {"other": 1.0}, {"mean": 1.0}],
        [{"mean": -1.0}, {"mean": 1.0}, {"mean": 1.0}],
    ],
)
def test_calibrate_limits_rejects_incomplete_or_invalid_repeats(
    repeats: list[dict[str, float]],
) -> None:
    with pytest.raises(ValueError):
        calibrate_limits(repeats)


def test_max_absolute_difference_checks_shape_values_and_nonempty() -> None:
    assert max_absolute_difference(
        np.array([1.0, 2.0]), np.array([1.1, 1.9])
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="shapes"):
        max_absolute_difference(np.zeros(1), np.zeros(2))
    with pytest.raises(ValueError, match="empty"):
        max_absolute_difference(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        max_absolute_difference(np.array([np.nan]), np.zeros(1))


def test_assert_within_limits_checks_exact_metric_set_and_values() -> None:
    assert_within_limits({"mean": 0.1}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="metric sets"):
        assert_within_limits({"other": 0.1}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="exceed frozen"):
        assert_within_limits({"mean": 0.3}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="exceed frozen"):
        assert_within_limits({"mean": float("nan")}, {"mean": 0.2})
