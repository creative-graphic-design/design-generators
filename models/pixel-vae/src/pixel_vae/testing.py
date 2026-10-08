"""Numerical helpers for preregistered CPU parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING
import math

import numpy as np
from jaxtyping import Float


def max_absolute_difference(
    reference: Float[np.ndarray, "..."], candidate: Float[np.ndarray, "..."]
) -> float:
    """Return the largest absolute difference between equally shaped arrays.

    Args:
        reference: Reference values.
        candidate: Values from the converted implementation.

    Returns:
        Largest elementwise absolute difference.

    Raises:
        ValueError: If shapes differ, arrays are empty, or values are non-finite.
    """
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate shapes must match")

    if reference_array.size == 0:
        raise ValueError("parity arrays must not be empty")

    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("parity arrays must contain only finite values")

    return float(np.max(np.abs(reference_array - candidate_array)))


def ceil_two_significant_figures(value: float) -> float:
    """Round a non-negative discrepancy up to two significant figures.

    Args:
        value: Non-negative discrepancy bound.

    Returns:
        ``value`` rounded upward to two significant figures.

    Raises:
        ValueError: If the input is negative or non-finite.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be finite and non-negative")

    if value == 0:
        return 0.0

    decimal_value = Decimal(str(value))
    quantum = Decimal(1).scaleb(decimal_value.adjusted() - 1)
    rounded = (decimal_value / quantum).to_integral_value(rounding=ROUND_CEILING)
    return float(rounded * quantum)


def calibrate_limits(
    repeats: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Calibrate each metric from three independent CPU repeat maxima.

    Args:
        repeats: Exactly three metric-to-maximum-absolute-difference mappings.

    Returns:
        Per-metric limits computed as ``ceil2(1.5 * max(repeats))``.

    Raises:
        ValueError: If the repeat count or metric sets differ, or values are
            negative or non-finite.
    """
    if len(repeats) != 3:
        raise ValueError("calibration requires exactly three independent repeats")

    metrics = set(repeats[0])
    if not metrics or any(set(repeat) != metrics for repeat in repeats[1:]):
        raise ValueError("all calibration repeats must report the same metrics")

    limits = {}
    for metric in sorted(metrics):
        values = [float(repeat[metric]) for repeat in repeats]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                f"calibration values for {metric} must be finite and non-negative"
            )

        limits[metric] = ceil_two_significant_figures(1.5 * max(values))

    return limits


def assert_within_limits(
    observed: Mapping[str, float], limits: Mapping[str, float]
) -> None:
    """Raise when held-out metrics are missing or exceed frozen limits.

    Args:
        observed: Held-out maximum absolute differences.
        limits: Frozen calibration limits.

    Raises:
        AssertionError: If metric sets differ or any limit is exceeded.
    """
    if set(observed) != set(limits):
        raise AssertionError(
            "held-out metric sets do not match calibrated metrics: "
            f"missing={tuple(sorted(set(limits) - set(observed)))}, "
            f"unexpected={tuple(sorted(set(observed) - set(limits)))}"
        )

    failures = {
        key: (observed[key], limits[key])
        for key in sorted(limits)
        if not math.isfinite(observed[key]) or observed[key] > limits[key]
    }
    if failures:
        raise AssertionError(f"held-out metrics exceed frozen limits: {failures}")
