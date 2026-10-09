"""Numerical helpers for preregistered CPU parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING
import math
from typing import TypedDict

import numpy as np
from jaxtyping import Float


class Float32DifferenceSummary(TypedDict):
    """Scale and ULP details for one float32 comparison."""

    reference_max_abs: float
    candidate_max_abs: float
    max_abs_diff: float
    max_relative_diff: float
    max_diff_index: list[int]
    reference_at_max_diff: float
    candidate_at_max_diff: float
    ulps_at_max_diff: int


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


def float32_difference_summary(
    reference: Float[np.ndarray, "..."], candidate: Float[np.ndarray, "..."]
) -> Float32DifferenceSummary:
    """Summarize float32 magnitudes, relative infinity error, and ULP distance.

    The relative difference divides the maximum absolute difference by the
    largest absolute reference value. ULP distance is measured at the first
    index attaining the maximum absolute difference.
    """
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate shapes must match")

    if reference_array.size == 0:
        raise ValueError("parity arrays must not be empty")

    if reference_array.dtype != np.float32 or candidate_array.dtype != np.float32:
        raise ValueError("float32 difference summaries require float32 arrays")

    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("parity arrays must contain only finite values")

    difference = np.abs(reference_array - candidate_array)
    if not np.isfinite(difference).all():
        raise ValueError("float32 difference overflowed")

    flat_index = int(np.argmax(difference))
    index = np.unravel_index(flat_index, difference.shape)
    reference_value = np.float32(reference_array[index])
    candidate_value = np.float32(candidate_array[index])
    raw_reference = (
        np.ascontiguousarray(reference_array)
        .view(np.uint32)
        .reshape(reference_array.shape)
        .astype(np.uint64)
    )
    raw_candidate = (
        np.ascontiguousarray(candidate_array)
        .view(np.uint32)
        .reshape(candidate_array.shape)
        .astype(np.uint64)
    )
    sign_bit = np.uint64(0x80000000)
    value_mask = np.uint64(0xFFFFFFFF)
    ordered_reference = np.where(
        raw_reference & sign_bit,
        (~raw_reference) & value_mask,
        raw_reference | sign_bit,
    )
    ordered_candidate = np.where(
        raw_candidate & sign_bit,
        (~raw_candidate) & value_mask,
        raw_candidate | sign_bit,
    )
    ulps = (
        0
        if difference[index] == 0
        else abs(int(ordered_reference[index]) - int(ordered_candidate[index]))
    )
    reference_max_abs = float(np.max(np.abs(reference_array)))
    max_abs_diff = float(difference[index])
    max_relative_diff = max_abs_diff / max(
        reference_max_abs, float(np.finfo(np.float32).tiny)
    )

    return {
        "reference_max_abs": reference_max_abs,
        "candidate_max_abs": float(np.max(np.abs(candidate_array))),
        "max_abs_diff": max_abs_diff,
        "max_relative_diff": max_relative_diff,
        "max_diff_index": [int(value) for value in index],
        "reference_at_max_diff": float(reference_value),
        "candidate_at_max_diff": float(candidate_value),
        "ulps_at_max_diff": ulps,
    }


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
    lower_bounds: Mapping[str, float],
) -> dict[str, float]:
    """Calibrate each metric from three independent CPU input-pair maxima.

    Args:
        repeats: Exactly three metric-to-maximum-absolute-difference mappings.
        lower_bounds: Previously registered per-metric floors.

    Returns:
        Per-metric limits computed as ``max(lower_bound, ceil2(1.5 * max(repeats)))``.

    Raises:
        ValueError: If the repeat count, metric sets, or values are invalid.
    """
    if len(repeats) != 3:
        raise ValueError("calibration requires exactly three independent repeats")

    metrics = set(repeats[0])
    if not metrics or any(set(repeat) != metrics for repeat in repeats[1:]):
        raise ValueError("all calibration repeats must report the same metrics")

    if set(lower_bounds) != metrics:
        raise ValueError("lower-bound metrics must match calibration metrics")

    limits = {}
    for metric in sorted(metrics):
        values = [float(repeat[metric]) for repeat in repeats]
        lower_bound = float(lower_bounds[metric])
        if any(
            not math.isfinite(value) or value < 0 for value in [*values, lower_bound]
        ):
            raise ValueError(
                f"calibration values and lower bound for {metric} must be finite and non-negative"
            )

        limits[metric] = max(
            lower_bound, ceil_two_significant_figures(1.5 * max(values))
        )

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
