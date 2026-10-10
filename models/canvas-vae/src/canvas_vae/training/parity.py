"""Shared helpers for CanvasVAE training parity checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple, TypeVar, TypedDict

import numpy as np
import torch
from jaxtyping import Float

ZERO_GRADIENT_LIMIT = 1e-6
KERAS_ADAM_EPSILON = 1e-7
S2_ADAM_RULE_LIMIT = 3.5e-4
S2_WELL_CONDITIONED_LIMIT = 1.6e-3
S3_ADAM_RULE_LIMIT = 5.1e-4
S3_WELL_CONDITIONED_LIMIT = 6.1e-3
WELL_CONDITIONED_SQRT_V = 100 * KERAS_ADAM_EPSILON
_Value = TypeVar("_Value")


class AdamUpdateCheck(TypedDict):
    """RICO-style Adam-rule and well-conditioned update measurements."""

    norm_rel: float
    package_adam_rule: float
    original_adam_rule: float
    well_conditioned_fraction: float
    well_conditioned_elements: int
    elements: int
    ignore_limit_in_calibration: bool
    well_conditioned_norm_rel: float
    adam_rule_limit: float
    well_conditioned_limit: float | None
    package_adam_rule_within: bool
    original_adam_rule_within: bool
    well_conditioned_within: bool | None
    sqrt_v_threshold: float
    near_zero_difference_sq: float
    total_difference_sq: float
    near_zero_share_of_difference: float
    within: bool


class ZeroGradientCheck(NamedTuple):
    """Maximum absolute gradients and their shared zero-gradient result."""

    package_max_abs: float
    vendor_max_abs: float
    within_limit: bool


def keras_adam_float64(
    gradient: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    first_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    second_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    step: int,
) -> tuple[
    Float[torch.Tensor, "..."],
    Float[torch.Tensor, "..."],
    Float[torch.Tensor, "..."],
    Float[torch.Tensor, "..."],
]:
    """Return a per-tensor-clipped Keras Adam update and state in float64."""
    grad = torch.as_tensor(gradient, dtype=torch.float64)
    grad = grad * 1.0 / max(float(grad.norm()), 1.0)
    first = torch.as_tensor(first_moment, dtype=torch.float64)
    second = torch.as_tensor(second_moment, dtype=torch.float64)
    first = first + (grad - first) * (1 - 0.9)
    second = second + (grad * grad - second) * (1 - 0.999)
    alpha = 1e-3 * math.sqrt(1 - 0.999**step) / (1 - 0.9**step)
    update = -(first * alpha) / (second.sqrt() + KERAS_ADAM_EPSILON)
    return update, grad, first, second


def check_keras_adam_update(
    package_update: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    original_update: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    package_gradient: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    original_gradient: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    first_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    second_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    step: int,
    *,
    adam_rule_limit: float,
    well_conditioned_limit: float | None,
) -> AdamUpdateCheck:
    """Measure per-system Adam-rule errors and cross-system conditioned updates."""
    package_actual = torch.as_tensor(package_update, dtype=torch.float64)
    original_actual = torch.as_tensor(original_update, dtype=torch.float64)
    package_exact, _, _, _ = keras_adam_float64(
        package_gradient, first_moment, second_moment, step
    )
    original_exact, _, _, original_v = keras_adam_float64(
        original_gradient, first_moment, second_moment, step
    )
    well = original_v.sqrt() >= WELL_CONDITIONED_SQRT_V
    difference = package_actual - original_actual
    package_rule = float(
        (package_actual - package_exact).norm() / package_exact.norm().clamp_min(1e-30)
    )
    original_rule = float(
        (original_actual - original_exact).norm()
        / original_exact.norm().clamp_min(1e-30)
    )
    well_relative = (
        float(difference[well].norm() / original_actual[well].norm().clamp_min(1e-30))
        if well.any()
        else 0.0
    )
    near_zero_difference_sq = float(difference[~well].square().sum())
    total_difference_sq = float(difference.square().sum())
    package_within = package_rule <= adam_rule_limit
    original_within = original_rule <= adam_rule_limit
    well_within = (
        None
        if well_conditioned_limit is None
        else well_relative <= well_conditioned_limit
    )

    return {
        "norm_rel": float(difference.norm() / original_actual.norm().clamp_min(1e-30)),
        "package_adam_rule": package_rule,
        "original_adam_rule": original_rule,
        "well_conditioned_fraction": float(well.double().mean()),
        "well_conditioned_elements": int(well.sum()),
        "elements": difference.numel(),
        "ignore_limit_in_calibration": True,
        "well_conditioned_norm_rel": well_relative,
        "adam_rule_limit": adam_rule_limit,
        "well_conditioned_limit": well_conditioned_limit,
        "package_adam_rule_within": package_within,
        "original_adam_rule_within": original_within,
        "well_conditioned_within": well_within,
        "sqrt_v_threshold": WELL_CONDITIONED_SQRT_V,
        "near_zero_difference_sq": near_zero_difference_sq,
        "total_difference_sq": total_difference_sq,
        "near_zero_share_of_difference": (
            near_zero_difference_sq / max(total_difference_sq, 1e-300)
        ),
        "within": package_within
        and original_within
        and (well_within is None or well_within),
    }


def split_attention_key_biases(
    parameters: Mapping[str, _Value],
) -> tuple[dict[str, _Value], dict[str, _Value]]:
    """Separate softmax-invariant key biases from relative comparisons."""
    zero_gradient = {
        name: value
        for name, value in parameters.items()
        if name.endswith("attention.k_proj.bias")
    }
    relative = {
        name: value for name, value in parameters.items() if name not in zero_gradient
    }
    return relative, zero_gradient


def _max_abs_gradient(
    gradient: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
) -> float:
    values = (
        gradient.detach().cpu().numpy()
        if isinstance(gradient, torch.Tensor)
        else gradient
    )
    return float(np.max(np.abs(values), initial=0.0))


def check_zero_gradient(
    package_gradient: Float[torch.Tensor, "..."],
    vendor_gradient: Float[np.ndarray, "..."],
    limit: float = ZERO_GRADIENT_LIMIT,
) -> ZeroGradientCheck:
    """Check each implementation's maximum absolute gradient against the limit."""
    package_max_abs = _max_abs_gradient(package_gradient)
    vendor_max_abs = _max_abs_gradient(vendor_gradient)
    return ZeroGradientCheck(
        package_max_abs,
        vendor_max_abs,
        max(package_max_abs, vendor_max_abs) <= limit,
    )
