"""Comparison reports for training parity traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .trace import StepTrace


@dataclass(frozen=True)
class TensorTolerance:
    """Absolute and relative tolerances for tensor comparison."""

    atol: float = 0.0
    rtol: float = 0.0


@dataclass(frozen=True)
class TensorComparison:
    """Result for one tensor comparison."""

    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    message: str


@dataclass(frozen=True)
class StepReport:
    """Comparison report for two training-step traces."""

    passed: bool
    comparisons: tuple[TensorComparison, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerStepReport:
    """Comparison report for parameters after an optimizer step."""

    passed: bool
    comparisons: tuple[TensorComparison, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class BatchStreamReport:
    """Comparison report for two dataloader streams."""

    passed: bool
    checked_steps: int
    first_mismatch: str | None = None


def compare_tensors(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: TensorTolerance | None = None,
) -> TensorComparison:
    """Compare two tensors and return max-difference diagnostics.

    Args:
        name: Tensor name.
        actual: Actual tensor.
        expected: Expected tensor.
        tolerance: Absolute and relative tolerance. Defaults to exact equality.

    Returns:
        Tensor comparison report.

    Raises:
        RuntimeError: If tensors cannot be broadcast for comparison.

    Examples:
        >>> compare_tensors("x", torch.ones(1), torch.ones(1)).passed
        True
    """
    tol = tolerance or TensorTolerance()
    if actual.shape != expected.shape:
        return TensorComparison(
            name, False, float("inf"), float("inf"), "shape mismatch"
        )
    actual_detached = actual.detach()
    expected_detached = expected.detach()
    diff = (actual_detached.float() - expected_detached.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_detached.float().abs().clamp_min(torch.finfo(diff.dtype).eps)
    rel = diff / denom
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    if actual_detached.is_floating_point() or expected_detached.is_floating_point():
        passed = torch.allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)
    else:
        passed = torch.equal(actual, expected)
    message = "ok" if passed else f"max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"
    return TensorComparison(name, passed, max_abs, max_rel, message)


def compare_step_trace(
    reference: StepTrace,
    target: StepTrace,
    tolerances: Mapping[str, TensorTolerance] | None = None,
) -> StepReport:
    """Compare two named step traces.

    Args:
        reference: Reference trace.
        target: Converted implementation trace.
        tolerances: Per-tensor tolerance map.

    Returns:
        Step comparison report.

    Raises:
        RuntimeError: If tensor comparisons fail unexpectedly.

    Examples:
        >>> from traingen_parity.trace import build_step_trace
        >>> a = build_step_trace("a", {"x": torch.ones(1)})
        >>> compare_step_trace(a, a).passed
        True
    """
    tol_map = tolerances or {}
    names = tuple(reference.tensors.keys())
    missing = tuple(name for name in names if name not in target.tensors)
    comparisons = tuple(
        compare_tensors(
            name,
            target.tensors[name],
            reference.tensors[name],
            tol_map.get(name),
        )
        for name in names
        if name not in missing
    )
    return StepReport(
        passed=not missing and all(item.passed for item in comparisons),
        comparisons=comparisons,
        missing=missing,
    )


def compare_optimizer_step(
    reference_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tolerances: Mapping[str, TensorTolerance] | None = None,
) -> OptimizerStepReport:
    """Compare two state dictionaries after an optimizer step."""
    tol_map = tolerances or {}
    missing = tuple(name for name in reference_state if name not in target_state)
    comparisons = tuple(
        compare_tensors(
            name, target_state[name], reference_state[name], tol_map.get(name)
        )
        for name in reference_state
        if name not in missing
    )
    return OptimizerStepReport(
        passed=not missing and all(item.passed for item in comparisons),
        comparisons=comparisons,
        missing=missing,
    )


def compare_batch_stream(
    reference_loader: Iterable[Mapping[str, torch.Tensor]],
    target_loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    steps: int,
) -> BatchStreamReport:
    """Compare two dataloader streams for exact tensor equality."""
    for step, (reference_batch, target_batch) in enumerate(
        zip(reference_loader, target_loader, strict=False)
    ):
        if step >= steps:
            break
        for key, reference_value in reference_batch.items():
            target_value = target_batch.get(key)
            if isinstance(reference_value, torch.Tensor) and isinstance(
                target_value, torch.Tensor
            ):
                if not torch.equal(reference_value, target_value):
                    return BatchStreamReport(False, step + 1, f"{step}:{key}")
    return BatchStreamReport(True, steps)
