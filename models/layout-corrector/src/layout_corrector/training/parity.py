"""Shared parity helpers for the Layout-Corrector training adapter."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from jaxtyping import Shaped

from traingen_parity.compare import (
    OptimizerStepReport,
    StepReport,
    TensorTolerance,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.trace import StepTrace, build_step_trace

TRACE_POINTS: tuple[str, ...] = (
    "t",
    "pt",
    "xt",
    "x0_recon",
    "recon_acc",
    "padding_mask",
    "logits",
    "bce_loss",
    "weighted_bce_loss",
    "train_loss",
)


def build_layout_corrector_step_trace(
    name: str,
    values: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> StepTrace:
    """Build the canonical Layout-Corrector trace from named tensors."""
    tensors = {key: value for key, value in values.items() if key in TRACE_POINTS}
    return build_step_trace(name, tensors, metadata={"trace_points": TRACE_POINTS})


def compare_layout_corrector_step(
    reference: StepTrace,
    target: StepTrace,
    *,
    tolerance: TensorTolerance | None = None,
) -> StepReport:
    """Compare the package and original pre-optimizer traces exactly."""
    tolerances = {name: tolerance or TensorTolerance() for name in reference.tensors}
    return compare_step_trace(reference, target, tolerances)


def compare_layout_corrector_optimizer_step(
    reference_state: Mapping[str, Shaped[torch.Tensor, "..."]],
    target_state: Mapping[str, Shaped[torch.Tensor, "..."]],
    *,
    tolerance: TensorTolerance | None = None,
) -> OptimizerStepReport:
    """Compare post-step package and original parameter state exactly."""
    tolerances = {name: tolerance or TensorTolerance() for name in reference_state}
    return compare_optimizer_step(reference_state, target_state, tolerances)
