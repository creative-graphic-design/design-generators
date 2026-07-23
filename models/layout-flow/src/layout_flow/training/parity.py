"""LayoutFlow-specific S0-S2 parity helpers."""

from __future__ import annotations

import torch

from traingen_parity.compare import (
    OptimizerStepReport,
    StepReport,
    TensorTolerance,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import RNGState
from traingen_parity.trace import StepTrace, TrainingStepModule, trace_training_step

TRACE_POINTS: tuple[str, ...] = (
    "bbox",
    "type",
    "mask",
    "length",
    "cond_mask",
    "x0",
    "x1",
    "t",
    "xt",
    "ut",
    "vt",
    "flow_loss",
    "geom_l1_loss",
    "train_loss",
)


def trace_layout_flow_step(
    module: TrainingStepModule,
    batch: dict[str, torch.Tensor],
    rng_state: RNGState | None = None,
) -> StepTrace:
    """Trace one LayoutFlow training step with the canonical trace points."""
    return trace_training_step(module, batch, rng_state, TRACE_POINTS)


def compare_layout_flow_step(
    reference: StepTrace,
    target: StepTrace,
    *,
    tolerance: TensorTolerance | None = None,
) -> StepReport:
    """Compare S1 LayoutFlow pre-optimizer traces."""
    tolerances = {name: tolerance or TensorTolerance() for name in TRACE_POINTS}
    return compare_step_trace(reference, target, tolerances)


def compare_layout_flow_optimizer_step(
    reference_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    *,
    tolerance: TensorTolerance | None = None,
) -> OptimizerStepReport:
    """Compare S2 LayoutFlow post-optimizer parameters."""
    tolerances = {name: tolerance or TensorTolerance() for name in reference_state}
    return compare_optimizer_step(reference_state, target_state, tolerances)
