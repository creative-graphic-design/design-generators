"""LayoutDiffusion-specific S0-S2 training-parity helpers."""

from __future__ import annotations

import torch
from jaxtyping import Shaped
from laygen.common.training import LAYOUTDIFFUSION_TRAINING_TRACE_POINTS
from traingen_parity.compare import (
    OptimizerStepReport,
    StepReport,
    TensorTolerance,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import RNGState
from traingen_parity.trace import StepTrace, TrainingStepModule, trace_training_step

TRACE_POINTS: tuple[str, ...] = LAYOUTDIFFUSION_TRAINING_TRACE_POINTS


def trace_layoutdiffusion_step(
    module: TrainingStepModule,
    batch: dict[str, Shaped[torch.Tensor, ...]],
    rng_state: RNGState | None = None,
) -> StepTrace:
    """Trace one LayoutDiffusion training step with canonical trace points."""
    return trace_training_step(module, batch, rng_state, TRACE_POINTS)


def compare_layoutdiffusion_step(
    reference: StepTrace,
    target: StepTrace,
    *,
    tolerance: TensorTolerance | None = None,
) -> StepReport:
    """Compare S1 LayoutDiffusion pre-optimizer traces."""
    tolerances = {name: tolerance or TensorTolerance() for name in TRACE_POINTS}
    return compare_step_trace(reference, target, tolerances)


def compare_layoutdiffusion_optimizer_step(
    reference_state: dict[str, Shaped[torch.Tensor, ...]],
    target_state: dict[str, Shaped[torch.Tensor, ...]],
    *,
    tolerance: TensorTolerance | None = None,
) -> OptimizerStepReport:
    """Compare S2 LayoutDiffusion post-optimizer parameters."""
    tolerances = {name: tolerance or TensorTolerance() for name in reference_state}
    return compare_optimizer_step(reference_state, target_state, tolerances)
