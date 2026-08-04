"""LayoutDM-specific S0-S2 training-parity helpers."""

from __future__ import annotations

import torch
from jaxtyping import Shaped

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
    "t",
    "pt",
    "xt",
    "log_model_prob",
    "kl",
    "decoder_nll",
    "kl_loss",
    "aux_loss",
    "train_loss",
)


def trace_layout_dm_step(
    module: TrainingStepModule,
    batch: dict[str, Shaped[torch.Tensor, "..."]],
    rng_state: RNGState | None = None,
) -> StepTrace:
    """Trace one LayoutDM training step with the canonical trace points."""
    return trace_training_step(module, batch, rng_state, TRACE_POINTS)


def compare_layout_dm_step(
    reference: StepTrace,
    target: StepTrace,
    *,
    tolerance: TensorTolerance | None = None,
) -> StepReport:
    """Compare S1 LayoutDM pre-optimizer traces."""
    names = set(reference.tensors) & set(target.tensors)
    tolerances = {name: tolerance or TensorTolerance() for name in names}
    return compare_step_trace(reference, target, tolerances)


def compare_layout_dm_optimizer_step(
    reference_state: dict[str, Shaped[torch.Tensor, "..."]],
    target_state: dict[str, Shaped[torch.Tensor, "..."]],
    *,
    tolerance: TensorTolerance | None = None,
) -> OptimizerStepReport:
    """Compare S0/S2 LayoutDM parameter state dictionaries."""
    names = set(reference_state) & set(target_state)
    tolerances = {name: tolerance or TensorTolerance() for name in names}
    return compare_optimizer_step(reference_state, target_state, tolerances)
