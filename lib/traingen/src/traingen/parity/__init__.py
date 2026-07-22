"""Parity tracing and comparison utilities."""

from __future__ import annotations

from .compare import (
    BatchStreamReport,
    OptimizerStepReport,
    StepReport,
    TensorComparison,
    TensorTolerance,
    compare_batch_stream,
    compare_optimizer_step,
    compare_step_trace,
    compare_tensors,
)
from .determinism import (
    DeterminismConfig,
    RNGState,
    capture_rng_state,
    restore_rng_state,
)
from .trace import StepTrace, TensorSummary, summarize_tensor, tensor_sha256

__all__ = [
    "BatchStreamReport",
    "DeterminismConfig",
    "OptimizerStepReport",
    "RNGState",
    "StepReport",
    "StepTrace",
    "TensorComparison",
    "TensorSummary",
    "TensorTolerance",
    "capture_rng_state",
    "compare_batch_stream",
    "compare_optimizer_step",
    "compare_step_trace",
    "compare_tensors",
    "restore_rng_state",
    "summarize_tensor",
    "tensor_sha256",
]
