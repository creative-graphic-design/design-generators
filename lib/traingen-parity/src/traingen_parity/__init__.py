"""Shared training parity helpers for generator packages."""

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
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
)
from .trace import (
    StepTrace,
    TensorSummary,
    TraceMetadata,
    TrainingStepModule,
    build_step_trace,
    scalar_trace_value,
    summarize_tensor,
    tensor_sha256,
    trace_training_step,
)

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
    "TraceMetadata",
    "TrainingStepModule",
    "apply_determinism",
    "build_step_trace",
    "capture_rng_state",
    "compare_batch_stream",
    "compare_optimizer_step",
    "compare_step_trace",
    "compare_tensors",
    "restore_rng_state",
    "scalar_trace_value",
    "summarize_tensor",
    "tensor_sha256",
    "trace_training_step",
]
