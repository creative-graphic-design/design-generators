"""Tensor summaries and step traces for training parity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from jaxtyping import Float

from .determinism import RNGState, restore_rng_state

TraceMetadata: TypeAlias = dict[str, object]


class TrainingStepModule(Protocol):
    """Protocol for objects that expose a Lightning-like training step."""

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one training step."""


@dataclass(frozen=True)
class TensorSummary:
    """Compact deterministic summary of a tensor."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    min: float | None
    max: float | None
    mean: float | None


@dataclass(frozen=True)
class StepTrace:
    """Named tensor trace from a training step."""

    name: str
    tensors: dict[str, torch.Tensor]
    summaries: dict[str, TensorSummary]
    metadata: TraceMetadata


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a SHA-256 digest for tensor bytes on CPU.

    Args:
        tensor: Tensor to hash.

    Returns:
        Hex digest of the contiguous CPU tensor bytes.

    Raises:
        RuntimeError: If the tensor cannot be copied to CPU.

    Examples:
        >>> tensor_sha256(torch.tensor([1, 2])).startswith("0")
        False
    """
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def summarize_tensor(tensor: torch.Tensor) -> TensorSummary:
    """Build a deterministic tensor summary.

    Args:
        tensor: Tensor to summarize.

    Returns:
        Tensor summary dataclass.

    Raises:
        RuntimeError: If tensor statistics cannot be computed.

    Examples:
        >>> summarize_tensor(torch.ones(2)).mean
        1.0
    """
    detached = tensor.detach()
    stats = detached.float()
    if detached.numel() == 0:
        min_value = max_value = mean_value = None
    else:
        min_value = float(stats.min().item())
        max_value = float(stats.max().item())
        mean_value = float(stats.mean().item())
    return TensorSummary(
        shape=tuple(detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        sha256=tensor_sha256(detached),
        min=min_value,
        max=max_value,
        mean=mean_value,
    )


def build_step_trace(
    name: str,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: TraceMetadata | None = None,
) -> StepTrace:
    """Build a trace from named tensors.

    Args:
        name: Trace name.
        tensors: Named tensor values.
        metadata: Optional non-tensor metadata.

    Returns:
        Step trace with summaries.

    Raises:
        RuntimeError: If tensor summaries cannot be computed.

    Examples:
        >>> trace = build_step_trace("step", {"loss": torch.tensor(1.0)})
        >>> "loss" in trace.summaries
        True
    """
    return StepTrace(
        name=name,
        tensors=tensors,
        summaries={key: summarize_tensor(value) for key, value in tensors.items()},
        metadata=metadata or {},
    )


def trace_training_step(
    module: TrainingStepModule,
    batch: dict[str, torch.Tensor],
    rng_state: RNGState | None,
    trace_points: tuple[str, ...],
) -> StepTrace:
    """Run ``module.training_step`` and collect requested trace points.

    Args:
        module: Object exposing ``training_step`` and optionally
            ``latest_step_trace``.
        batch: Training batch.
        rng_state: Optional RNG state restored before the step.
        trace_points: Requested tensor names.

    Returns:
        Step trace for requested tensor names.

    Raises:
        AttributeError: If the module does not expose ``training_step``.

    Examples:
        >>> class M:
        ...     def training_step(self, batch, batch_idx):
        ...         self.latest_step_trace = {"loss": torch.tensor(1.0)}
        ...         return torch.tensor(1.0)
        >>> trace_training_step(M(), {}, None, ("loss",)).tensors["loss"].item()
        1.0
    """
    if rng_state is not None:
        restore_rng_state(rng_state)
    loss = module.training_step(batch, 0)
    raw = dict(getattr(module, "latest_step_trace", {}))
    raw.setdefault("train_loss", loss)
    tensors = {
        key: value
        for key, value in raw.items()
        if key in trace_points and isinstance(value, torch.Tensor)
    }
    return build_step_trace(
        getattr(module, "__class__", type(module)).__name__,
        tensors,
        metadata={"trace_points": trace_points},
    )


def scalar_trace_value(value: Float[torch.Tensor, ""]) -> float:
    """Return a Python scalar from a scalar tensor."""
    return float(value.detach().cpu().item())
