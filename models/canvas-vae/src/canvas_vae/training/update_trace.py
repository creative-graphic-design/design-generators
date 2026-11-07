"""Record production training objectives and their input batches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from _thread import LockType
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ParamSpec, Protocol, TextIO, TypeAlias, TypeVar, cast

import lightning as L
import numpy as np
import numpy.typing as npt
from jaxtyping import Shaped
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..data import CRELLO_CONTEXT_FIELDS, CRELLO_SEQUENCE_FIELDS, CrelloBatch

TRACE_BATCH_FIELDS = (*CRELLO_CONTEXT_FIELDS, *CRELLO_SEQUENCE_FIELDS)
FitParameters = ParamSpec("FitParameters")
FitResult = TypeVar("FitResult")


class _TensorValue(Protocol):
    def numpy(self) -> Shaped[np.ndarray, "..."]: ...


class _DetachedTensorValue(Protocol):
    def cpu(self) -> _TensorValue: ...


class _TorchTensorValue(Protocol):
    def detach(self) -> _DetachedTensorValue: ...


TraceValue: TypeAlias = npt.ArrayLike | _TensorValue | _TorchTensorValue
TraceOutput: TypeAlias = Mapping[str, TraceValue]


class _KerasOptimizer(Protocol):
    iterations: _TensorValue


class _KerasModel(Protocol):
    optimizer: _KerasOptimizer | None
    compute_loss: Callable[..., TraceValue]
    train_step: Callable[..., TraceOutput]
    _canvas_vae_trace_active: bool
    fit: Callable[..., FitResult]


_DataSpecArgument: TypeAlias = bool | int | str | None
_DataSpecResult = TypeVar("_DataSpecResult", covariant=True)


class _DataSpec(Protocol[_DataSpecResult]):
    def make_dataset(
        self,
        split: str,
        *args: _DataSpecArgument,
        **kwargs: _DataSpecArgument,
    ) -> _DataSpecResult: ...


def _as_array(value: TraceValue) -> Shaped[np.ndarray, "..."]:
    if isinstance(value, np.ndarray | np.generic | float | int | bool):
        return np.asarray(value)

    if hasattr(value, "detach"):
        return np.asarray(cast(_TorchTensorValue, value).detach().cpu().numpy())

    if hasattr(value, "numpy"):
        return np.asarray(cast(_TensorValue, value).numpy())

    return np.asarray(value)


def _batch_sha256(batch: Mapping[str, TraceValue]) -> str:
    missing = set(TRACE_BATCH_FIELDS) - batch.keys()
    if missing:
        raise ValueError(f"training batch is missing fields: {sorted(missing)}")

    digest = hashlib.sha256()
    for name in TRACE_BATCH_FIELDS:
        values = _as_array(batch[name])
        if values.dtype.kind not in "biuf":
            raise ValueError(f"training batch field {name} is not numeric")

        dtype = values.dtype.newbyteorder("<")
        values = np.ascontiguousarray(values.astype(dtype, copy=False))
        metadata = json.dumps(
            [name, values.dtype.str, values.shape], separators=(",", ":")
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(b"\0")
        digest.update(values.tobytes())

    return digest.hexdigest()


@dataclass
class TrainingUpdateTrace:
    """Write one exact objective and input-batch digest per optimizer update."""

    path: str | Path
    system: str
    seed: int
    _handle: TextIO = field(init=False, repr=False)
    _lock: LockType = field(init=False, default_factory=threading.Lock, repr=False)
    _last_update: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate metadata and create the exclusive trace file."""
        if (
            self.system not in {"original", "package"}
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("training trace needs a valid system and seed")

        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.path = destination
        self._handle = destination.open("x", encoding="utf-8")
        self._write(
            {
                "record": "metadata",
                "schema_version": 1,
                "system": self.system,
                "seed": self.seed,
                "optimizer": "adam",
                "objective": "scalar loss used for the optimizer update",
                "batch_sha256_fields": list(TRACE_BATCH_FIELDS),
                "batch_sha256_encoding": "field name, dtype, shape, and C-order bytes in field order",
            }
        )

    @property
    def last_update(self) -> int:
        """Return the number of sequential updates already written."""
        return self._last_update

    def record(
        self, update: int, objective: TraceValue, batch: Mapping[str, TraceValue]
    ) -> None:
        """Append one completed optimizer update and its input batch hash."""
        with self._lock:
            if type(update) is not int or update != self._last_update + 1:
                raise ValueError("training trace updates must be consecutive from one")

            values = _as_array(objective)
            if values.size != 1:
                raise ValueError("training objective must be scalar")

            loss = float(values.reshape(-1)[0])
            if not math.isfinite(loss):
                raise ValueError("training objective must be finite")

            row = {
                "record": "update",
                "update": update,
                "objective": loss,
                "batch_sha256": _batch_sha256(batch),
            }
            self._write(row)
            self._last_update = update

    def close(self) -> None:
        """Close the JSONL file after the training run completes."""
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def _write(self, row: Mapping[str, str | int | float | bool | list[str]]) -> None:
        self._handle.write(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self._handle.flush()


class TrainingUpdateTraceCallback(L.Callback):
    """Optionally record each completed CanvasVAE package optimizer update."""

    def __init__(self) -> None:
        """Initialize an optional package-side update recorder."""
        super().__init__()
        self.trace: TrainingUpdateTrace | None = None
        self.last_global_step = 0
        self.initial_global_step = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Open the trace when the launcher requested one."""
        del pl_module
        trace_path = os.environ.get("CANVAS_VAE_UPDATE_TRACE")
        if not trace_path:
            return

        seed_value = os.environ.get("CANVAS_VAE_UPDATE_TRACE_SEED")
        if seed_value is None or not seed_value.isdecimal():
            raise ValueError(
                "training trace seed must be set to a non-negative integer"
            )

        if trainer.accumulate_grad_batches != 1 or len(trainer.optimizers) != 1:
            raise ValueError("training traces require one optimizer update per batch")

        self.trace = TrainingUpdateTrace(
            trace_path, system="package", seed=int(seed_value)
        )
        self.initial_global_step = self.last_global_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: CrelloBatch,
        batch_idx: int,
    ) -> None:
        """Record the loss and batch only after a completed optimizer step."""
        del pl_module, batch_idx
        if self.trace is None:
            return

        global_step = int(trainer.global_step)
        if global_step == self.last_global_step:
            return

        if global_step != self.last_global_step + 1:
            raise RuntimeError("package optimizer updates are not consecutive")

        if not isinstance(batch, Mapping):
            raise RuntimeError("package training batch must be a feature mapping")

        trace_batch = cast(Mapping[str, TraceValue], batch)
        if isinstance(outputs, Mapping):
            output_values = cast(Mapping[str, TraceValue], outputs)
            objective_value = output_values.get("loss")
            if objective_value is None:
                raise RuntimeError("package training step did not return its loss")

            objective = objective_value
        elif outputs is None:
            raise RuntimeError("package training step did not return its loss")
        else:
            objective = cast(TraceValue, outputs)

        self.trace.record(self.trace.last_update + 1, objective, trace_batch)
        self.last_global_step = global_step

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check the trace count and close its file at normal completion."""
        del pl_module
        if self.trace is not None:
            completed_updates = int(trainer.global_step) - self.initial_global_step
            if self.trace.last_update != completed_updates:
                raise RuntimeError(
                    "package trace count differs from completed optimizer updates"
                )

        self._close()

    def on_exception(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        """Close the trace if training exits with an exception."""
        del trainer, pl_module, exception
        self._close()

    def _close(self) -> None:
        if self.trace is None:
            return

        self.trace.close()
        self.trace = None


def install_tensorflow_training_trace(
    model_class: type[_KerasModel], path: str | Path, seed: int
) -> None:
    """Trace the original Keras trainer without changing its source files."""
    import tensorflow as tf

    original_fit = model_class.fit

    def traced_fit(
        model: _KerasModel,
        *args: FitParameters.args,
        **kwargs: FitParameters.kwargs,
    ) -> FitResult:
        optimizer = model.optimizer
        if optimizer is None:
            raise RuntimeError("original model must be compiled before tracing")

        initial_update = int(_as_array(optimizer.iterations).item())
        if initial_update != 0:
            raise ValueError("original training trace requires a fresh optimizer")

        trace = TrainingUpdateTrace(path, system="original", seed=seed)
        original_compute_loss = model.compute_loss
        original_train_step = model.train_step
        model._canvas_vae_trace_active = False

        def traced_compute_loss(
            x: Mapping[str, TraceValue] | None = None,
            y: TraceValue | None = None,
            y_pred: TraceValue | None = None,
            sample_weight: TraceValue | None = None,
        ) -> TraceValue:
            loss = original_compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
            if not model._canvas_vae_trace_active:
                return loss

            if not isinstance(x, Mapping):
                raise ValueError("original training inputs must be a feature mapping")

            batch = {name: x[name] for name in TRACE_BATCH_FIELDS}

            def record_update(loss_value: TraceValue, *values: TraceValue) -> np.int64:
                traced_batch = dict(zip(TRACE_BATCH_FIELDS, values, strict=True))
                trace.record(
                    trace.last_update + 1,
                    _as_array(loss_value),
                    traced_batch,
                )
                return np.int64(0)

            side_effect = tf.py_function(
                record_update,
                [loss, *(batch[name] for name in TRACE_BATCH_FIELDS)],
                Tout=tf.int64,
            )
            with tf.control_dependencies([side_effect]):
                return tf.identity(loss)

        def traced_train_step(data: Mapping[str, TraceValue]) -> TraceOutput:
            model._canvas_vae_trace_active = True
            try:
                return original_train_step(data)
            finally:
                model._canvas_vae_trace_active = False

        model.compute_loss = traced_compute_loss
        model.train_step = traced_train_step
        try:
            result = original_fit(model, *args, **kwargs)
            completed_updates = (
                int(_as_array(optimizer.iterations).item()) - initial_update
            )
            if trace.last_update != completed_updates:
                raise RuntimeError(
                    f"original trace count {trace.last_update} differs from "
                    f"completed optimizer updates {completed_updates}"
                )

            return result
        finally:
            trace.close()
            model.__dict__.pop("compute_loss", None)
            model.__dict__.pop("train_step", None)
            model.__dict__.pop("_canvas_vae_trace_active", None)
            model_class.fit = original_fit

    model_class.fit = traced_fit


def install_tensorflow_s4_train_stream(
    data_spec_class: type[_DataSpec[_DataSpecResult]],
) -> None:
    """Use the verified canonical record order without changing the source loader."""
    original_make_dataset = data_spec_class.make_dataset

    def make_dataset(
        data_spec: _DataSpec[_DataSpecResult],
        split: str,
        *args: _DataSpecArgument,
        **kwargs: _DataSpecArgument,
    ) -> _DataSpecResult:
        if split == "train" and kwargs.get("shuffle") is True:
            kwargs["shuffle"] = False

        return original_make_dataset(data_spec, split, *args, **kwargs)

    setattr(data_spec_class, "make_dataset", make_dataset)


__all__ = [
    "TRACE_BATCH_FIELDS",
    "TrainingUpdateTrace",
    "TrainingUpdateTraceCallback",
    "install_tensorflow_s4_train_stream",
    "install_tensorflow_training_trace",
]
