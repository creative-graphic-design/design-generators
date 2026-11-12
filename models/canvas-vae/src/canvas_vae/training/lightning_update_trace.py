"""Lightning callback for package-side update traces."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..data import CrelloBatch
from ..run_metadata import write_run_metadata
from ..update_trace import TrainingUpdateTrace, TraceValue


class TrainingUpdateTraceCallback(L.Callback):
    """Optionally record each completed CanvasVAE package optimizer update."""

    def __init__(self) -> None:
        """Initialize an optional package-side update recorder."""
        super().__init__()
        self.trace: TrainingUpdateTrace | None = None
        self.trace_executor: ThreadPoolExecutor | None = None
        self.trace_future: Future[None] | None = None
        self.last_global_step = 0
        self.initial_global_step = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Open the trace when the launcher requested one."""
        del pl_module
        write_run_metadata(
            Path(trainer.default_root_dir) / "run-metadata.json", "torch"
        )
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
        self.trace_executor = ThreadPoolExecutor(max_workers=1)
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

        self._wait_for_trace()
        if self.trace_executor is None:
            raise RuntimeError("package trace executor is not initialized")

        self.trace_future = self.trace_executor.submit(
            self.trace.record, self.trace.last_update + 1, objective, trace_batch
        )
        self.last_global_step = global_step

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check the trace count and close its file at normal completion."""
        del pl_module
        self._wait_for_trace()
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
        self._wait_for_trace()
        if self.trace_executor is not None:
            self.trace_executor.shutdown(wait=True)
            self.trace_executor = None

        if self.trace is None:
            return

        self.trace.close()
        self.trace = None

    def _wait_for_trace(self) -> None:
        if self.trace_future is None:
            return

        future = self.trace_future
        self.trace_future = None
        future.result()
