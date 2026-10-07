"""Opt-in training trace callback for package evidence runs."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Shaped
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn
from traingen_parity.trace import tensor_sha256


def _tensor_hashes(values: Mapping[str, Shaped[torch.Tensor, "..."]]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(values.items())}


def _gradient_norm(parameters: Mapping[str, nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name in sorted(parameters):
        parameter = parameters[name]
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float64)
            total += gradient.square().sum()

    return float(total.sqrt().item())


def _optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    values: dict[str, Shaped[torch.Tensor, "..."]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            for key, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    values[f"group{group_index}.parameter{parameter_index}.{key}"] = (
                        value
                    )

    return values


def _mapping_l2_norm(values: Mapping[str, Shaped[torch.Tensor, "..."]]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name in sorted(values):
        value = values[name]
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
        total += tensor.square().sum()

    return float(total.sqrt().item())


class LaceTrainingTraceCallback(Callback):
    """Write optimizer-boundary records when a trace path is configured."""

    def __init__(self) -> None:
        """Enable recording only when the caller supplies a trace path."""
        trace_path = os.environ.get("LACE_TRAINING_TRACE_PATH")
        self.trace_path = Path(trace_path) if trace_path else None
        self._batch_ids: list[str] = []
        self._record: dict[str, object] | None = None
        self._pre_clip_hashes: dict[str, str] = {}
        self._pre_clip_norm = 0.0
        self.records: list[dict[str, object]] = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Mapping[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> None:
        """Capture the current batch identifiers before its optimizer step."""
        del trainer, batch_idx
        if self.trace_path is None:
            return

        self._finalize_record(pl_module)
        ids = batch.get("id", [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise TypeError("LACE training batches must carry string ids")

        self._batch_ids = cast(list[str], ids)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Capture gradients before Lightning applies clipping."""
        del trainer
        if self.trace_path is None:
            return

        parameters = dict(pl_module.named_parameters())
        self._pre_clip_hashes = _tensor_hashes(
            {
                name: parameter.grad.detach()
                for name, parameter in parameters.items()
                if parameter.grad is not None
            }
        )
        self._pre_clip_norm = _gradient_norm(parameters)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Capture the loss and optimizer settings before the update."""
        del trainer
        if self.trace_path is None:
            return

        trace = getattr(pl_module, "latest_step_trace", {})
        self._record = {
            "batch_ids": list(self._batch_ids),
            "loss": float(cast(torch.Tensor, trace["train_loss"]).item()),
            "gradient_hashes": self._pre_clip_hashes,
            "gradient_norm": self._pre_clip_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Shaped[torch.Tensor, ""]
        | Mapping[str, Shaped[torch.Tensor, "..."] | list[str]]
        | None,
        batch: Mapping[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> None:
        """Capture clipped gradients and post-step state."""
        del trainer, outputs, batch, batch_idx
        if self.trace_path is None:
            return

        if self._record is None:
            raise RuntimeError("LACE training trace missed an optimizer boundary")

        optimizer = cast(torch.optim.Optimizer, pl_module.optimizers())
        model = cast(nn.Module, getattr(pl_module, "model"))
        parameters = dict(pl_module.named_parameters())
        gradients = {
            name: parameter.grad.detach()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }
        optimizer_state = _optimizer_state(optimizer)
        self._record.update(
            {
                "clipped_gradient_hashes": _tensor_hashes(gradients),
                "clipped_gradient_norm": _gradient_norm(parameters),
                "optimizer_state_hashes": _tensor_hashes(optimizer_state),
                "optimizer_state_l2_norm": _mapping_l2_norm(optimizer_state),
                "parameter_l2_norm": _mapping_l2_norm(model.state_dict()),
                "parameter_hashes": _tensor_hashes(model.state_dict()),
            }
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write records and the bounded trainer state to the trace path."""
        if self.trace_path is None:
            return

        self._finalize_record(pl_module)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        logger = trainer.logger
        logger_path = getattr(logger, "log_dir", None) if logger else None
        payload = {
            "records": self.records,
            "trainer_state": {
                "global_step": trainer.global_step,
                "current_epoch": trainer.current_epoch,
                "max_steps": trainer.max_steps,
                "num_training_batches": trainer.num_training_batches,
                "logger_path": str(logger_path) if logger_path else None,
            },
        }
        self.trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _finalize_record(self, pl_module: LightningModule) -> None:
        if self._record is None:
            return

        ema_state = cast(
            Mapping[str, torch.Tensor], getattr(pl_module, "latest_ema_state")
        )
        self.records.append({**self._record, "ema_hashes": _tensor_hashes(ema_state)})
        self._record = None
