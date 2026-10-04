"""Shared parity helpers for the Layout-Corrector training adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import torch
from jaxtyping import Shaped
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from traingen_parity.compare import (
    OptimizerStepReport,
    StepReport,
    TensorTolerance,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import (
    DeterminismConfig,
    RNGState,
    apply_determinism,
    capture_rng_state,
)
from traingen_parity.trace import StepTrace, build_step_trace
from traingen_parity.trace import tensor_sha256

from .dataset import LayoutCorrectorDataModule
from .lightning_module import LayoutCorrectorTrainingModule


class _TrainerWithDataModule(Protocol):
    datamodule: LayoutCorrectorDataModule | None


if TYPE_CHECKING:
    DigestValue: TypeAlias = (
        Shaped[torch.Tensor, "..."]
        | Mapping[str | int, "DigestValue"]
        | Sequence["DigestValue"]
        | str
        | int
        | float
        | bool
        | None
    )


def _tensor_digest(value: Shaped[torch.Tensor, "..."]) -> str:
    """Digest one tensor using the shared parity representation."""
    return tensor_sha256(value)


def _parameter_state_digest(
    state: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> str:
    """Digest named parameters in the reference harness's stable form."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode())
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )

    return digest.hexdigest()


def _nested_value_digest(value: DigestValue) -> str:
    """Digest nested state with the reference harness's recursive encoding."""
    digest = hashlib.sha256()

    def update(item: DigestValue) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            digest.update(_tensor_digest(item).encode())

            return

        if isinstance(item, Mapping):
            digest.update(b"dict")
            pairs = sorted(item.items(), key=lambda pair: str(pair[0]))
            for key, child in pairs:
                digest.update(str(key).encode())
                update(cast("DigestValue", child))

            return

        if isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                update(child)

            return

        digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _optimizer_state_digest(optimizer: torch.optim.Optimizer) -> str:
    """Digest optimizer state with the reference harness's recursive encoding."""
    return _nested_value_digest(cast("DigestValue", optimizer.state_dict()))


def _rng_digest(state: RNGState) -> str:
    """Digest all RNG streams captured at a training-step boundary."""
    digest = hashlib.sha256()
    digest.update(
        state.torch_cpu.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    )
    for item in state.torch_cuda:
        digest.update(
            item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )

    digest.update(repr(state.python).encode())
    digest.update(str(state.numpy[0]).encode())
    digest.update(state.numpy[1].tobytes())
    return digest.hexdigest()


def _scheduler_state_digest(state: DigestValue) -> str:
    """Digest the configured scheduler state for the production record."""
    return _nested_value_digest(state)


class ProductionTraceCallback(Callback):
    """Record the configured Lightning training path for parity evidence."""

    def __init__(
        self,
        *,
        trace_path: str,
        initial_state_path: str,
        final_state_path: str,
        seed: int,
    ) -> None:
        """Initialize paths and buffers for one production training run.

        Args:
            trace_path: JSON output path for the per-step trace.
            initial_state_path: State dictionary loaded before training.
            final_state_path: Output path for the final model state.
            seed: Seed applied to the production training process.
        """
        self.trace_path = Path(trace_path)
        self.initial_state_path = Path(initial_state_path)
        self.final_state_path = Path(final_state_path)
        self.seed = seed
        self.rows: list[dict[str, object]] = []
        self._batch_rng_digest: str | None = None
        self._batch_digest: dict[str, str] = {}
        self._initial_cpu_rng_state: Shaped[torch.Tensor, "..."] | None = None
        self._pre_loader_rng_digest: str | None = None
        self._validation_executed = False
        self._validation_loss: float | None = None

    def _restore_initial_cpu_rng(self, batch_idx: int) -> None:
        """Undo Lightning's first iterator seed draw once per production run."""
        if batch_idx == 0:
            if self._initial_cpu_rng_state is None:
                raise RuntimeError("production trace callback did not initialize RNG")

            torch.set_rng_state(self._initial_cpu_rng_state)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Load the paired initial state before the first production batch."""
        del trainer
        initial_state = torch.load(
            self.initial_state_path, map_location="cpu", weights_only=True
        )
        if not isinstance(initial_state, dict):
            raise TypeError("the initial state artifact must be a state dictionary")

        module = cast(LayoutCorrectorTrainingModule, pl_module)
        module.model.model.load_state_dict(initial_state, strict=True)
        module.train()
        apply_determinism(
            DeterminismConfig(seed=self.seed, deterministic_algorithms=False)
        )
        self._initial_cpu_rng_state = torch.get_rng_state()
        self._pre_loader_rng_digest = _rng_digest(capture_rng_state())

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Mapping[str, Shaped[torch.Tensor, "..."] | str],
        batch_idx: int,
    ) -> None:
        """Capture input and RNG digests at the production batch boundary."""
        del trainer, pl_module
        self._restore_initial_cpu_rng(batch_idx)

        self._batch_rng_digest = _rng_digest(capture_rng_state())
        self._batch_digest = {
            "input_ids": _tensor_digest(
                cast(Shaped[torch.Tensor, "..."], batch["input_ids"])
            ),
            "attention_mask": _tensor_digest(
                cast(Shaped[torch.Tensor, "..."], batch["attention_mask"])
            ),
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Mapping[str, Shaped[torch.Tensor, "..."] | str],
        batch_idx: int,
    ) -> None:
        """Record the completed production update and its state digests."""
        del outputs
        module = cast(LayoutCorrectorTrainingModule, pl_module)
        trace = module.latest_step_trace
        gradient_norm = module.latest_gradient_norm
        if gradient_norm is None or self._batch_rng_digest is None:
            raise RuntimeError("production trace callback observed an incomplete step")

        optimizer = trainer.optimizers[0]
        sample_ids = batch.get("id", [])
        normalized_sample_ids = (
            [str(value) for value in sample_ids]
            if isinstance(sample_ids, (list, tuple))
            else [str(sample_ids)]
        )
        self.rows.append(
            {
                "step": batch_idx,
                "sample_ids": normalized_sample_ids,
                "loss": float(trace["train_loss"].detach().cpu().item()),
                "gradient_norm": float(gradient_norm.detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_state_digest": _optimizer_state_digest(optimizer),
                "parameter_state_digest": _parameter_state_digest(
                    module.model.model.state_dict()
                ),
                "rng_digest": self._batch_rng_digest,
                "input_ids_digest": self._batch_digest["input_ids"],
                "attention_mask_digest": self._batch_digest["attention_mask"],
                "timesteps_digest": _tensor_digest(trace["t"]),
                "importance_probability_digest": _tensor_digest(trace["pt"]),
                "corrupted_tokens_digest": _tensor_digest(trace["xt"]),
                "reconstructed_tokens_digest": _tensor_digest(trace["x0_recon"]),
                "model_training": module.model.training,
            }
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Capture the validation metric consumed by the plateau scheduler."""
        del pl_module
        metric = trainer.callback_metrics.get("val_loss")
        if not isinstance(metric, torch.Tensor):
            raise RuntimeError("production trace did not receive val_loss")

        self._validation_executed = True
        self._validation_loss = float(metric.detach().cpu().item())

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Persist the final state and production wiring metadata."""
        module = cast(LayoutCorrectorTrainingModule, pl_module)
        torch.save(module.model.model.state_dict(), self.final_state_path)
        scheduler_state: DigestValue = {}
        scheduler_class: str | None = None
        if trainer.lr_scheduler_configs:
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            scheduler_class = type(scheduler).__name__
            scheduler_state = cast("DigestValue", scheduler.state_dict())

        datamodule = cast(_TrainerWithDataModule, trainer).datamodule
        if datamodule is None:
            raise RuntimeError("production trace did not receive a data module")

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(
            json.dumps(
                {
                    "rows": self.rows,
                    "scheduler_class": scheduler_class,
                    "trace_seed": self.seed,
                    "pre_model_rng_digest": module.pre_model_rng_digest,
                    "pre_loader_rng_digest": self._pre_loader_rng_digest,
                    "scheduler_state_digest": _scheduler_state_digest(scheduler_state),
                    "validation_executed": self._validation_executed,
                    "validation_loss": self._validation_loss,
                    "validation_batches": trainer.num_val_batches,
                    "train_batches": trainer.num_training_batches,
                    "observed_train_batches": len(self.rows),
                    "max_steps": trainer.max_steps,
                    "num_workers": getattr(datamodule, "num_workers", None),
                    "max_epochs": trainer.max_epochs,
                    "model_training": module.model.training,
                },
                sort_keys=True,
            )
            + "\n"
        )


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
