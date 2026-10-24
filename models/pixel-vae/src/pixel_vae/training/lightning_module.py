"""Lightning training module for PixelVAE."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import cast

import lightning as L
import torch
from jaxtyping import Float
from torch.optim import Optimizer

from ..configuration_pixel_vae import PixelVAEConfig
from ..modeling_pixel_vae import PixelVAEModel
from .datamodule import PixelVAEDataModule
from traingen.optim import KerasAdam


class PixelVAEEpochTimingCallback(L.Callback):
    """Write each training epoch's train-and-validation wall time when requested."""

    def __init__(self, output_path: str | None = None) -> None:
        """Store an optional path for epoch timing records."""
        super().__init__()
        self.output_path = output_path
        self._epoch_started_at: float | None = None
        self._epoch_started_at_utc: str | None = None
        self._sampler_epoch: int | None = None
        self._next_sampler_epoch: int | None = None
        self._segment_start_checked = False

    def state_dict(self) -> dict[str, int]:
        """Persist the next source-order epoch across Lightning checkpoints."""
        if self._next_sampler_epoch is None:
            return {}

        return {"next_sampler_epoch": self._next_sampler_epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore the source-order epoch from a resumable checkpoint."""
        self._next_sampler_epoch = state_dict.get("next_sampler_epoch")
        self._segment_start_checked = False

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Resolve the optional runtime output path before training starts."""
        del trainer, pl_module, stage
        if self.output_path is None:
            self.output_path = os.environ.get("PIXELVAE_EPOCH_TIMING_FILE") or None

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Set and record the source-order index before each training epoch."""
        del pl_module
        datamodule = cast(PixelVAEDataModule, trainer.datamodule)
        sampler = datamodule.samplers["train"]
        if not self._segment_start_checked:
            expected_epoch = datamodule.sampler_start_epoch
            if (
                self._next_sampler_epoch is not None
                and self._next_sampler_epoch != expected_epoch
            ):
                raise RuntimeError(
                    "checkpoint sampler epoch does not match the segment start: "
                    f"{self._next_sampler_epoch} != {expected_epoch}"
                )
            self._next_sampler_epoch = expected_epoch

            self._segment_start_checked = True

        assert self._next_sampler_epoch is not None

        sampler.set_epoch(self._next_sampler_epoch)
        self._sampler_epoch = sampler.epoch
        self._next_sampler_epoch += 1
        self._epoch_started_at = monotonic()
        self._epoch_started_at_utc = datetime.now(timezone.utc).isoformat()

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Record elapsed time once the epoch's validation pass has finished."""
        del pl_module
        if trainer.sanity_checking or self._epoch_started_at is None:
            return

        elapsed = monotonic() - self._epoch_started_at
        finished_at = datetime.now(timezone.utc)
        started_at_utc = self._epoch_started_at_utc
        self._epoch_started_at = None
        self._epoch_started_at_utc = None
        if self.output_path is None:
            return

        assert self._sampler_epoch is not None, "sampler epoch was not recorded"
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, int | float | str] = {
            "epoch": self._sampler_epoch + 1,
            "lightning_epoch": trainer.current_epoch + 1,
            "sampler_epoch": self._sampler_epoch,
            "train_and_validation_wall_seconds": elapsed,
            "started_at_utc": started_at_utc or "",
            "finished_at_utc": finished_at.isoformat(),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._sampler_epoch = None


class PixelVAETrainingModule(L.LightningModule):
    """Train PixelVAE with the original reconstruction, KL, and L2 objective."""

    def __init__(
        self,
        latent_dim: int = 256,
        kl_weight: float = 100.0,
        l2_weight: float = 1e-6,
        learning_rate: float = 1e-4,
    ) -> None:
        """Build PixelVAE from the selected objective and optimizer settings."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = PixelVAEModel(
            PixelVAEConfig(
                latent_dim=latent_dim,
                kl_weight=kl_weight,
                l2_weight=l2_weight,
            )
        )

    def on_train_epoch_start(self) -> None:
        """Keep the legacy zero-worker loader's CPU RNG progression."""
        torch.empty((), dtype=torch.int64, device="cpu").random_()

    def on_validation_start(self) -> None:
        """Keep the legacy zero-worker validation loader's CPU RNG progression."""
        torch.empty((), dtype=torch.int64, device="cpu").random_()

    def training_step(
        self,
        batch: Mapping[str, Float[torch.Tensor, "batch 4 256 256"]],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Return and log the source objective for one image batch."""
        del batch_idx
        output = self.model(batch["pixel_values"], sample=True)
        assert output.loss is not None
        assert output.reconstruction_loss is not None
        assert output.kl_divergence is not None
        batch_size = int(batch["pixel_values"].shape[0])
        self.log("train_loss", output.loss, on_epoch=True, batch_size=batch_size)
        self.log(
            "train_reconstruction",
            output.reconstruction_loss,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_kl_divergence",
            output.kl_divergence,
            on_epoch=True,
            batch_size=batch_size,
        )
        return output.loss

    def validation_step(
        self,
        batch: Mapping[str, Float[torch.Tensor, "batch 4 256 256"]],
        batch_idx: int,
    ) -> None:
        """Log source validation reconstruction for best-checkpoint selection."""
        del batch_idx
        output = self.model(batch["pixel_values"], sample=True)
        assert output.reconstruction_loss is not None
        self.log(
            "val_reconstruction",
            output.reconstruction_loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch["pixel_values"].shape[0]),
        )

    def configure_optimizers(self) -> Optimizer:
        """Return the shared optimizer with Keras 2 Adam update semantics."""
        return KerasAdam(self.model.parameters(), lr=self.learning_rate)


__all__ = ["PixelVAETrainingModule", "PixelVAEEpochTimingCallback"]
