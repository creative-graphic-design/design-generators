"""Callbacks that enforce package training behavior."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..data import CrelloBatch


class StopOnNonFiniteLoss(L.Callback):
    """Stop after the first training batch with a non-finite loss."""

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: CrelloBatch,
        batch_idx: int,
    ) -> None:
        """Request training stop when a completed batch loss is not finite."""
        del pl_module, batch, batch_idx
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("CanvasVAE training step did not return a tensor loss")

        if not bool(torch.isfinite(loss.detach()).all()):
            trainer.should_stop = True
