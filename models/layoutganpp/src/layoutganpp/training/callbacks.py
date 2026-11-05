"""Callbacks that preserve the original LayoutGAN++ validation cadence."""

from __future__ import annotations

import math
from typing import cast

from lightning.pytorch import Callback, LightningModule, Trainer

from .datamodule import LayoutGANPPDataModule
from .dataset import LayoutGANPPDataset


class LayoutGANPPVendorSchedule(Callback):
    """Match the original iteration target and validation cadence."""

    def __init__(self, *, iterations: int) -> None:
        """Initialize the original iteration target."""
        super().__init__()
        self.iterations = iterations
        self._batches_per_epoch: int | None = None
        self._max_epochs: int | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Use the original sample-count formula to finish at an epoch boundary."""
        del pl_module
        if self.iterations < 1:
            raise ValueError("iterations must be positive")

        datamodule_value = getattr(trainer, "datamodule", None)
        if datamodule_value is None:
            raise ValueError("training schedule requires a data module")

        datamodule = cast(LayoutGANPPDataModule, datamodule_value)
        train_loader = datamodule.train_dataloader()
        train_dataset = cast(LayoutGANPPDataset, train_loader.dataset)
        if train_loader.batch_size is None or len(train_dataset) < 1:
            raise ValueError("training schedule requires a nonempty batched dataset")

        train_size = len(train_dataset)
        batch_size = train_loader.batch_size
        batches_per_epoch = len(train_loader)
        if batches_per_epoch < 1:
            raise ValueError("training schedule requires at least one train batch")

        self._batches_per_epoch = batches_per_epoch
        if trainer.fit_loop.epoch_loop.max_steps > 0:
            self._max_epochs = None
            return

        if trainer.fit_loop.max_epochs == 1_000:
            raise ValueError(
                "max_epochs resolves to Lightning's default of 1000; set "
                "max_epochs=-1 for the reference schedule or use a non-default "
                "max_epochs or positive max_steps"
            )

        if trainer.fit_loop.max_epochs != -1:
            self._max_epochs = None
            return

        self._max_epochs = math.ceil(self.iterations * batch_size / train_size)
        trainer.fit_loop.max_epochs = self._max_epochs
        trainer.fit_loop.epoch_loop.max_steps = -1

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Schedule the next validation after 10,000 training iterations."""
        del pl_module
        if self._batches_per_epoch is None:
            raise RuntimeError("training schedule was not initialized")

        interval_epochs = math.ceil(10_000 / self._batches_per_epoch)
        next_epoch = trainer.current_epoch + 1 + interval_epochs
        trainer.check_val_every_n_epoch = (
            min(next_epoch, self._max_epochs)
            if self._max_epochs is not None
            else next_epoch
        )
