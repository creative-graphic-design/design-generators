"""Temporary diagnostics for PixelVAE checkpoint decisions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any  # noqa: TID251 - Lightning checkpoint payloads are heterogeneous.

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class CheckpointDecisionLogger(Callback):
    """Record checkpoint callback state at each training and validation boundary."""

    def __init__(self) -> None:
        """Read the diagnostic log path supplied by the segment runner."""
        self.log_path = Path(os.environ["PIXELVAE_CHECKPOINT_DECISIONS_LOG"])

    def _write(self, event: str, record: dict[str, Any]) -> None:
        payload = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        if re.search(r"hf_[A-Za-z0-9]{20,}", serialized, re.IGNORECASE):
            raise ValueError("checkpoint diagnostic contains an HF token-shaped value")

        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def _checkpoint_rows(self, trainer: pl.Trainer) -> list[dict[str, Any]]:
        rows = []
        for index, callback in enumerate(trainer.callbacks):
            if not isinstance(callback, ModelCheckpoint):
                continue

            try:
                should_skip = callback._should_skip_saving_checkpoint(trainer)
            except Exception as error:
                should_skip = f"error:{type(error).__name__}:{error}"

            try:
                save_on_train_epoch_end_effective = (
                    callback._should_save_on_train_epoch_end(trainer)
                )
            except Exception as error:
                save_on_train_epoch_end_effective = (
                    f"error:{type(error).__name__}:{error}"
                )

            checkpoint_dir = Path(callback.dirpath) if callback.dirpath else None
            last_path = checkpoint_dir / "last.ckpt" if checkpoint_dir else None
            rows.append(
                {
                    "index": index,
                    "state_key": callback.state_key,
                    "dirpath": callback.dirpath,
                    "save_last": callback.save_last,
                    "save_top_k": callback.save_top_k,
                    "monitor": callback.monitor,
                    "every_n_epochs": callback.every_n_epochs,
                    "save_on_train_epoch_end": callback._save_on_train_epoch_end,
                    "save_on_train_epoch_end_effective": save_on_train_epoch_end_effective,
                    "_last_global_step_saved": callback._last_global_step_saved,
                    "_last_checkpoint_saved": callback._last_checkpoint_saved,
                    "trainer_global_step": trainer.global_step,
                    "trainer_current_epoch": trainer.current_epoch,
                    "_should_skip_saving_checkpoint": should_skip,
                    "last_model_path": callback.last_model_path,
                    "best_model_path": callback.best_model_path,
                    "last_ckpt_exists": bool(last_path and last_path.exists()),
                    "last_ckpt_is_symlink": bool(last_path and last_path.is_symlink()),
                    "state_dict": callback.state_dict(),
                }
            )

        return rows

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Record serialized callback states found in the resume checkpoint."""
        del pl_module
        callbacks = checkpoint.get("callbacks", {})
        states = {
            key: {
                "serialized_state": value,
                "_last_global_step_saved_serialized": value.get(
                    "_last_global_step_saved"
                ),
                "_last_checkpoint_saved_serialized": value.get(
                    "_last_checkpoint_saved"
                ),
                "has_dirpath": "dirpath" in value,
                "has_last_model_path": "last_model_path" in value,
            }
            for key, value in callbacks.items()
            if isinstance(value, dict)
            and ("dirpath" in value or "last_model_path" in value)
        }
        self._write(
            "checkpoint_state_from_input",
            {
                "trainer_global_step": trainer.global_step,
                "trainer_current_epoch": trainer.current_epoch,
                "model_checkpoint_states": states,
            },
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record callback state after Lightning restores the checkpoint."""
        del pl_module
        self._write(
            "restored_checkpoint_state_at_fit_start",
            {"checkpoints": self._checkpoint_rows(trainer)},
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record checkpoint callback state at the end of each training epoch."""
        del pl_module
        self._write(
            "on_train_epoch_end", {"checkpoints": self._checkpoint_rows(trainer)}
        )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record checkpoint callback state at the end of each validation run."""
        del pl_module
        self._write(
            "on_validation_end", {"checkpoints": self._checkpoint_rows(trainer)}
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record checkpoint callback state after training stops."""
        del pl_module
        self._write("on_train_end", {"checkpoints": self._checkpoint_rows(trainer)})
