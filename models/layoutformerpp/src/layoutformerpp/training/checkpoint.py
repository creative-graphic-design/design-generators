"""Lightning checkpoint hooks for resumable LayoutFormer++ training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer

from traingen.checkpointing import (
    SafeValue,
    capture_rng_state,
    is_safe_mapping,
    load_weights_only_checkpoint,
    publish_checkpoint_bundle,
    retain_checkpoint_bundles,
    restore_rng_state,
)


def _identity() -> dict[str, SafeValue]:
    raw = os.environ.get("LAYOUTFORMERPP_CHECKPOINT_IDENTITY")
    if not raw:
        return {}

    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("LAYOUTFORMERPP_CHECKPOINT_IDENTITY must be an object")

    return cast(dict[str, SafeValue], value)


def _string_mapping(value: SafeValue, name: str) -> dict[str, SafeValue]:
    if not is_safe_mapping(value):
        raise TypeError(f"{name} must be a safe mapping")

    result: dict[str, SafeValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")

        result[key] = item

    return result


class LayoutFormerPPModelCheckpoint(ModelCheckpoint):
    """Save complete Lightning state at every epoch and preserve the best save."""

    def __init__(
        self,
        *,
        dirpath: str | None = None,
        monitor: str | None = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = False,
    ) -> None:
        """Initialize a callback with an owner-selected checkpoint directory."""
        pinned_dirpath = dirpath or os.environ.get("LAYOUTFORMERPP_CHECKPOINT_DIR")
        if not pinned_dirpath:
            raise ValueError("LayoutFormer++ checkpoint dirpath must be pinned")

        super().__init__(
            dirpath=pinned_dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            save_on_train_epoch_end=False,
        )
        self._last_epoch_saved: int | None = None

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, SafeValue],
    ) -> None:
        """Add package RNG, loader, and launch identity state to the save."""
        del pl_module
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "sampler_epoch"):
            datamodule.sampler_epoch = trainer.current_epoch

        checkpoint["layoutformerpp_identity"] = _identity()
        checkpoint["layoutformerpp_rng"] = capture_rng_state()
        checkpoint["layoutformerpp_datamodule"] = (
            datamodule.state_dict() if datamodule is not None else {}
        )

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, SafeValue],
    ) -> None:
        """Restore package RNG and loader state before native Lightning resume."""
        del pl_module
        identity = checkpoint.get("layoutformerpp_identity")
        if identity is not None and identity != _identity():
            raise RuntimeError("checkpoint identity mismatch")

        rng = checkpoint.get("layoutformerpp_rng")
        if rng is not None:
            restore_rng_state(_string_mapping(rng, "layoutformerpp_rng"))

        datamodule = getattr(trainer, "datamodule", None)
        state = checkpoint.get("layoutformerpp_datamodule")
        if datamodule is not None and state is not None:
            datamodule.load_state_dict(_string_mapping(state, "datamodule state"))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a complete safe-to-load checkpoint after every epoch."""
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.sanity_checking or self._last_epoch_saved == trainer.current_epoch:
            return

        if self.dirpath is None:
            raise RuntimeError("LayoutFormer++ checkpoint dirpath must be pinned")

        path = Path(self.dirpath) / (
            f"epoch-{trainer.current_epoch:06d}-step-{trainer.global_step:09d}.ckpt"
        )
        trainer.save_checkpoint(str(path), weights_only=False)
        self._last_epoch_saved = trainer.current_epoch
        self._retain_epoch_checkpoints()
        self._publish(path)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Publish a newly selected best checkpoint without changing evaluation."""
        previous_best = self.best_model_path
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path and self.best_model_path != previous_best:
            self._publish(Path(self.best_model_path), best=True)

    def _retain_epoch_checkpoints(self) -> None:
        if self.dirpath is None:
            return

        paths = sorted(Path(self.dirpath).glob("epoch-*.ckpt"))
        keep = set(paths[-2:])
        if self.best_model_path:
            keep.add(Path(self.best_model_path))

        for path in paths:
            if path not in keep:
                path.unlink()

    def _publish(self, path: Path, *, best: bool = False) -> None:
        manifest = os.environ.get("LAYOUTFORMERPP_LAUNCH_MANIFEST")
        durable = os.environ.get("LAYOUTFORMERPP_DURABLE_ROOT")
        track = os.environ.get("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE")
        if not manifest or not durable or not track:
            return

        relative = Path(track) / (Path("best") / path.stem if best else path.stem)
        publish_checkpoint_bundle(
            {"checkpoint.ckpt": path}, Path(manifest), Path(durable), relative
        )
        retain_checkpoint_bundles(Path(durable), track)


__all__ = ["LayoutFormerPPModelCheckpoint", "load_weights_only_checkpoint"]
