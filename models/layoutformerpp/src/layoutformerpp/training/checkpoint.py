"""Lightning checkpoint hooks for resumable LayoutFormer++ training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer

from traingen.checkpointing import (
    SafeValue,
    capture_rng_state,
    is_safe_mapping,
    publish_checkpoint_bundle,
    resolve_durable_root,
    retain_checkpoint_bundles,
    restore_rng_state,
    safe_string_mapping,
)


Identity: TypeAlias = dict[str, str | None]
_IDENTITY_KEYS = {
    "config_sha256",
    "source_commit",
    "vendor_source_commit",
    "device",
    "checkpoint_dirpath",
    "campaign_root",
}


def _identity() -> Identity:
    raw = os.environ.get("LAYOUTFORMERPP_CHECKPOINT_IDENTITY")
    if not raw:
        raise RuntimeError("LAYOUTFORMERPP_CHECKPOINT_IDENTITY is required")

    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("LAYOUTFORMERPP_CHECKPOINT_IDENTITY must be an object")

    identity: Identity = {}
    for key, item in value.items():
        if not isinstance(key, str) or (item is not None and not isinstance(item, str)):
            raise ValueError("checkpoint identity values must be strings or null")

        identity[key] = item

    if set(identity) != _IDENTITY_KEYS:
        raise ValueError("checkpoint identity must contain the complete identity")

    if any(not identity[key] for key in _IDENTITY_KEYS - {"vendor_source_commit"}):
        raise ValueError("checkpoint identity contains an empty value")

    return identity


def _checkpoint_dirpath(identity: Identity) -> Path:
    raw_dirpath = identity["checkpoint_dirpath"]
    if raw_dirpath is None:
        raise ValueError("checkpoint identity has no checkpoint dirpath")

    path = Path(raw_dirpath)
    if path.is_absolute():
        return path.resolve()

    campaign_root = os.environ.get("CAMPAIGN_ROOT")
    if not campaign_root:
        raise RuntimeError(
            "CAMPAIGN_ROOT is required for a relative checkpoint dirpath"
        )

    return (Path(campaign_root) / path).resolve()


def _validate_identity(dirpath: str | Path) -> Identity:
    identity = _identity()
    if Path(dirpath).resolve() != _checkpoint_dirpath(identity):
        raise RuntimeError("checkpoint identity mismatch: checkpoint_dirpath")

    return identity


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
        if dirpath is None:
            raise ValueError("LayoutFormer++ checkpoint dirpath must be pinned")

        super().__init__(
            dirpath=dirpath,
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
        """Add package RNG and launch identity state to the save."""
        del trainer, pl_module
        identity = _validate_identity(self.dirpath or "")
        identity_state: dict[str, SafeValue] = {}
        for key, value in identity.items():
            identity_state[key] = value

        checkpoint["layoutformerpp_identity"] = identity_state
        checkpoint["layoutformerpp_rng"] = capture_rng_state()

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, SafeValue],
    ) -> None:
        """Restore package RNG and validate launch identity before resume."""
        del trainer, pl_module
        identity = checkpoint.get("layoutformerpp_identity")
        if not is_safe_mapping(identity):
            raise RuntimeError("checkpoint identity is missing or invalid")

        if identity != _validate_identity(self.dirpath or ""):
            raise RuntimeError("checkpoint identity mismatch")

        rng = checkpoint.get("layoutformerpp_rng")
        restore_rng_state(safe_string_mapping(rng, "checkpoint RNG state"))

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
        self._publish(path, trainer.current_epoch)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Publish a newly selected best checkpoint without changing evaluation."""
        previous_best = self.best_model_path
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path and self.best_model_path != previous_best:
            self._publish(Path(self.best_model_path), trainer.current_epoch, best=True)

    def _retain_epoch_checkpoints(self) -> None:
        if self.dirpath is None:
            return

        paths = sorted(Path(self.dirpath).glob("epoch-*.ckpt"))
        for path in paths[:-2]:
            path.unlink()

    def _publish(self, path: Path, epoch: int, *, best: bool = False) -> None:
        manifest = os.environ.get("LAYOUTFORMERPP_LAUNCH_MANIFEST")
        durable = os.environ.get("LAYOUTFORMERPP_DURABLE_ROOT")
        track = os.environ.get("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE")
        if not manifest and not durable and not track:
            raise RuntimeError("durable checkpoint publishing is required")

        if not manifest or not durable or not track:
            raise RuntimeError("durable checkpoint publishing is partially configured")

        durable_root = resolve_durable_root(
            durable, env_var="LAYOUTFORMERPP_DURABLE_ROOT"
        )
        suffix = "-best" if best else ""
        relative = Path(track) / f"epoch-{epoch:06d}{suffix}"
        publish_checkpoint_bundle(
            {"checkpoint.ckpt": path}, Path(manifest), durable_root, relative
        )
        retain_checkpoint_bundles(durable_root, track)


__all__ = ["LayoutFormerPPModelCheckpoint"]
