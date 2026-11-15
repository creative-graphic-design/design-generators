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
    safe_value_type,
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
    if not path.is_absolute():
        raise ValueError("checkpoint identity dirpath must be absolute")

    return path.resolve()


def _validate_identity(dirpath: str | Path) -> Identity:
    identity = _identity()
    if Path(dirpath).resolve() != _checkpoint_dirpath(identity):
        raise RuntimeError("checkpoint identity mismatch: checkpoint_dirpath")

    return identity


def _durable_configuration(
    dirpath: Path,
) -> tuple[Identity, Path, Path, str] | None:
    opt_in = (
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    )
    if not any(os.environ.get(name) for name in opt_in):
        return None

    required = (
        "CAMPAIGN_ROOT",
        *opt_in,
    )
    values = {name: os.environ.get(name) for name in required}
    if any(not value for value in values.values()):
        raise RuntimeError("durable checkpoint settings must be configured together")

    campaign_root_value = os.environ["CAMPAIGN_ROOT"]
    manifest_value = os.environ["LAYOUTFORMERPP_LAUNCH_MANIFEST"]
    durable_root_value = os.environ["LAYOUTFORMERPP_DURABLE_ROOT"]
    track_value = os.environ["LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE"]
    identity = _validate_identity(dirpath)
    raw_campaign_root = identity["campaign_root"]
    if raw_campaign_root is None:
        raise ValueError("checkpoint identity has no campaign root")

    campaign_root = Path(raw_campaign_root)
    if not campaign_root.is_absolute():
        raise ValueError("checkpoint identity campaign root must be absolute")

    campaign_root = campaign_root.resolve()
    if Path(campaign_root_value).resolve() != campaign_root:
        raise RuntimeError("checkpoint identity mismatch: campaign_root")

    checkpoint_dirpath = _checkpoint_dirpath(identity)
    if not checkpoint_dirpath.is_relative_to(campaign_root):
        raise ValueError("checkpoint dirpath must be inside the campaign root")

    manifest = Path(manifest_value)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    manifest_identity = json.loads(manifest.read_text(encoding="utf-8"))
    expected_manifest = {
        key: value for key, value in identity.items() if key != "campaign_root"
    }
    if manifest_identity != expected_manifest:
        raise RuntimeError("launch manifest does not match checkpoint identity")

    durable_root = resolve_durable_root(
        durable_root_value,
        env_var="LAYOUTFORMERPP_DURABLE_ROOT",
    )
    track = Path(track_value)
    if track.is_absolute() or ".." in track.parts:
        raise ValueError("durable track must be relative to the durable root")

    return identity, manifest, durable_root, track.as_posix()


class LayoutFormerPPModelCheckpoint(ModelCheckpoint):
    """Save complete Lightning state at every epoch and preserve the best save."""

    def __init__(
        self,
        *,
        dirpath: str,
        monitor: str | None = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = False,
    ) -> None:
        """Initialize a callback with an owner-selected checkpoint directory."""
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            save_on_train_epoch_end=False,
        )
        self._checkpoint_directory = Path(dirpath)
        self._last_epoch_saved: int | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve local paths from the trainer root and validate durable setup."""
        if self._checkpoint_directory.is_absolute():
            self._checkpoint_directory = self._checkpoint_directory.resolve()
        else:
            self._checkpoint_directory = (
                Path(trainer.default_root_dir) / self._checkpoint_directory
            ).resolve()

        self.dirpath = str(self._checkpoint_directory)
        _durable_configuration(self._checkpoint_directory)
        super().setup(trainer, pl_module, stage)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, safe_value_type],
    ) -> None:
        """Capture RNG state and add launch identity when durable saving is enabled."""
        del trainer, pl_module
        configuration = _durable_configuration(self._checkpoint_directory)
        if configuration is not None:
            identity_state: dict[str, safe_value_type] = {
                key: value for key, value in configuration[0].items()
            }
            checkpoint["layoutformerpp_identity"] = identity_state

        checkpoint["layoutformerpp_rng"] = capture_rng_state()

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, safe_value_type],
    ) -> None:
        """Restore package RNG and validate launch identity before resume."""
        del trainer, pl_module
        configuration = _durable_configuration(self._checkpoint_directory)
        if configuration is None:
            if "layoutformerpp_identity" in checkpoint:
                raise RuntimeError(
                    "checkpoint identity cannot be validated without durable settings"
                )
        else:
            identity = checkpoint.get("layoutformerpp_identity")
            if not is_safe_mapping(identity):
                raise RuntimeError("checkpoint identity is missing or invalid")

            if identity != configuration[0]:
                raise RuntimeError("checkpoint identity mismatch")

        rng = checkpoint.get("layoutformerpp_rng")
        restore_rng_state(safe_string_mapping(rng, "checkpoint RNG state"))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a complete safe-to-load checkpoint after every epoch."""
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.sanity_checking or self._last_epoch_saved == trainer.current_epoch:
            return

        path = self._checkpoint_directory / (
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
        paths = sorted(self._checkpoint_directory.glob("epoch-*.ckpt"))
        for path in paths[:-2]:
            path.unlink()

    def _publish(self, path: Path, epoch: int, *, best: bool = False) -> None:
        configuration = _durable_configuration(self._checkpoint_directory)
        if configuration is None:
            return

        _, manifest, durable_root, track = configuration
        suffix = "-best" if best else ""
        relative = Path(track) / f"epoch-{epoch:06d}{suffix}"
        publish_checkpoint_bundle(
            {"checkpoint.ckpt": path}, manifest, durable_root, relative
        )
        retain_checkpoint_bundles(durable_root, track)


__all__ = ["LayoutFormerPPModelCheckpoint"]
