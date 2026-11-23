"""Shared validation and exact logging support for PubLayNet pre-checks."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


class Campaign(Protocol):
    """Launcher operations needed to validate pre-check provenance."""

    @property
    def REPOSITORY(self) -> Path: ...

    @property
    def CONFIG_SHA256(self) -> Mapping[str, str]: ...

    def config_path(self, family: str) -> Path: ...

    def current_commit(self) -> str: ...

    def require_clean_tree(self) -> None: ...

    def sha256(self, path: Path) -> str: ...

    def verify_splits(self, data_root: Path) -> dict[str, dict[str, int | str]]: ...


def capture_provenance(
    campaign: Campaign, family: str, data_root: Path
) -> dict[str, object]:
    """Validate and return the source, config, and processed-split identity."""
    campaign.require_clean_tree()
    config_path = campaign.config_path(family)
    config_hash = campaign.sha256(config_path)
    if config_hash != campaign.CONFIG_SHA256[family]:
        raise RuntimeError("checked-in family configuration hash mismatch")

    split_hashes = campaign.verify_splits(data_root)
    return {
        "source_commit": campaign.current_commit(),
        "clean_tree": True,
        "config": config_path.relative_to(campaign.REPOSITORY).as_posix(),
        "config_sha256": config_hash,
        "input_splits": split_hashes,
    }


class NativeObjectiveCapture:
    """Write exact scalar objectives at contiguous native logger indices."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.next_step = 0

    def record(self, data: object, step: object) -> None:
        if not isinstance(data, Mapping) or "training_loss" not in data:
            return
        if isinstance(step, bool) or not isinstance(step, int):
            raise RuntimeError("training objective has no native logger index")
        if step != self.next_step:
            raise RuntimeError(
                "training objectives lack contiguous native logger indices"
            )

        import torch

        value = cast(Mapping[str, object], data)["training_loss"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError("training objective is not a scalar")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("training objective is not a scalar")
        objective = float(value)
        if not math.isfinite(objective):
            raise RuntimeError("training objective is not finite")

        row = {
            "record": "update",
            "update": self.next_step + 1,
            "logger_step": step,
            "objective": objective,
        }
        mode = "x" if self.next_step == 0 else "a"
        with self.path.open(mode, encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.next_step += 1


def instrument_original_objective(wandb: ModuleType, path: Path) -> None:
    """Capture the unrounded training objective passed to the vendor logger."""
    recorder = NativeObjectiveCapture(path)
    original_log = getattr(wandb, "log", None)
    if not callable(original_log):
        raise RuntimeError("vendor logger does not expose a callable log function")
    typed_original_log = cast(Callable[..., object], original_log)

    def recording_log(*args: object, **kwargs: object) -> object:
        data = args[0] if args else kwargs.get("data")
        step = args[1] if len(args) > 1 else kwargs.get("step")
        recorder.record(data, step)
        return typed_original_log(*args, **kwargs)

    setattr(wandb, "log", recording_log)


def install_original_objective_capture() -> None:
    """Install the vendor logger hook when its output path is configured."""
    trace_value = os.environ.get("LAYOUTFORMERPP_OBJECTIVE_TRACE")
    if not trace_value:
        return

    import wandb

    instrument_original_objective(wandb, Path(trace_value))
