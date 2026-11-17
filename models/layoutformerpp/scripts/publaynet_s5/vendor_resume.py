"""Prepare the pinned LayoutFormer++ vendor tree with resumable state support."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import torch

from traingen.checkpointing import (
    PythonRandomState,
    capture_rng_state,
    is_safe_mapping,
    load_weights_only_checkpoint,
    make_weights_only_safe,
    publish_checkpoint_bundle,
    restore_rng_state,
    safe_input_type,
    safe_string_mapping,
    safe_value_type,
    save_weights_only_checkpoint,
)


class _AttributeContainer(Protocol):
    __dict__: dict[str, "VendorNode"]


VendorNode: TypeAlias = (
    safe_input_type
    | random.Random
    | torch.Generator
    | np.random.RandomState
    | dict[str | int, "VendorNode"]
    | list["VendorNode"]
    | tuple["VendorNode", ...]
    | _AttributeContainer
)


def _generator_states(value: VendorNode) -> dict[str, safe_value_type]:
    states: dict[str, safe_value_type] = {}
    seen: set[int] = set()

    def visit(item: VendorNode, path: str) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, random.Random):
            states[path] = item.getstate()
            return
        if isinstance(item, torch.Generator):
            states[path] = cast(safe_value_type, item.get_state())
            return
        if isinstance(item, np.random.RandomState):
            state = cast(
                tuple[str, np.ndarray, int, int, float],
                item.get_state(),
            )
            states[path] = {
                "algorithm": state[0],
                "keys": torch.as_tensor(state[1], dtype=torch.int64),
                "position": int(state[2]),
                "has_gauss": int(state[3]),
                "cached_gaussian": float(state[4]),
            }
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            for name, child in attributes.items():
                visit(child, f"{path}.{name}")

    visit(value, "root")
    return states


def _vendor_mapping(value: VendorNode) -> dict[str | int, VendorNode]:
    if not isinstance(value, dict):
        raise ValueError("generator state path does not name a mapping")

    return cast(dict[str | int, VendorNode], value)


def _resolve_path(root: VendorNode, path: str) -> VendorNode:
    value = root
    for component in path.removeprefix("root").split("."):
        if not component:
            continue
        while component:
            match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", component)
            if match is None:
                raise ValueError(f"invalid generator state path: {path}")
            name, index = match.groups()
            if isinstance(value, dict):
                value = _vendor_mapping(value)[name]
            else:
                value = getattr(value, name)
            if index is None:
                break
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"invalid generator state path: {path}")

            value = value[int(index)]
            component = ""
    return value


def _restore_generator_states(
    root: VendorNode, states: Mapping[str, safe_value_type]
) -> None:
    for path, state in states.items():
        generator = _resolve_path(root, path)
        if isinstance(generator, random.Random):
            generator.setstate(cast(PythonRandomState, state))
        elif isinstance(generator, torch.Generator):
            if not isinstance(state, torch.Tensor):
                raise TypeError("vendor generator state is not a tensor")

            generator.set_state(state)
        elif isinstance(generator, np.random.RandomState):
            if not is_safe_mapping(state) or not all(
                isinstance(key, str) for key in state
            ):
                raise TypeError("vendor NumPy generator state is not a mapping")

            mapping: dict[str, safe_value_type] = {
                key: value for key, value in state.items() if isinstance(key, str)
            }
            keys = mapping["keys"]
            if isinstance(keys, torch.Tensor):
                keys = keys.detach().cpu().numpy()
            position = mapping["position"]
            has_gauss = mapping["has_gauss"]
            cached_gaussian = mapping["cached_gaussian"]
            if not isinstance(position, (int, float)):
                raise TypeError("vendor NumPy generator position is not numeric")

            if not isinstance(has_gauss, (int, float)):
                raise TypeError("vendor NumPy generator flag is not numeric")

            if not isinstance(cached_gaussian, (int, float)):
                raise TypeError("vendor NumPy cached value is not numeric")

            generator.set_state(
                (
                    str(mapping["algorithm"]),
                    np.asarray(keys, dtype=np.uint32),
                    int(position),
                    int(has_gauss),
                    float(cached_gaussian),
                )
            )


def _identity() -> dict[str, safe_value_type]:
    raw = os.environ.get("LAYOUTFORMERPP_CHECKPOINT_IDENTITY")
    return {} if not raw else cast(dict[str, safe_value_type], json.loads(raw))


def _normalize_vendor_scalars(
    values: Mapping[str, safe_input_type],
) -> dict[str, safe_input_type]:
    def normalize(value: safe_input_type) -> safe_input_type:
        if isinstance(value, np.generic):
            scalar = value.item()
            if not isinstance(scalar, (bool, int, float, str)):
                raise TypeError(
                    "vendor NumPy scalar is not a supported checkpoint value"
                )

            return scalar

        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, safe_input_type], value)
            return {key: normalize(item) for key, item in mapping.items()}

        if isinstance(value, list):
            return [normalize(item) for item in value]

        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)

        return value

    return {key: normalize(value) for key, value in values.items()}


def _restore_layout_arrays(value: safe_value_type) -> safe_input_type:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, dict):
        mapping = safe_string_mapping(value, "vendor layout state")
        return {key: _restore_layout_arrays(item) for key, item in mapping.items()}

    if isinstance(value, list):
        return [_restore_layout_arrays(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_restore_layout_arrays(item) for item in value)

    return value


def _resume_directory(trainer: _AttributeContainer) -> Path:
    args = cast(_AttributeContainer, getattr(trainer, "args"))
    return Path(cast(str, getattr(args, "out_dir"))) / "resume"


def save_resume_state(
    trainer: _AttributeContainer,
    *,
    epoch: int,
    global_step: int,
    extras: Mapping[str, safe_value_type],
) -> Path:
    """Save one complete vendor epoch state and publish its durable copy."""
    model = cast(torch.nn.Module, getattr(trainer, "model"))
    optimizer = cast(torch.optim.Optimizer, getattr(trainer, "optimizer"))
    scheduler = getattr(trainer, "scheduler", None)
    train_loader = getattr(trainer, "train_dataloader", None)
    sampler = getattr(train_loader, "sampler", None)
    saved_extras = dict(extras)
    best_epoch_path = _resume_directory(trainer).parent / "best_epoch"
    if saved_extras.get("best_epoch") is None and best_epoch_path.is_file():
        saved_extras["best_epoch"] = int(best_epoch_path.read_text(encoding="utf-8"))
    payload = make_weights_only_safe(
        cast(
            Mapping[str, safe_input_type],
            {
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "step": global_step,
                "rng": capture_rng_state(),
                "sampler_epoch": getattr(sampler, "epoch", epoch),
                "dataset_generators": _generator_states(
                    {
                        "train": getattr(trainer, "train_dataset"),
                        "val": getattr(trainer, "val_dataset"),
                    }
                ),
                "identity": _identity(),
                "extras": make_weights_only_safe(
                    _normalize_vendor_scalars(
                        cast(Mapping[str, safe_input_type], saved_extras)
                    )
                ),
            },
        )
    )
    path = (
        _resume_directory(trainer)
        / f"epoch-{epoch:06d}-step-{global_step:09d}.resume.pt"
    )
    save_weights_only_checkpoint(path, payload)

    manifest = os.environ.get("LAYOUTFORMERPP_LAUNCH_MANIFEST")
    durable_root = os.environ.get("LAYOUTFORMERPP_DURABLE_ROOT")
    track_relative = os.environ.get("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE")
    if manifest and durable_root and track_relative:
        publish_checkpoint_bundle(
            {"checkpoint.pt": path},
            Path(manifest),
            Path(durable_root),
            Path(track_relative) / f"epoch-{epoch:06d}",
        )
    return path


def load_resume_state(trainer: _AttributeContainer) -> tuple[int, int]:
    """Load the selected full state and return its next epoch and global step."""
    resume_path = os.environ.get("LAYOUTFORMERPP_RESUME_PATH")
    if not resume_path:
        return 0, 0
    payload = load_weights_only_checkpoint(Path(resume_path))
    if os.environ.get("LAYOUTFORMERPP_CHECKPOINT_IDENTITY") and (
        payload.get("identity") != _identity()
    ):
        raise RuntimeError("vendor resume checkpoint identity mismatch")

    model = cast(torch.nn.Module, getattr(trainer, "model"))
    optimizer = cast(torch.optim.Optimizer, getattr(trainer, "optimizer"))
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise TypeError("vendor model state is not a tensor mapping")

    model_state = {
        name: value
        for name, value in model_payload.items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }
    if len(model_state) != len(model_payload):
        raise TypeError("vendor model state is not a tensor mapping")

    model.load_state_dict(model_state, strict=True)
    optimizer_payload = payload["optimizer"]
    if not is_safe_mapping(optimizer_payload) or not all(
        isinstance(key, str) for key in optimizer_payload
    ):
        raise TypeError("vendor optimizer state is not a mapping")

    optimizer_state: dict[str, safe_value_type] = {
        key: value for key, value in optimizer_payload.items() if isinstance(key, str)
    }
    optimizer.load_state_dict(optimizer_state)
    scheduler = getattr(trainer, "scheduler", None)
    scheduler_state = payload.get("scheduler")
    if (
        scheduler is not None
        and is_safe_mapping(scheduler_state)
        and all(isinstance(key, str) for key in scheduler_state)
    ):
        scheduler.load_state_dict(
            {key: value for key, value in scheduler_state.items()}
        )
    rng_state = payload["rng"]
    if not is_safe_mapping(rng_state) or not all(
        isinstance(key, str) for key in rng_state
    ):
        raise TypeError("vendor RNG state is not a mapping")

    rng_values: dict[str, safe_value_type] = {}
    for key, value in rng_state.items():
        if not isinstance(key, str):
            raise TypeError("vendor RNG state key is not a string")

        rng_values[key] = value

    restore_rng_state(rng_values)
    dataset_states = payload.get("dataset_generators")
    if is_safe_mapping(dataset_states) and all(
        isinstance(key, str) for key in dataset_states
    ):
        dataset_values: dict[str, safe_value_type] = {}
        for key, value in dataset_states.items():
            if not isinstance(key, str):
                raise TypeError("vendor dataset state key is not a string")

            dataset_values[key] = value

        _restore_generator_states(
            {
                "train": getattr(trainer, "train_dataset"),
                "val": getattr(trainer, "val_dataset"),
            },
            dataset_values,
        )
    sampler = getattr(getattr(trainer, "train_dataloader"), "sampler", None)
    if sampler is not None and hasattr(sampler, "epoch"):
        sampler_epoch = payload["sampler_epoch"]
        if not isinstance(sampler_epoch, (int, float)):
            raise TypeError("vendor sampler epoch is not numeric")

        setattr(sampler, "epoch", int(sampler_epoch))
    extras = payload.get("extras", {})
    if isinstance(extras, Mapping):
        measurement = extras.get("best_measurement")
        if measurement is not None:
            if not isinstance(measurement, (int, float)):
                raise TypeError("vendor best measurement is not numeric")

            setattr(trainer, "_resume_best_measurement", float(measurement))
        best_epoch = extras.get("best_epoch")
        if best_epoch is not None:
            if not isinstance(best_epoch, (int, float)):
                raise TypeError("vendor best epoch is not numeric")

            best_epoch_path = _resume_directory(trainer).parent / "best_epoch"
            best_epoch_path.write_text(f"{int(best_epoch)}\n", encoding="utf-8")
    restored_extras = dict(extras) if isinstance(extras, Mapping) else {}
    for name in ("gold_layouts", "task_gold_layouts"):
        layouts = restored_extras.get(name)
        if layouts is not None:
            restored_extras[name] = _restore_layout_arrays(layouts)

    setattr(trainer, "_resume_extras", restored_extras)
    next_epoch = payload["next_epoch"]
    step = payload["step"]
    if not isinstance(next_epoch, (int, float)) or not isinstance(step, (int, float)):
        raise TypeError("vendor resume counters are not numeric")

    return int(next_epoch), int(step)


def prepare_vendor_tree(vendor_source: Path, cache_root: Path) -> Path:
    """Copy the pinned source and apply only the resumable-training patch."""
    if not vendor_source.is_dir():
        raise FileNotFoundError(vendor_source)
    destination = cache_root / "vendor_run" / "layoutformerpp"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copytree(
            vendor_source,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _patch_vendor_sources(staging)
        if destination.is_symlink():
            raise RuntimeError("cached vendor tree is not a directory")

        if destination.exists():
            if not destination.is_dir():
                raise RuntimeError("cached vendor tree is not a directory")

            shutil.rmtree(destination)

        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return destination


def _replace_exactly_once(text: str, old: str, new: str, source: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"vendor patch anchor in {source} must occur exactly once; found {count}"
        )

    return text.replace(old, new, 1)


def _patch_vendor_sources(root: Path) -> None:
    resume_module = Path(__file__).with_name("vendor_resume.py")
    shutil.copy2(resume_module, root / "layoutformerpp_resume.py")
    basic = root / "trainer" / "basic_trainer.py"
    basic_text = basic.read_text(encoding="utf-8")
    basic_text = _replace_exactly_once(
        basic_text,
        "from .utils import CheckpointMeasurement\n",
        "from .utils import CheckpointMeasurement\nfrom layoutformerpp_resume import load_resume_state, save_resume_state\n",
        str(basic),
    )
    basic_text = _replace_exactly_once(
        basic_text,
        "        self._setup_dataloader()\n",
        "        self._setup_dataloader()\n        self._resume_start_epoch, self._resume_global_step = load_resume_state(self)\n",
        str(basic),
    )
    basic_text = _replace_exactly_once(
        basic_text,
        "        global_step = 0\n        for epoch in range(self.args.epoch):\n",
        "        resume_best_measurement = getattr(self, '_resume_best_measurement', None)\n        if resume_best_measurement is not None:\n            self.ckpt_measurement._best_value = resume_best_measurement\n        resume_extras = getattr(self, '_resume_extras', {})\n        gold_layouts = resume_extras.get('gold_layouts', []) if isinstance(resume_extras, dict) else []\n        global_step = self._resume_global_step\n        for epoch in range(self._resume_start_epoch, self.args.epoch):\n",
        str(basic),
    )
    basic_text = _replace_exactly_once(
        basic_text,
        "            self.do_checkpointing(epoch, is_best)\n",
        "            self.do_checkpointing(epoch, is_best)\n            save_resume_state(self, epoch=epoch, global_step=global_step, extras={\n                'best_measurement': self.ckpt_measurement.best_value,\n                'best_epoch': epoch if is_best else None,\n                'gold_layouts': gold_layouts,\n            })\n",
        str(basic),
    )
    basic.write_text(basic_text, encoding="utf-8")

    multitask = root / "trainer" / "multitask_trainer.py"
    multitask_text = multitask.read_text(encoding="utf-8")
    multitask_text = _replace_exactly_once(
        multitask_text,
        "from trainer.utils import CheckpointMeasurement\n",
        "from trainer.utils import CheckpointMeasurement\nfrom layoutformerpp_resume import load_resume_state, save_resume_state\n",
        str(multitask),
    )
    class_start = "class MultiTaskTrainer(Trainer):\n"
    class_end = "\nclass DSMultiTaskTrainer"
    if multitask_text.count(class_start) != 1 or multitask_text.count(class_end) != 1:
        raise RuntimeError(f"vendor patch class anchors in {multitask} are not unique")

    start = multitask_text.index(class_start)
    end = multitask_text.index(class_end, start)
    prefix, multitask_class, suffix = (
        multitask_text[:start],
        multitask_text[start:end],
        multitask_text[end:],
    )
    multitask_class = _replace_exactly_once(
        multitask_class,
        "        self.ckpt_measurement.reset()\n        global_step = 0\n        for epoch in range(self.args.epoch):\n",
        "        self.ckpt_measurement.reset()\n        resume_best_measurement = getattr(self, '_resume_best_measurement', None)\n        if resume_best_measurement is not None:\n            self.ckpt_measurement._best_value = resume_best_measurement\n        global_step = self._resume_global_step\n        resume_extras = getattr(self, '_resume_extras', {})\n        if isinstance(resume_extras, dict) and resume_extras.get('task_best_miou'):\n            task_best_miou = resume_extras['task_best_miou']\n        if isinstance(resume_extras, dict) and resume_extras.get('task_gold_layouts'):\n            task_gold_layouts = resume_extras['task_gold_layouts']\n        for epoch in range(self._resume_start_epoch, self.args.epoch):\n",
        str(multitask),
    )
    multitask_class = _replace_exactly_once(
        multitask_class,
        "\n    def update_best_miou(self, epoch, curr_miou, best_miou):\n",
        "\n            save_resume_state(self, epoch=epoch, global_step=global_step, extras={\n                'best_measurement': self.ckpt_measurement.best_value,\n                'best_epoch': None,\n                'task_best_miou': task_best_miou,\n                'task_gold_layouts': task_gold_layouts,\n            })\n\n    def update_best_miou(self, epoch, curr_miou, best_miou):\n",
        str(multitask),
    )
    multitask_text = prefix + multitask_class + suffix
    multitask.write_text(multitask_text, encoding="utf-8")
