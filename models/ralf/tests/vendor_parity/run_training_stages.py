"""Run fail-closed RALF S0-S4 training reproduction probes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
import torch.version
from jaxtyping import Shaped
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader
from traingen_parity.determinism import RNGState, capture_rng_state, restore_rng_state

from ralf import RalfConfig, RalfForConditionalLayoutGeneration
from ralf.retrieval import RalfRetrievedBatch
from ralf.training.datamodule import (
    RalfDataModule,
    RalfSampleValue,
    RalfTrainingBatch,
    collate_training_batch,
)
from ralf.training.lightning_module import RalfTrainingModule

from training_reference import (
    build_vendor_model,
    move_retrieved,
    named_optimizer_state,
    reseed,
    state_sha256,
    VendorTrainingModel,
    vendor_preprocess,
)

ROOT = Path(__file__).parents[4]
DEFAULT_STEPS = {"S1": 1, "S2": 1, "S3": 4, "S4": 8}

RalfRawSample = Mapping[str, RalfSampleValue | Shaped[Tensor, "..."]]


class _GradientCoverage(TypedDict):
    """Named-parameter gradient presence summary."""

    named_parameter_count: int
    present_count: int
    absent_count: int
    presence_digest: str


class _OptimizerGroup(TypedDict):
    """One optimizer group with stable parameter-name evidence."""

    lr: float
    weight_decay: float
    names: list[str]


class _GradientComparison(TypedDict):
    """Per-parameter gradient comparison result."""

    first_divergence: str | None
    max_abs_diff: float
    package: _GradientCoverage
    vendor: _GradientCoverage


class _TensorComparison(TypedDict):
    """Tensor-state comparison result."""

    first_divergence: str | None
    max_abs_diff: float


class _LearningRateGroup(TypedDict):
    """Learning-rate comparison for one optimizer group."""

    index: int
    package: float
    vendor: float
    abs_diff: float


class _LearningRateComparison(TypedDict):
    """Learning-rate comparison result."""

    first_divergence: int | None
    max_abs_diff: float
    groups: list[_LearningRateGroup]


class _TrainingContext(TypedDict):
    """Static package data used by the stage functions."""

    batch: RalfTrainingBatch
    samples: Sequence[RalfRawSample]
    table: Mapping[str | int, Sequence[int]]


class _NaturalEnvelope(TypedDict):
    """Run-to-run natural-trajectory envelope."""

    run_count: int
    step_count: int
    max_abs_diff: dict[str, dict[str, float]]
    first_package_state_hash_divergence_step: int | None
    package_state_hashes_equal: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("S0", "S1", "S2", "S3", "S4"), required=True
    )
    parser.add_argument("--dataset", choices=("cgl", "pku"), default="cgl")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _dataset_config(dataset: str, cache_dir: Path) -> RalfConfig:
    if dataset == "cgl":
        with (cache_dir / "dataset" / "cgl" / "vocabulary.json").open() as handle:
            vocabulary = json.load(handle)
        labels: dict[int, str] = {
            index: str(name) for index, name in enumerate(sorted(vocabulary["label"]))
        }
        fidnet_name = "cgl"
        dataset_name: Literal["cgl"] = "cgl"
    elif dataset == "pku":
        labels = {0: "logo", 1: "text", 2: "underlay"}
        fidnet_name = "pku10"
        dataset_name = "pku"
    else:
        raise ValueError(f"unsupported RALF dataset: {dataset}")
    precomputed = cache_dir / "PRECOMPUTED_WEIGHT_DIR"
    return RalfConfig(
        dataset_name=dataset_name,
        task="unconditional",
        id2label=cast(Mapping[int | str, str], labels),
        max_seq_length=10,
        num_bin=128,
        top_k=16,
        retrieval_backbone="dreamsim",
        resnet_weights_path=str(precomputed / "resnet50_a1_0-14fe96d1.pth"),
        fidnet_weights_path=str(
            precomputed / "fidnet" / fidnet_name / "model_best.pth.tar"
        ),
    )


def _retrieval_path(cache_dir: Path, dataset: str, split: str) -> Path:
    name = "cgl" if dataset == "cgl" else "pku"
    return (
        cache_dir
        / "PRECOMPUTED_WEIGHT_DIR"
        / "retrieval_indexes"
        / f"{name}_{split}_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt"
    )


def _load_context(
    args: argparse.Namespace,
) -> tuple[RalfConfig, RalfDataModule, _TrainingContext]:
    config = _dataset_config(args.dataset, args.cache_dir)
    train_index = _retrieval_path(args.cache_dir, args.dataset, "train")
    val_index = _retrieval_path(args.cache_dir, args.dataset, "val")
    for path in (
        train_index,
        val_index,
        Path(cast(str, config.resnet_weights_path)),
        Path(cast(str, config.fidnet_weights_path)),
    ):
        if not path.exists():
            raise FileNotFoundError(f"required RALF parity asset is missing: {path}")
    data = RalfDataModule(
        config=config,
        data_root=str(args.cache_dir / "dataset"),
        retrieval_index_path=str(train_index),
        validation_retrieval_index_path=str(val_index),
        batch_size=args.batch_size,
        num_workers=0,
    )
    data.setup("fit")
    if data.train_dataset is None or data.validation_dataset is None:
        raise RuntimeError("training and validation datasets were not initialized")
    samples = data.train_dataset.samples
    table_payload = torch.load(train_index, map_location="cpu", weights_only=False)
    table = cast(dict[str | int, Sequence[int]], dict(table_payload))
    batch = collate_training_batch(
        [
            data.train_dataset[index]
            for index in range(min(args.batch_size, len(data.train_dataset)))
        ]
    )
    return config, data, {"batch": batch, "samples": samples, "table": table}


def _recipe_epochs(dataset: str) -> int:
    """Read the selected member recipe instead of duplicating its epoch count."""
    import yaml

    recipe_path = ROOT / "models" / "ralf" / "configs" / "training" / f"{dataset}.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    if not isinstance(recipe, dict):
        raise RuntimeError(f"training recipe is not a mapping: {recipe_path}")
    trainer = recipe.get("trainer")
    if not isinstance(trainer, dict) or not isinstance(trainer.get("max_epochs"), int):
        raise RuntimeError(
            f"training recipe has no integer trainer.max_epochs: {recipe_path}"
        )
    return trainer["max_epochs"]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RALF training parity")
    return torch.device("cuda")


def _move_batch(batch: RalfTrainingBatch, device: torch.device) -> RalfTrainingBatch:
    return {
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": batch["pixel_values"].to(device),
        "saliency": batch["saliency"].to(device),
        "layout_labels": batch["layout_labels"].to(device),
        "layout_bbox": batch["layout_bbox"].to(device),
        "layout_mask": batch["layout_mask"].to(device),
        "retrieved": move_retrieved(batch["retrieved"], device),
    }


def _vendor_move(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _vendor_move(item, device) for key, item in value.items()}
    return value


def _models(
    config: RalfConfig, cache_dir: Path, device: torch.device, seed: int
) -> tuple[RalfTrainingModule, torch.nn.Module]:
    reseed(seed)
    package_model = RalfForConditionalLayoutGeneration(config)
    reseed(seed)
    vendor_model = build_vendor_model(config, cache_dir=cache_dir)
    package_model.load_state_dict(vendor_model.state_dict(), strict=True)
    package_model.to(device)
    vendor_model.to(device)
    package_module = RalfTrainingModule(
        config=config,
        model=package_model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        epochs=_recipe_epochs(str(config.dataset_name)),
        scheduler="multi_step",
        scheduler_milestones=(0.7,),
        condition_type="unconditional",
    )
    return package_module, vendor_model


def _loss_pair(
    package_module: RalfTrainingModule,
    vendor_model: torch.nn.Module,
    batch: RalfTrainingBatch,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    package_batch = _move_batch(batch, device)
    vendor_inputs, vendor_targets = vendor_preprocess(vendor_model, batch)
    vendor_inputs = cast(dict[str, object], _vendor_move(vendor_inputs, device))
    vendor_targets = cast(dict[str, object], _vendor_move(vendor_targets, device))
    package_model = package_module.model
    package_model.train()
    vendor_model.train()
    reseed(seed)
    package_output = package_model(
        input_ids=package_batch["input_ids"],
        labels=package_batch["labels"],
        attention_mask=package_batch["attention_mask"],
        pixel_values=package_batch["pixel_values"],
        saliency=package_batch["saliency"],
        retrieved=package_batch["retrieved"],
        condition_type="unconditional",
    )
    reseed(seed)
    vendor_training_model = cast(VendorTrainingModel, vendor_model)
    vendor_output, vendor_losses = vendor_training_model.train_loss(
        vendor_inputs, vendor_targets
    )
    vendor_logits = vendor_output["logits"]
    vendor_loss = vendor_losses["nll_loss"]
    if package_output.loss is None:
        raise RuntimeError("package model returned no loss")
    return package_output.loss, vendor_loss, package_output.logits, vendor_logits


def _max_abs(actual: Tensor, expected: Tensor) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    return float(
        (actual.detach().float() - expected.detach().float()).abs().max().item()
    )


def _gradient_presence_digest(gradients: Mapping[str, Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name in sorted(gradients):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"1" if gradients[name] is not None else b"0")
        digest.update(b"\0")
    return digest.hexdigest()


def _gradient_coverage(gradients: Mapping[str, Tensor | None]) -> _GradientCoverage:
    return {
        "named_parameter_count": len(gradients),
        "present_count": sum(gradient is not None for gradient in gradients.values()),
        "absent_count": sum(gradient is None for gradient in gradients.values()),
        "presence_digest": _gradient_presence_digest(gradients),
    }


def _compare_named_gradients(
    package_model: torch.nn.Module,
    vendor_model: torch.nn.Module,
    stage: str,
    *,
    enforce: bool = True,
) -> _GradientComparison:
    package_parameters = dict(package_model.named_parameters())
    vendor_parameters = dict(vendor_model.named_parameters())
    if set(package_parameters) != set(vendor_parameters):
        missing = sorted(set(vendor_parameters) - set(package_parameters))
        extra = sorted(set(package_parameters) - set(vendor_parameters))
        message = (
            f"first divergence at {stage}.coverage; missing={missing}; extra={extra}"
        )
        if enforce:
            raise RuntimeError(message)
        return {
            "first_divergence": "coverage",
            "max_abs_diff": float("inf"),
            "package": _gradient_coverage({name: None for name in package_parameters}),
            "vendor": _gradient_coverage({name: None for name in vendor_parameters}),
        }
    package_gradients = {
        name: parameter.grad.detach() if parameter.grad is not None else None
        for name, parameter in package_parameters.items()
    }
    vendor_gradients = {
        name: parameter.grad.detach() if parameter.grad is not None else None
        for name, parameter in vendor_parameters.items()
    }
    package_coverage = _gradient_coverage(package_gradients)
    vendor_coverage = _gradient_coverage(vendor_gradients)
    first_divergence: str | None = None
    aggregate_max_abs = 0.0
    for name in sorted(package_gradients):
        package_gradient = package_gradients[name]
        vendor_gradient = vendor_gradients[name]
        if (package_gradient is None) != (vendor_gradient is None):
            first_divergence = first_divergence or f"{name}.presence"
            aggregate_max_abs = float("inf")
            continue
        if package_gradient is None or vendor_gradient is None:
            continue
        aggregate_max_abs = max(
            aggregate_max_abs, _max_abs(package_gradient, vendor_gradient)
        )
        try:
            _assert_close(f"{stage}.{name}", package_gradient, vendor_gradient)
        except RuntimeError:
            first_divergence = first_divergence or name
    if first_divergence is not None and enforce:
        raise RuntimeError(
            f"first divergence at {stage}.{first_divergence}; "
            f"aggregate max_abs_diff={aggregate_max_abs:.8g}; "
            f"package_coverage={package_coverage}; vendor_coverage={vendor_coverage}"
        )
    return {
        "first_divergence": first_divergence,
        "max_abs_diff": aggregate_max_abs,
        "package": package_coverage,
        "vendor": vendor_coverage,
    }


def _assert_close(name: str, actual: Tensor, expected: Tensor) -> float:
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as exc:
        raise RuntimeError(
            f"first divergence at {name}: {_max_abs(actual, expected):.8g}; {exc}"
        ) from exc
    return _max_abs(actual, expected)


def _optimizer_for(
    module: RalfTrainingModule, vendor_model: torch.nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    package_optimizer = torch.optim.AdamW(module.optim_groups(), lr=1e-4, foreach=False)
    vendor_groups = cast(VendorTrainingModel, vendor_model).optim_groups(
        base_lr=1e-4,
        weight_decay=1e-4,
        custom_lr={"encoder.extractor.body": 1e-5},
    )
    vendor_optimizer = torch.optim.AdamW(vendor_groups, lr=1e-4, foreach=False)
    return package_optimizer, vendor_optimizer


def _group_names(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> list[_OptimizerGroup]:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "names": [by_id[id(parameter)] for parameter in group["params"]],
        }
        for group in optimizer.param_groups
    ]


def _compare_learning_rates(
    package_optimizer: torch.optim.Optimizer,
    vendor_optimizer: torch.optim.Optimizer,
    *,
    enforce: bool = True,
) -> _LearningRateComparison:
    package_groups = package_optimizer.param_groups
    vendor_groups = vendor_optimizer.param_groups
    if len(package_groups) != len(vendor_groups):
        raise RuntimeError(
            "first divergence at learning_rates.group_count; "
            f"package={len(package_groups)}; vendor={len(vendor_groups)}"
        )
    groups: list[_LearningRateGroup] = []
    first_divergence: int | None = None
    aggregate_max_abs = 0.0
    for index, (package_group, vendor_group) in enumerate(
        zip(package_groups, vendor_groups)
    ):
        package_lr = float(package_group["lr"])
        vendor_lr = float(vendor_group["lr"])
        abs_diff = abs(package_lr - vendor_lr)
        aggregate_max_abs = max(aggregate_max_abs, abs_diff)
        groups.append(
            {
                "index": index,
                "package": package_lr,
                "vendor": vendor_lr,
                "abs_diff": abs_diff,
            }
        )
        try:
            _assert_close(
                f"learning_rates.group[{index}].lr",
                torch.tensor(package_lr),
                torch.tensor(vendor_lr),
            )
        except RuntimeError:
            first_divergence = (
                first_divergence if first_divergence is not None else index
            )
    if first_divergence is not None and enforce:
        raise RuntimeError(
            f"first divergence at learning_rates.group[{first_divergence}].lr; "
            f"aggregate max_abs_diff={aggregate_max_abs:.8g}; groups={groups}"
        )
    return {
        "first_divergence": first_divergence,
        "max_abs_diff": aggregate_max_abs,
        "groups": groups,
    }


def _effective_config_digest(config: RalfConfig, cache_dir: Path) -> str:
    payload = config.to_dict()
    for key in ("resnet_weights_path", "fidnet_weights_path"):
        value = payload.get(key)
        if value is None:
            continue
        path = Path(str(value))
        try:
            payload[key] = path.relative_to(cache_dir).as_posix()
        except ValueError:
            payload[key] = path.name
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relevant_source_files() -> list[Path]:
    roots = (
        ROOT / "models/ralf/pyproject.toml",
        ROOT / "models/ralf/configs/training",
        ROOT / "models/ralf/src/ralf",
        ROOT / "models/ralf/tests/test_training.py",
        ROOT / "models/ralf/tests/vendor_parity/run_training_stages.py",
        ROOT / "models/ralf/tests/vendor_parity/test_training_harness.py",
        ROOT / "models/ralf/tests/vendor_parity/training_reference.py",
    )
    files: list[Path] = []
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if ".egg-info" in relative.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def _candidate_metadata() -> dict[str, object]:
    files = _relevant_source_files()
    digest = hashlib.sha256()
    relative_files: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        relative_files.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {
        "head_commit": head_commit,
        "worktree_dirty": dirty,
        "relevant_source_digest": digest.hexdigest(),
        "relevant_source_files": relative_files,
    }


def _vendor_metadata() -> dict[str, str]:
    status = subprocess.check_output(
        ["git", "submodule", "status", "--", "vendor/ralf"], cwd=ROOT, text=True
    ).strip()
    fields = status.split()
    if not fields:
        raise RuntimeError("pinned vendor submodule status is unavailable")
    return {
        "revision": fields[0].lstrip("+-"),
        "submodule_status": status,
    }


def _copy_batch_to_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, RalfRetrievedBatch):
        return RalfRetrievedBatch(
            image=value.image.detach().cpu().clone(),
            saliency=value.saliency.detach().cpu().clone(),
            bbox=value.bbox.detach().cpu().clone(),
            labels=value.labels.detach().cpu().clone(),
            mask=value.mask.detach().cpu().clone(),
            indexes=(
                None if value.indexes is None else value.indexes.detach().cpu().clone()
            ),
        )
    if isinstance(value, Mapping):
        return {key: _copy_batch_to_cpu(item) for key, item in value.items()}
    return value


def _serialized_sha256(value: object) -> str:
    """Hash a stream value with names, tensor metadata, and tensor bytes."""
    digest = hashlib.sha256()

    def update(item: object, path: str) -> None:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, RalfRetrievedBatch):
            update(
                {
                    "image": item.image,
                    "saliency": item.saliency,
                    "bbox": item.bbox,
                    "labels": item.labels,
                    "mask": item.mask,
                    "indexes": item.indexes,
                },
                f"{path}.retrieved",
            )
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            mapping = cast(Mapping[object, object], item)
            for key in sorted(mapping, key=str):
                update(mapping[key], f"{path}.{key}")
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            digest.update(b"sequence\0")
            for index, nested in enumerate(item):
                update(nested, f"{path}[{index}]")
            return
        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(repr(item).encode("utf-8"))

    update(value, "root")
    return digest.hexdigest()


def _gradient_l2_norm(model: torch.nn.Module) -> Tensor:
    total: Tensor | None = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().pow(2).sum()
        total = contribution if total is None else total + contribution
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total.sqrt()


def _compare_state_dicts(
    package_state: Mapping[str, Tensor],
    vendor_state: Mapping[str, Tensor],
    stage: str,
    *,
    enforce: bool = True,
) -> _TensorComparison:
    if set(package_state) != set(vendor_state):
        missing = sorted(set(vendor_state) - set(package_state))
        extra = sorted(set(package_state) - set(vendor_state))
        message = (
            f"first divergence at {stage}.coverage; missing={missing}; extra={extra}"
        )
        if enforce:
            raise RuntimeError(message)
        return {"first_divergence": "coverage", "max_abs_diff": float("inf")}
    aggregate_max_abs = 0.0
    first_divergence: str | None = None
    for name in sorted(package_state):
        aggregate_max_abs = max(
            aggregate_max_abs,
            _max_abs(package_state[name], vendor_state[name]),
        )
        try:
            _assert_close(f"{stage}.{name}", package_state[name], vendor_state[name])
        except RuntimeError:
            first_divergence = first_divergence or name
    if first_divergence is not None and enforce:
        raise RuntimeError(
            f"first divergence at {stage}.{first_divergence}; "
            f"aggregate max_abs_diff={aggregate_max_abs:.8g}"
        )
    return {
        "first_divergence": first_divergence,
        "max_abs_diff": aggregate_max_abs,
    }


def _compare_optimizer_states(
    package_optimizer: torch.optim.Optimizer,
    vendor_optimizer: torch.optim.Optimizer,
    package_model: torch.nn.Module,
    vendor_model: torch.nn.Module,
    stage: str,
    *,
    enforce: bool = True,
) -> _TensorComparison:
    package_state = named_optimizer_state(package_optimizer, package_model)
    vendor_state = named_optimizer_state(vendor_optimizer, vendor_model)
    if set(package_state) != set(vendor_state):
        missing = sorted(set(vendor_state) - set(package_state))
        extra = sorted(set(package_state) - set(vendor_state))
        message = (
            f"first divergence at {stage}.coverage; missing={missing}; extra={extra}"
        )
        if enforce:
            raise RuntimeError(message)
        return {"first_divergence": "coverage", "max_abs_diff": float("inf")}
    aggregate_max_abs = 0.0
    first_divergence: str | None = None
    for name in sorted(package_state):
        package_entries = package_state[name]
        vendor_entries = vendor_state[name]
        if set(package_entries) != set(vendor_entries):
            first_divergence = first_divergence or f"{name}.keys"
            continue
        for key in sorted(package_entries):
            aggregate_max_abs = max(
                aggregate_max_abs,
                _max_abs(package_entries[key], vendor_entries[key]),
            )
            try:
                _assert_close(
                    f"{stage}.{name}.{key}",
                    package_entries[key],
                    vendor_entries[key],
                )
            except RuntimeError:
                first_divergence = first_divergence or f"{name}.{key}"
    if first_divergence is not None and enforce:
        raise RuntimeError(
            f"first divergence at {stage}.{first_divergence}; "
            f"aggregate max_abs_diff={aggregate_max_abs:.8g}"
        )
    return {
        "first_divergence": first_divergence,
        "max_abs_diff": aggregate_max_abs,
    }


class RalfS3TraceCallback(Callback):
    """Compare a production Lightning run against the independent reference."""

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        seed: int | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("RALF_S3_CACHE_DIR", ""))
        self.seed = (
            seed if seed is not None else int(os.environ.get("RALF_S3_SEED", "1"))
        )
        self.output_dir = Path(output_dir or os.environ.get("RALF_S3_TRACE_DIR", ""))
        if not self.cache_dir or not self.output_dir:
            raise ValueError("RALF S3 callback paths must be configured")
        self.evidence_mode = os.environ.get("RALF_S3_MODE", "natural")
        if self.evidence_mode not in {"natural", "synchronized"}:
            raise ValueError("RALF_S3_MODE must be 'natural' or 'synchronized'")
        self.synchronized = self.evidence_mode == "synchronized"
        self.vendor_model: torch.nn.Module | None = None
        self.vendor_optimizer: torch.optim.Optimizer | None = None
        self.vendor_scheduler: torch.optim.lr_scheduler.MultiStepLR | None = None
        self.package_scheduler: torch.optim.lr_scheduler.MultiStepLR | None = None
        self.current_batch: RalfTrainingBatch | None = None
        self.current_rng: RNGState | None = None
        self.raw_results: list[_GradientComparison] = []
        self.clipped_result: _GradientComparison | None = None
        self.raw_norms: list[dict[str, float]] = []
        self.clipped_norms: list[dict[str, float]] = []
        self.train_trajectory: list[dict[str, object]] = []
        self.scheduler_trajectory: list[dict[str, object]] = []
        self.logging_trace: dict[str, list[dict[str, object]]] = {
            "train": [],
            "validation": [],
        }
        self.microbatch_count = 0
        self.optimizer_step_count = 0
        self.scheduler_step_epochs: set[int] = set()
        self.last_global_step = 0
        self.package_batch_lr: _LearningRateComparison | None = None
        self.package_batch_epoch = -1
        self.package_batch_index = -1
        self.package_loss: float | None = None
        self.vendor_loss: float | None = None
        self.vendor_rng_restored = False
        self.initial_state_sync: dict[str, object] = {}
        self.state_sync_records: list[dict[str, object]] = []
        self.first_divergence: dict[str, object] | None = None

    def _remember_divergence(self, location: str, max_abs_diff: float) -> None:
        if self.first_divergence is None:
            self.first_divergence = {
                "location": location,
                "max_abs_diff": max_abs_diff,
            }

    def _observe_tensor(self, location: str, actual: Tensor, expected: Tensor) -> float:
        try:
            return _assert_close(location, actual, expected)
        except RuntimeError:
            if self.synchronized:
                raise
            max_abs_diff = _max_abs(actual, expected)
            self._remember_divergence(location, max_abs_diff)
            return max_abs_diff

    def _require_models(self) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        if self.vendor_model is None or self.vendor_optimizer is None:
            raise RuntimeError("S3 reference models were not initialized")
        return self.vendor_model, self.vendor_optimizer

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if os.environ.get("PARITY_REQUIRE") != "1":
            raise RuntimeError("PARITY_REQUIRE=1 is required for S3")
        ralf_module = cast(RalfTrainingModule, pl_module)
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "first divergence at S3.deterministic_algorithms; "
                "PyTorch deterministic algorithms are not enabled"
            )
        if not torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError(
                "first divergence at S3.deterministic_warn_only; "
                "PyTorch deterministic algorithms are not configured for warning mode"
            )
        torch.cuda.reset_peak_memory_stats()
        package_rng = capture_rng_state()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reseed(self.seed)
        vendor_model = build_vendor_model(
            ralf_module.ralf_config, cache_dir=self.cache_dir
        )
        vendor_model.to(ralf_module.device)
        vendor_model.train()
        self.vendor_model = vendor_model
        _, self.vendor_optimizer = _optimizer_for(ralf_module, vendor_model)
        self.vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.vendor_optimizer,
            milestones=[
                int(value * ralf_module.epochs)
                for value in ralf_module.scheduler_milestones
            ],
            gamma=0.1,
        )
        restore_rng_state(package_rng)
        package_state = cast(Mapping[str, Tensor], ralf_module.model.state_dict())
        vendor_state = cast(Mapping[str, Tensor], vendor_model.state_dict())
        package_before_sync = state_sha256(package_state)
        vendor_initial_state = state_sha256(vendor_state)
        ralf_module.model.load_state_dict(vendor_state, strict=True)
        _compare_state_dicts(
            cast(Mapping[str, Tensor], ralf_module.model.state_dict()),
            vendor_state,
            "S3.initial_parameters",
        )
        self.initial_state_sync = {
            "package_state_sha256_before_sync": package_before_sync,
            "vendor_state_sha256": vendor_initial_state,
            "package_state_sha256_after_sync": state_sha256(
                ralf_module.model.state_dict()
            ),
            "package_model_object_preserved": True,
            "vendor_model_injected": False,
        }
        ralf_module._gradient_trace_hook = self

        optimizers = trainer.optimizers
        if len(optimizers) != 1:
            raise RuntimeError(
                f"first divergence at S3.optimizer_count; package={len(optimizers)}"
            )
        package_optimizer = optimizers[0]
        package_scheduler_configs = trainer.lr_scheduler_configs
        if len(package_scheduler_configs) != 1:
            raise RuntimeError(
                "first divergence at S3.scheduler_count; "
                f"package={len(package_scheduler_configs)}"
            )
        scheduler_config = package_scheduler_configs[0]
        if scheduler_config.interval != "epoch" or scheduler_config.frequency != 1:
            raise RuntimeError(
                "first divergence at S3.scheduler_cadence; "
                f"interval={scheduler_config.interval}; frequency={scheduler_config.frequency}"
            )
        if not isinstance(
            scheduler_config.scheduler, torch.optim.lr_scheduler.MultiStepLR
        ):
            raise RuntimeError(
                "first divergence at S3.scheduler_type; "
                f"package={type(scheduler_config.scheduler).__name__}"
            )
        self.package_scheduler = scheduler_config.scheduler
        if self.vendor_scheduler is None:
            raise RuntimeError("S3 vendor scheduler was not initialized")
        if self.package_scheduler.milestones != self.vendor_scheduler.milestones:
            raise RuntimeError(
                "first divergence at S3.scheduler_milestones; "
                f"package={self.package_scheduler.milestones}; "
                f"vendor={self.vendor_scheduler.milestones}"
            )
        _compare_learning_rates(
            package_optimizer,
            self.vendor_optimizer,
            enforce=self.synchronized,
        )
        self.scheduler_trajectory.append(
            {
                "epoch": 0,
                "package_last_epoch": self.package_scheduler.last_epoch,
                "vendor_last_epoch": self.vendor_scheduler.last_epoch,
                "package_lrs": [
                    float(group["lr"]) for group in package_optimizer.param_groups
                ],
                "vendor_lrs": [
                    float(group["lr"]) for group in self.vendor_optimizer.param_groups
                ],
            }
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        ralf_module = cast(RalfTrainingModule, pl_module)
        if self.vendor_optimizer is None:
            raise RuntimeError("S3 reference optimizer was not initialized")
        package_optimizer = trainer.optimizers[0]
        if self.synchronized and self.optimizer_step_count:
            self.vendor_model = self.vendor_model or self._require_models()[0]
            self.vendor_model.load_state_dict(
                ralf_module.model.state_dict(), strict=True
            )
            self.vendor_optimizer.load_state_dict(
                copy.deepcopy(package_optimizer.state_dict())
            )
            if self.package_scheduler is not None and self.vendor_scheduler is not None:
                self.vendor_scheduler.load_state_dict(
                    self.package_scheduler.state_dict()
                )
        package_batch_lr = _compare_learning_rates(
            package_optimizer,
            self.vendor_optimizer,
            enforce=self.synchronized,
        )
        self.package_batch_lr = package_batch_lr
        if package_batch_lr["first_divergence"] is not None:
            self._remember_divergence(
                f"S3.epoch[{trainer.current_epoch}].batch[{batch_idx}].learning_rates",
                package_batch_lr["max_abs_diff"],
            )
        if self.package_scheduler is None or self.vendor_scheduler is None:
            raise RuntimeError("S3 schedulers were not initialized")
        self.scheduler_trajectory.append(
            {
                "epoch": trainer.current_epoch,
                "package_last_epoch": self.package_scheduler.last_epoch,
                "vendor_last_epoch": self.vendor_scheduler.last_epoch,
                "package_lrs": [
                    float(group["lr"]) for group in package_optimizer.param_groups
                ],
                "vendor_lrs": [
                    float(group["lr"]) for group in self.vendor_optimizer.param_groups
                ],
            }
        )
        self.current_batch = cast(RalfTrainingBatch, _copy_batch_to_cpu(batch))
        self.current_rng = capture_rng_state()
        self.package_batch_epoch = trainer.current_epoch
        self.package_batch_index = batch_idx
        self.microbatch_count += 1
        self.raw_results = []
        self.clipped_result = None
        self.raw_norms = []
        self.clipped_norms = []
        self.vendor_rng_restored = False

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ralf_module = cast(RalfTrainingModule, pl_module)
        vendor_model, _ = self._require_models()
        if self.current_batch is None or self.current_rng is None:
            raise RuntimeError("S3 backward hook has no current production batch")
        package_rng_after_backward = capture_rng_state()
        restore_rng_state(self.current_rng)
        self.vendor_rng_restored = True
        vendor_inputs, vendor_targets = vendor_preprocess(
            vendor_model, self.current_batch
        )
        vendor_inputs = cast(
            dict[str, object], _vendor_move(vendor_inputs, ralf_module.device)
        )
        vendor_targets = cast(
            dict[str, object], _vendor_move(vendor_targets, ralf_module.device)
        )
        vendor_model.train()
        vendor_training_model = cast(VendorTrainingModel, vendor_model)
        vendor_output, vendor_losses = vendor_training_model.train_loss(
            vendor_inputs, vendor_targets
        )
        vendor_logits = vendor_output["logits"]
        vendor_loss = vendor_losses["nll_loss"]
        accumulation = trainer.accumulate_grad_batches
        (vendor_loss / accumulation).backward()
        restore_rng_state(package_rng_after_backward)
        package_trace = ralf_module.latest_step_trace
        package_loss = package_trace["train_loss"]
        package_logits = package_trace["logits"]
        self._observe_tensor(
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].loss",
            package_loss,
            vendor_loss,
        )
        self._observe_tensor(
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].logits",
            package_logits,
            vendor_logits,
        )
        raw_result = _compare_named_gradients(
            ralf_module.model,
            vendor_model,
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].raw_gradients",
            enforce=self.synchronized,
        )
        if raw_result["first_divergence"] is not None:
            self._remember_divergence(
                "S3.epoch[{}].batch[{}].raw_gradients.{}".format(
                    self.package_batch_epoch,
                    self.package_batch_index,
                    raw_result["first_divergence"],
                ),
                float(raw_result["max_abs_diff"]),
            )
        package_norm = _gradient_l2_norm(ralf_module.model)
        vendor_norm = _gradient_l2_norm(vendor_model)
        self._observe_tensor(
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].raw_gradient_norm",
            package_norm,
            vendor_norm,
        )
        self.raw_results.append(raw_result)
        self.raw_norms.append(
            {
                "package": float(package_norm.detach().cpu()),
                "vendor": float(vendor_norm.detach().cpu()),
                "max_abs_diff": _max_abs(package_norm, vendor_norm),
            }
        )
        self.package_loss = float(package_loss.detach().cpu())
        self.vendor_loss = float(vendor_loss.detach().cpu())

    def on_package_gradients_clipped(self, pl_module: RalfTrainingModule) -> None:
        vendor_model, _ = self._require_models()
        package_norm = _gradient_l2_norm(pl_module.model)
        torch.nn.utils.clip_grad_norm_(
            vendor_model.parameters(), pl_module.clip_max_norm
        )
        clipped_result = _compare_named_gradients(
            pl_module.model,
            vendor_model,
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].clipped_gradients",
            enforce=self.synchronized,
        )
        if clipped_result["first_divergence"] is not None:
            self._remember_divergence(
                "S3.epoch[{}].batch[{}].clipped_gradients.{}".format(
                    self.package_batch_epoch,
                    self.package_batch_index,
                    clipped_result["first_divergence"],
                ),
                float(clipped_result["max_abs_diff"]),
            )
        vendor_norm = _gradient_l2_norm(vendor_model)
        self._observe_tensor(
            f"S3.epoch[{self.package_batch_epoch}].batch[{self.package_batch_index}].clipped_gradient_norm",
            package_norm,
            vendor_norm,
        )
        self.clipped_result = clipped_result
        self.clipped_norms.append(
            {
                "package": float(package_norm.detach().cpu()),
                "vendor": float(vendor_norm.detach().cpu()),
                "max_abs_diff": _max_abs(package_norm, vendor_norm),
            }
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs, batch
        ralf_module = cast(RalfTrainingModule, pl_module)
        vendor_model, vendor_optimizer = self._require_models()
        if self.clipped_result is None:
            return
        if self.package_batch_lr is None:
            raise RuntimeError(
                f"first divergence at S3.batch[{batch_idx}].optimizer_step_hooks"
            )
        optimizer_step_index = len(self.train_trajectory) + 1
        trainer_global_step = trainer.global_step
        vendor_optimizer.step()
        parameter_result = _compare_state_dicts(
            cast(Mapping[str, Tensor], ralf_module.model.state_dict()),
            cast(Mapping[str, Tensor], vendor_model.state_dict()),
            f"S3.global_step[{optimizer_step_index}].parameters",
            enforce=self.synchronized,
        )
        if parameter_result["first_divergence"] is not None:
            self._remember_divergence(
                "S3.global_step[{}].parameters.{}".format(
                    optimizer_step_index, parameter_result["first_divergence"]
                ),
                float(parameter_result["max_abs_diff"]),
            )
        optimizer_result = _compare_optimizer_states(
            trainer.optimizers[0],
            vendor_optimizer,
            ralf_module.model,
            vendor_model,
            f"S3.global_step[{optimizer_step_index}].optimizer_state",
            enforce=self.synchronized,
        )
        if optimizer_result["first_divergence"] is not None:
            self._remember_divergence(
                "S3.global_step[{}].optimizer_state.{}".format(
                    optimizer_step_index, optimizer_result["first_divergence"]
                ),
                float(optimizer_result["max_abs_diff"]),
            )
        package_state_sha256 = state_sha256(ralf_module.model.state_dict())
        vendor_state_sha256 = state_sha256(vendor_model.state_dict())
        sync_result: dict[str, object] | None = None
        if self.synchronized:
            package_optimizer = trainer.optimizers[0]
            vendor_model.load_state_dict(ralf_module.model.state_dict(), strict=True)
            vendor_optimizer.load_state_dict(
                copy.deepcopy(package_optimizer.state_dict())
            )
            sync_result = {
                "parameters": _compare_state_dicts(
                    cast(Mapping[str, Tensor], ralf_module.model.state_dict()),
                    cast(Mapping[str, Tensor], vendor_model.state_dict()),
                    f"S3.global_step[{optimizer_step_index}].state_sync.parameters",
                ),
                "optimizer_state": _compare_optimizer_states(
                    package_optimizer,
                    vendor_optimizer,
                    ralf_module.model,
                    vendor_model,
                    f"S3.global_step[{optimizer_step_index}].state_sync.optimizer_state",
                ),
                "package_state_sha256": state_sha256(ralf_module.model.state_dict()),
                "vendor_state_sha256": state_sha256(vendor_model.state_dict()),
            }
            self.state_sync_records.append(sync_result)
        self.train_trajectory.append(
            {
                "evidence_mode": self.evidence_mode,
                "epoch": self.package_batch_epoch,
                "batch_idx": batch_idx,
                "global_step": optimizer_step_index,
                "trainer_global_step_at_batch_end": trainer_global_step,
                "loss": {"package": self.package_loss, "vendor": self.vendor_loss},
                "learning_rates": self.package_batch_lr,
                "raw_gradients": self.raw_results,
                "clipped_gradients": self.clipped_result,
                "raw_gradient_norm": self.raw_norms,
                "clipped_gradient_norm": self.clipped_norms,
                "parameters": parameter_result,
                "optimizer_state": optimizer_result,
                "package_state_sha256": package_state_sha256,
                "vendor_state_sha256": vendor_state_sha256,
            }
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "s3-partial-trace.json").write_text(
            json.dumps(
                {
                    "status": "IN_PROGRESS",
                    "optimizer_step_count": len(self.train_trajectory),
                    "global_step": optimizer_step_index,
                    "trainer_global_step_at_batch_end": trainer_global_step,
                    "trajectory": self.train_trajectory,
                    "state_sync": self.state_sync_records,
                    "first_divergence": self.first_divergence,
                    "peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        vendor_optimizer.zero_grad(set_to_none=True)
        self.optimizer_step_count = optimizer_step_index

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        self.logging_trace["train"].append(
            {"epoch": epoch, "train_loss": "train_loss" in metrics}
        )
        if epoch in self.scheduler_step_epochs:
            return
        if self.vendor_scheduler is None:
            raise RuntimeError("S3 vendor scheduler was not initialized")
        self.vendor_scheduler.step()
        self.scheduler_step_epochs.add(epoch)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        del pl_module
        self.logging_trace["validation"].append(
            {
                "epoch": trainer.current_epoch,
                "val_loss": "val_loss" in trainer.callback_metrics,
            }
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self.vendor_scheduler is None or self.package_scheduler is None:
            raise RuntimeError("S3 schedulers were not initialized")
        if self.package_scheduler.last_epoch != self.vendor_scheduler.last_epoch:
            message = (
                "first divergence at S3.final_scheduler.last_epoch; "
                f"package={self.package_scheduler.last_epoch}; "
                f"vendor={self.vendor_scheduler.last_epoch}"
            )
            if self.synchronized:
                raise RuntimeError(message)
            self._remember_divergence(
                "S3.final_scheduler.last_epoch",
                float(
                    abs(
                        self.package_scheduler.last_epoch
                        - self.vendor_scheduler.last_epoch
                    )
                ),
            )
        if self.vendor_optimizer is None:
            raise RuntimeError("S3 vendor optimizer was not initialized")
        final_learning_rates = _compare_learning_rates(
            trainer.optimizers[0],
            self.vendor_optimizer,
            enforce=self.synchronized,
        )
        if final_learning_rates["first_divergence"] is not None:
            self._remember_divergence(
                "S3.final_learning_rates.group[{}].lr".format(
                    final_learning_rates["first_divergence"]
                ),
                float(final_learning_rates["max_abs_diff"]),
            )
        if trainer.global_step != len(self.train_trajectory):
            raise RuntimeError(
                "first divergence at S3.global_step_count; "
                f"trainer={trainer.global_step}; "
                f"trace={len(self.train_trajectory)}"
            )
        checkpoints = [
            callback
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        if len(checkpoints) != 1:
            raise RuntimeError(
                f"first divergence at S3.checkpoint_callback_count; count={len(checkpoints)}"
            )
        checkpoint = checkpoints[0]
        best_model_path = checkpoint.best_model_path
        last_model_path = checkpoint.last_model_path
        if best_model_path is None or last_model_path is None:
            raise RuntimeError(
                "first divergence at S3.checkpoint_paths; paths are unset"
            )
        best_path = Path(best_model_path)
        last_path = Path(last_model_path)
        if not best_path.is_file() or not last_path.is_file():
            raise RuntimeError(
                "first divergence at S3.checkpoint_files; "
                f"best={best_path}; last={last_path}"
            )
        metrics_paths: list[str] = []
        for logger in trainer.loggers:
            log_dir_value = logger.log_dir
            if log_dir_value is None:
                raise RuntimeError(
                    "first divergence at S3.logging.log_dir; path is unset"
                )
            log_dir = Path(log_dir_value)
            metrics_path = log_dir / "metrics.csv"
            if not metrics_path.is_file():
                raise RuntimeError(
                    f"first divergence at S3.logging.metrics_file; path={metrics_path}"
                )
            metrics_paths.append(metrics_path.as_posix())
        max_epochs = trainer.max_epochs
        if max_epochs is None:
            raise RuntimeError("first divergence at S3.max_epochs; value is unset")
        result = {
            "status": "PASS" if self.synchronized else "RECORDED",
            "evidence_mode": self.evidence_mode,
            "production_model": "RalfTrainingModule",
            "production_datamodule": "RalfDataModule",
            "initial_state_sync": self.initial_state_sync,
            "epochs": max_epochs,
            "train_batches": trainer.num_training_batches * max_epochs,
            "validation_epochs": len(self.logging_trace["validation"]),
            "microbatch_count": self.microbatch_count,
            "optimizer_step_count": len(self.train_trajectory),
            "global_step": trainer.global_step,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "trainer_deterministic_mode": "warn",
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "determinism_note": (
                "Lightning Trainer deterministic='warn' and PyTorch deterministic algorithms "
                "enabled with warn_only; fixed seeds and per-batch RNG restoration remain "
                "enabled. cuDNN deterministic kernels, benchmark off, and TF32 disabled."
            ),
            "scheduler": {
                "interval": trainer.lr_scheduler_configs[0].interval,
                "frequency": trainer.lr_scheduler_configs[0].frequency,
                "milestones": sorted(
                    int(item) for item in self.package_scheduler.milestones
                ),
                "last_epoch": self.package_scheduler.last_epoch,
                "trajectory": self.scheduler_trajectory,
            },
            "final_learning_rates": final_learning_rates,
            "logging": self.logging_trace,
            "state_synchronized_lockstep": self.state_sync_records,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoint": {
                "best_model_path": best_path.as_posix(),
                "last_model_path": last_path.as_posix(),
                "best_model_score": (
                    None
                    if checkpoint.best_model_score is None
                    else float(checkpoint.best_model_score.detach().cpu())
                ),
            },
            "logger_metrics": metrics_paths,
            "trajectory": self.train_trajectory,
            "first_divergence": self.first_divergence,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "s3-trace.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )


class RalfS3ModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint configured only through S3 environment metadata."""

    def __init__(self) -> None:
        super().__init__(
            dirpath=os.environ.get("RALF_S3_CHECKPOINT_DIR"),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="epoch={epoch:02d}-val_loss={val_loss:.6f}",
        )


class RalfS3CSVLogger(CSVLogger):
    """Construct the production CSV logger from the S3 run environment."""

    def __init__(self) -> None:
        super().__init__(
            save_dir=os.environ["RALF_S3_LOGGER_DIR"],
            name="csv",
            version=None,
        )


def _fresh_s3_run_root(base_root: Path) -> Path:
    """Return the first unused run directory without touching prior evidence."""
    runs_root = base_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = runs_root / f"run-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no unused S3 run directory under {runs_root}")


def _s3_child_import_gate() -> None:
    """Resolve every class path passed to the spawned production trainer."""
    callback_root = ROOT / "models" / "ralf" / "tests" / "vendor_parity"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(callback_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    paths = (
        "run_training_stages.RalfS3TraceCallback",
        "run_training_stages.RalfS3ModelCheckpoint",
        "run_training_stages.RalfS3CSVLogger",
    )
    probe = (
        "from importlib import import_module\n"
        f"paths = {paths!r}\n"
        "for path in paths:\n"
        "    module_name, attribute_name = path.rsplit('.', 1)\n"
        "    getattr(import_module(module_name), attribute_name)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "S3 child import gate failed before the production trainer: "
            f"{completed.stdout.strip()}"
        )


def _s0(
    args: argparse.Namespace,
    config: RalfConfig,
    data: RalfDataModule,
    context: _TrainingContext,
    device: torch.device,
) -> dict[str, object]:
    package_module, vendor_model = _models(config, args.cache_dir, device, args.seed)
    package_optimizer, vendor_optimizer = _optimizer_for(package_module, vendor_model)
    package_groups = _group_names(package_optimizer, package_module.model)
    vendor_groups = _group_names(vendor_optimizer, vendor_model)
    if package_groups != vendor_groups:
        raise RuntimeError("S0 optimizer parameter groups differ")
    package_state = package_module.model.state_dict()
    vendor_state = vendor_model.state_dict()
    if set(package_state) != set(vendor_state):
        raise RuntimeError("S0 state-dict key coverage differs")
    for key in package_state:
        if not torch.equal(package_state[key].cpu(), vendor_state[key].cpu()):
            raise RuntimeError(f"S0 copied initialized state differs at {key}")
    first_id = data.train_dataset.samples[0].get("id") if data.train_dataset else None
    return {
        "status": "PASS",
        "parameter_count": sum(
            parameter.numel() for parameter in package_module.model.parameters()
        ),
        "state_dict_keys": len(package_state),
        "state_sha256": state_sha256(package_state),
        "optimizer_groups": {"package": package_groups, "vendor": vendor_groups},
        "scheduler_milestone_epoch": int(
            package_module.scheduler_milestones[0] * package_module.epochs
        ),
        "tokenizer": {
            "vocab_size": config.vocab_size,
            "max_token_length": config.max_token_length,
            "pad_token_id": config.pad_token_id,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
        },
        "dataset": {
            "train_count": len(data.train_dataset) if data.train_dataset else None,
            "validation_count": len(data.validation_dataset)
            if data.validation_dataset
            else None,
            "first_train_id": first_id,
            "seed": args.seed,
        },
        "package_model_type": type(package_module.model).__name__,
        "vendor_model_type": type(vendor_model).__name__,
        "batch_keys": sorted(context["batch"]),
    }


def _s1(
    args: argparse.Namespace,
    config: RalfConfig,
    context: _TrainingContext,
    device: torch.device,
) -> dict[str, object]:
    package_module, vendor_model = _models(config, args.cache_dir, device, args.seed)
    batch = context["batch"]
    vendor_inputs, vendor_targets = vendor_preprocess(vendor_model, batch)
    package_ids = batch["input_ids"]
    package_labels = batch["labels"]
    vendor_ids = cast(Tensor, vendor_inputs["seq"])
    vendor_labels = cast(Tensor, vendor_targets["seq"])
    input_diff = _assert_close("prepared.input_ids", package_ids, vendor_ids)
    label_diff = _assert_close("prepared.labels", package_labels, vendor_labels)
    package_loss, vendor_loss, package_logits, vendor_logits = _loss_pair(
        package_module, vendor_model, batch, device, args.seed
    )
    loss_diff = _assert_close("train_loss", package_loss, vendor_loss)
    logit_diff = _assert_close("logits", package_logits, vendor_logits)
    return {
        "status": "PASS",
        "batch_size": package_ids.size(0),
        "input_shape": list(package_ids.shape),
        "max_abs_diff": {
            "prepared_input_ids": input_diff,
            "prepared_labels": label_diff,
            "loss": loss_diff,
            "logits": logit_diff,
        },
        "seed": args.seed,
    }


def _s2(
    args: argparse.Namespace,
    config: RalfConfig,
    context: _TrainingContext,
    device: torch.device,
) -> dict[str, object]:
    package_module, vendor_model = _models(config, args.cache_dir, device, args.seed)
    package_optimizer, vendor_optimizer = _optimizer_for(package_module, vendor_model)
    learning_rates = _compare_learning_rates(package_optimizer, vendor_optimizer)
    batch = context["batch"]
    package_loss, vendor_loss, package_logits, vendor_logits = _loss_pair(
        package_module, vendor_model, batch, device, args.seed
    )
    _assert_close("train_loss", package_loss, vendor_loss)
    _assert_close("logits", package_logits, vendor_logits)
    package_loss.backward()
    vendor_loss.backward()
    raw_gradients = _compare_named_gradients(
        package_module.model, vendor_model, "raw_gradients"
    )
    package_norm = torch.nn.utils.clip_grad_norm_(
        package_module.model.parameters(), 0.1
    )
    vendor_norm = torch.nn.utils.clip_grad_norm_(vendor_model.parameters(), 0.1)
    _assert_close("clipped_gradient_norm", package_norm, vendor_norm)
    clipped_gradients = _compare_named_gradients(
        package_module.model, vendor_model, "clipped_gradients"
    )
    package_optimizer.step()
    vendor_optimizer.step()
    package_state = package_module.model.state_dict()
    vendor_state = vendor_model.state_dict()
    parameter_diff = max(
        _max_abs(package_state[key], vendor_state[key]) for key in package_state
    )
    if parameter_diff > 1e-5:
        raise RuntimeError(
            f"first divergence at post_step.parameter: {parameter_diff:.8g}"
        )
    package_opt_state = named_optimizer_state(package_optimizer, package_module.model)
    vendor_opt_state = named_optimizer_state(vendor_optimizer, vendor_model)
    if package_opt_state.keys() != vendor_opt_state.keys():
        raise RuntimeError("S2 optimizer state parameter coverage differs")
    optimizer_diff = 0.0
    for name in package_opt_state:
        for key in package_opt_state[name]:
            optimizer_diff = max(
                optimizer_diff,
                _max_abs(package_opt_state[name][key], vendor_opt_state[name][key]),
            )
    if optimizer_diff > 1e-5:
        raise RuntimeError(f"first divergence at optimizer_state: {optimizer_diff:.8g}")
    gradient_norm_diff = _max_abs(package_norm, vendor_norm)
    return {
        "status": "PASS",
        "loss": float(package_loss.detach().cpu()),
        "gradient_norm": {
            "package": float(package_norm.detach().cpu()),
            "vendor": float(vendor_norm.detach().cpu()),
            "max_abs_diff": gradient_norm_diff,
        },
        "gradients": {"raw": raw_gradients, "clipped": clipped_gradients},
        "max_abs_diff": {
            "post_step_parameters": parameter_diff,
            "optimizer_state": optimizer_diff,
        },
        "learning_rates": learning_rates,
        "seed": args.seed,
    }


def _run_s3_fit(
    args: argparse.Namespace,
    run_base: Path,
    mode: str,
    train_limit: float,
    validation_limit: float,
) -> dict[str, object]:
    run_root = _fresh_s3_run_root(run_base)
    trace_root = run_root / "trace"
    checkpoint_root = run_root / "checkpoints"
    logger_root = run_root / "logger"
    resnet_path = (
        args.cache_dir / "PRECOMPUTED_WEIGHT_DIR" / "resnet50_a1_0-14fe96d1.pth"
    )
    fidnet_path = (
        args.cache_dir
        / "PRECOMPUTED_WEIGHT_DIR"
        / "fidnet"
        / ("cgl" if args.dataset == "cgl" else "pku10")
        / "model_best.pth.tar"
    )
    train_index_path = _retrieval_path(args.cache_dir, args.dataset, "train")
    validation_index_path = _retrieval_path(args.cache_dir, args.dataset, "val")
    callback_root = ROOT / "models" / "ralf" / "tests" / "vendor_parity"
    for path in (trace_root, checkpoint_root, logger_root):
        path.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--active",
        "--no-sync",
        "--package",
        "ralf",
        "--extra",
        "training",
        "--extra",
        "vendor",
        "traingen",
        "fit",
        "--config",
        f"models/ralf/configs/training/{args.dataset}.yaml",
        f"--seed_everything={args.seed}",
        "--trainer.accelerator=gpu",
        "--trainer.devices=1",
        f"--trainer.max_epochs={_recipe_epochs(args.dataset)}",
        f"--trainer.limit_train_batches={train_limit}",
        f"--trainer.limit_val_batches={validation_limit}",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.check_val_every_n_epoch=1",
        "--trainer.accumulate_grad_batches=1",
        "--trainer.deterministic=warn",
        f"--data.init_args.batch_size={args.batch_size}",
        f"--trainer.default_root_dir={run_root}",
        "--trainer.enable_progress_bar=false",
        "--trainer.enable_model_summary=false",
        f"--model.init_args.config.init_args.resnet_weights_path={resnet_path}",
        f"--model.init_args.config.init_args.fidnet_weights_path={fidnet_path}",
        f"--data.init_args.config.init_args.resnet_weights_path={resnet_path}",
        f"--data.init_args.config.init_args.fidnet_weights_path={fidnet_path}",
        f"--data.init_args.data_root={args.cache_dir / 'dataset'}",
        f"--data.init_args.retrieval_index_path={train_index_path}",
        f"--data.init_args.validation_retrieval_index_path={validation_index_path}",
        "--trainer.callbacks=[run_training_stages.RalfS3TraceCallback,run_training_stages.RalfS3ModelCheckpoint]",
        "--trainer.logger=run_training_stages.RalfS3CSVLogger",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(callback_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["RALF_RESNET_WEIGHTS_PATH"] = str(resnet_path)
    env["RALF_FIDNET_WEIGHTS_PATH"] = str(fidnet_path)
    env["RALF_DATA_ROOT"] = str(args.cache_dir / "dataset")
    env["RALF_RETRIEVAL_INDEX_PATH"] = str(train_index_path)
    env["RALF_VALIDATION_RETRIEVAL_INDEX_PATH"] = str(validation_index_path)
    env["RALF_S3_CACHE_DIR"] = str(args.cache_dir)
    env["RALF_S3_SEED"] = str(args.seed)
    env["RALF_S3_MODE"] = mode
    env["RALF_S3_TRACE_DIR"] = str(trace_root)
    env["RALF_S3_CHECKPOINT_DIR"] = str(checkpoint_root)
    env["RALF_S3_LOGGER_DIR"] = str(logger_root)
    stdout_path = run_root / "traingen.stdout.log"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    runtime_seconds = time.monotonic() - started
    stdout_path.write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            "S3 documented traingen fit failed with exit "
            f"{completed.returncode}; first failure output:\n{tail}"
        )
    trace_path = trace_root / "s3-trace.json"
    if not trace_path.is_file():
        raise RuntimeError(f"S3 trace callback did not write {trace_path}")
    trace = cast(dict[str, object], json.loads(trace_path.read_text()))
    trace.update(
        {
            "run_mode": mode,
            "traingen_command": shlex.join(command),
            "traingen_exit_code": completed.returncode,
            "runtime_seconds": runtime_seconds,
            "trace_artifact": trace_path.relative_to(ROOT).as_posix(),
            "stdout_artifact": stdout_path.relative_to(ROOT).as_posix(),
            "checkpoint_root": checkpoint_root.relative_to(ROOT).as_posix(),
            "logger_root": logger_root.relative_to(ROOT).as_posix(),
            "seed_scope": {
                "training_seed": args.seed,
                "rng": "package batch-start RNG restored for independent vendor backward",
            },
        }
    )
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    return trace


def _natural_metric_entries(value: object) -> list[Mapping[str, object]]:
    """Normalize one trajectory metric to its per-optimizer entry list.

    Recorded traces store the loss as one mapping but store each gradient
    norm as a list with one entry per tracked optimizer.
    """
    if isinstance(value, list):
        return cast(list[Mapping[str, object]], value)

    return [cast(Mapping[str, object], value)]


def _natural_run_envelope(
    first: Mapping[str, object], second: Mapping[str, object]
) -> _NaturalEnvelope:
    first_trajectory = cast(list[Mapping[str, object]], first["trajectory"])
    second_trajectory = cast(list[Mapping[str, object]], second["trajectory"])
    if len(first_trajectory) != len(second_trajectory):
        raise RuntimeError(
            "first divergence at S3.natural_run_length; "
            f"first={len(first_trajectory)}; second={len(second_trajectory)}"
        )
    max_abs: dict[str, dict[str, float]] = {
        "loss": {"package": 0.0, "vendor": 0.0},
        "raw_gradient_norm": {"package": 0.0, "vendor": 0.0},
        "clipped_gradient_norm": {"package": 0.0, "vendor": 0.0},
    }
    first_hash_divergence: int | None = None
    for first_step, second_step in zip(
        first_trajectory, second_trajectory, strict=True
    ):
        step = int(cast(int, first_step["global_step"]))
        second_step_number = int(cast(int, second_step["global_step"]))
        if step != second_step_number:
            raise RuntimeError(
                f"first divergence at S3.natural_run_step; first={step}; "
                f"second={second_step_number}"
            )
        for metric in max_abs:
            first_entries = _natural_metric_entries(first_step[metric])
            second_entries = _natural_metric_entries(second_step[metric])
            if len(first_entries) != len(second_entries):
                raise RuntimeError(
                    f"first divergence at S3.natural_{metric}_entry_count; "
                    f"first={len(first_entries)}; second={len(second_entries)}"
                )

            for first_values, second_values in zip(
                first_entries, second_entries, strict=True
            ):
                for side in ("package", "vendor"):
                    first_value = float(cast(float, first_values[side]))
                    second_value = float(cast(float, second_values[side]))
                    max_abs[metric][side] = max(
                        max_abs[metric][side], abs(first_value - second_value)
                    )
        if first_step["package_state_sha256"] != second_step["package_state_sha256"]:
            first_hash_divergence = (
                step if first_hash_divergence is None else first_hash_divergence
            )
    return {
        "run_count": 2,
        "step_count": len(first_trajectory),
        "max_abs_diff": max_abs,
        "first_package_state_hash_divergence_step": first_hash_divergence,
        "package_state_hashes_equal": first_hash_divergence is None,
    }


def _s3(
    args: argparse.Namespace,
    config: RalfConfig,
    data: RalfDataModule,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    del config, device, steps
    if not isinstance(data, RalfDataModule):
        raise RuntimeError("S3 requires the package RalfDataModule")
    if RalfTrainingModule.__name__ != "RalfTrainingModule":
        raise RuntimeError("S3 requires the package RalfTrainingModule")
    if data.train_dataset is None or data.validation_dataset is None:
        raise RuntimeError(
            "S3 requires initialized package train and validation datasets"
        )
    _s3_child_import_gate()

    run_base = ROOT / ".cache" / "ralf" / "training-reproduction" / args.dataset / "s3"
    train_batch_count = (
        len(data.train_dataset) + args.batch_size - 1
    ) // args.batch_size
    validation_batch_count = (
        len(data.validation_dataset) + args.batch_size - 1
    ) // args.batch_size
    train_limit = 3 / train_batch_count
    validation_limit = 2 / validation_batch_count
    natural_runs = [
        _run_s3_fit(args, run_base, "natural", train_limit, validation_limit),
        _run_s3_fit(args, run_base, "natural", train_limit, validation_limit),
    ]
    synchronized_run = _run_s3_fit(
        args, run_base, "synchronized", train_limit, validation_limit
    )
    natural_envelope = _natural_run_envelope(natural_runs[0], natural_runs[1])
    return {
        "status": synchronized_run["status"],
        "production_model": synchronized_run["production_model"],
        "production_datamodule": synchronized_run["production_datamodule"],
        "evidence_layers": {
            "natural_trajectory": natural_runs,
            "state_synchronized_lockstep": synchronized_run,
        },
        "natural_run_to_run_envelope": natural_envelope,
        "limits": {
            "train_batches_per_epoch": 3,
            "validation_batches_per_epoch": 2,
            "train_batch_count": train_batch_count,
            "validation_batch_count": validation_batch_count,
            "train_fraction": train_limit,
            "validation_fraction": validation_limit,
        },
        "first_divergence": {
            "natural_run_1": natural_runs[0]["first_divergence"],
            "natural_run_2": natural_runs[1]["first_divergence"],
            "state_synchronized_lockstep": synchronized_run["first_divergence"],
        },
        "optimizer_step_count": synchronized_run["optimizer_step_count"],
        "global_step": synchronized_run["global_step"],
        "accumulate_grad_batches": synchronized_run["accumulate_grad_batches"],
        "scheduler": synchronized_run["scheduler"],
        "logging": synchronized_run["logging"],
        "checkpoint": synchronized_run["checkpoint"],
        "run_artifacts": {
            "natural": [
                {
                    "trace": run["trace_artifact"],
                    "stdout": run["stdout_artifact"],
                    "checkpoint_root": run["checkpoint_root"],
                    "logger_root": run["logger_root"],
                }
                for run in natural_runs
            ],
            "synchronized": {
                "trace": synchronized_run["trace_artifact"],
                "stdout": synchronized_run["stdout_artifact"],
                "checkpoint_root": synchronized_run["checkpoint_root"],
                "logger_root": synchronized_run["logger_root"],
            },
        },
    }


def _s4(
    args: argparse.Namespace,
    config: RalfConfig,
    data: RalfDataModule,
    context: _TrainingContext,
    steps: int,
) -> dict[str, object]:
    if config.dataset_name != "cgl":
        raise RuntimeError("S4 is scoped to CGL; PKU stream evidence is not claimed")
    if data.train_dataset is None or data.validation_dataset is None:
        raise RuntimeError("S4 package train and validation datasets are unavailable")
    if steps < 1:
        raise ValueError("S4 requires at least one loader batch")
    _ = context
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    require_vendor = __import__(
        "models.ralf.tests.vendor_parity.training_reference",
        fromlist=["require_vendor"],
    ).require_vendor
    require_vendor(args.cache_dir)
    import image2layout.train.global_variables as vendor_globals

    vendor_globals.PRECOMPUTED_WEIGHT_DIR = str(
        args.cache_dir / "PRECOMPUTED_WEIGHT_DIR"
    )
    from image2layout.train.config import get_mock_train_cfg
    from image2layout.train.data import collate_fn as vendor_collate_fn
    from image2layout.train.data import get_dataset as vendor_get_dataset
    from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from image2layout.train.helpers.retrieval_dataset_wrapper import (
        RetrievalDatasetWrapper,
    )

    data_cfg = get_mock_train_cfg(
        config.max_seq_length, str(args.cache_dir / "dataset" / "cgl")
    )
    vendor_dataset, vendor_features = vendor_get_dataset(
        dataset_cfg=data_cfg.dataset,
        transforms=["image", "sort_label", "sort_lexicographic"],
    )
    vendor_tokenizer = LayoutSequenceTokenizer(
        label_feature=vendor_features["label"].feature,
        max_seq_length=config.max_seq_length,
        num_bin=config.num_bin,
        var_order=list(config.var_order),
        special_tokens=list(config.special_tokens),
        is_loc_vocab_shared=config.is_loc_vocab_shared,
        geo_quantization=config.geo_quantization,
    )

    package_datasets = {
        "train": data.train_dataset,
        "val": data.validation_dataset,
    }
    vendor_wrappers = {
        split: RetrievalDatasetWrapper(
            dataset_name="cgl",
            dataset=vendor_dataset[split],
            db_dataset=vendor_dataset["train"],
            split=split,
            top_k=config.top_k,
            max_seq_length=config.max_seq_length,
            retrieval_backbone=config.retrieval_backbone,
            random_retrieval=False,
            saliency_k="None",
        )
        for split in ("train", "val")
    }
    package_loaders = {
        "train": data.train_dataloader(),
        "val": data.val_dataloader(),
    }
    vendor_loaders = {
        split: DataLoader(
            wrapper,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=True,
            collate_fn=partial(vendor_collate_fn, max_seq_length=config.max_seq_length),
            drop_last=False,
        )
        for split, wrapper in vendor_wrappers.items()
    }

    def _first_difference(actual: object, expected: object, path: str) -> str | None:
        if isinstance(actual, Tensor) and isinstance(expected, Tensor):
            return None if torch.equal(actual, expected) else path
        if isinstance(actual, Mapping) and isinstance(expected, Mapping):
            actual_mapping = cast(Mapping[object, object], actual)
            expected_mapping = cast(Mapping[object, object], expected)
            if set(actual_mapping) != set(expected_mapping):
                return path
            for key in sorted(actual_mapping, key=str):
                difference = _first_difference(
                    actual_mapping[key], expected_mapping[key], f"{path}.{key}"
                )
                if difference is not None:
                    return difference
            return None
        if isinstance(actual, RalfRetrievedBatch) and isinstance(
            expected, RalfRetrievedBatch
        ):
            return _first_difference(
                {
                    "image": actual.image,
                    "saliency": actual.saliency,
                    "bbox": actual.bbox,
                    "labels": actual.labels,
                    "mask": actual.mask,
                    "indexes": actual.indexes,
                },
                {
                    "image": expected.image,
                    "saliency": expected.saliency,
                    "bbox": expected.bbox,
                    "labels": expected.labels,
                    "mask": expected.mask,
                    "indexes": expected.indexes,
                },
                path,
            )
        return None if actual == expected else path

    def _package_record(
        item: Mapping[str, object], sample_id: object
    ) -> dict[str, object]:
        retrieved = cast(RalfRetrievedBatch, item["retrieved"])
        bbox = cast(Tensor, item["layout_bbox"])
        return {
            "id": str(sample_id),
            "image": cast(Tensor, item["pixel_values"]),
            "saliency": cast(Tensor, item["saliency"]),
            "labels": cast(Tensor, item["layout_labels"]),
            "bbox": bbox,
            "mask": cast(Tensor, item["layout_mask"]),
            "tokens": torch.cat(
                [
                    cast(Tensor, item["input_ids"])[None, :1],
                    cast(Tensor, item["labels"])[None],
                ],
                dim=1,
            ),
            "retrieved": {
                "image": retrieved.image[0],
                "saliency": retrieved.saliency[0],
                "bbox": retrieved.bbox[0],
                "labels": retrieved.labels[0],
                "mask": retrieved.mask[0],
                "indexes": None if retrieved.indexes is None else retrieved.indexes[0],
            },
        }

    def _vendor_record(
        batch: Mapping[str, object], position: int, retrieval_indexes: object
    ) -> dict[str, object]:
        labels = cast(Tensor, batch["label"])[position].long()
        bbox = torch.stack(
            [
                cast(Tensor, batch[key])[position].float()
                for key in ("center_x", "center_y", "width", "height")
            ],
            dim=-1,
        )
        mask = cast(Tensor, batch["mask"])[position].bool()
        tokens = cast(
            Tensor,
            vendor_tokenizer.encode(
                {
                    "label": labels[None],
                    "center_x": bbox[None, ..., 0],
                    "center_y": bbox[None, ..., 1],
                    "width": bbox[None, ..., 2],
                    "height": bbox[None, ..., 3],
                    "mask": mask[None],
                }
            )["seq"],
        )
        retrieved = cast(list[Mapping[str, object]], batch["retrieved"])[0]
        retrieved_bbox = torch.stack(
            [
                cast(Tensor, retrieved[key])[position].float()
                for key in ("center_x", "center_y", "width", "height")
            ],
            dim=-1,
        )
        return {
            "id": str(cast(Sequence[object], batch["id"])[position]),
            "image": cast(Tensor, batch["image"])[position],
            "saliency": cast(Tensor, batch["saliency"])[position],
            "labels": labels,
            "bbox": bbox,
            "mask": mask,
            "tokens": tokens,
            "retrieved": {
                "image": cast(Tensor, retrieved["image"])[position],
                "saliency": cast(Tensor, retrieved["saliency"])[position],
                "bbox": retrieved_bbox,
                "labels": cast(Tensor, retrieved["label"])[position],
                "mask": cast(Tensor, retrieved["mask"])[position].bool(),
                "indexes": torch.as_tensor(retrieval_indexes, dtype=torch.long),
            },
        }

    split_membership: dict[str, object] = {}
    package_ids_by_split: dict[str, list[str]] = {}
    vendor_ids_by_split: dict[str, list[str]] = {}
    for split, package_dataset in package_datasets.items():
        package_ids = [
            str(sample.get("id", index))
            for index, sample in enumerate(package_dataset.samples)
        ]
        vendor_ids = [str(value) for value in vendor_dataset[split]["id"]]
        package_ids_by_split[split] = package_ids
        vendor_ids_by_split[split] = vendor_ids
        split_membership[split] = {
            "package_count": len(package_ids),
            "vendor_count": len(vendor_ids),
            "package_sha256": _serialized_sha256(package_ids),
            "vendor_sha256": _serialized_sha256(vendor_ids),
            "equal": package_ids == vendor_ids,
        }
        if package_ids != vendor_ids:
            raise RuntimeError(f"first divergence at {split}.split_membership")
    package_overlap = sorted(
        set(package_ids_by_split["train"]) & set(package_ids_by_split["val"])
    )
    vendor_overlap = sorted(
        set(vendor_ids_by_split["train"]) & set(vendor_ids_by_split["val"])
    )
    if package_overlap or vendor_overlap:
        raise RuntimeError("first divergence at split_membership.overlap")
    split_membership["train_val_overlap"] = {
        "package_count": len(package_overlap),
        "vendor_count": len(vendor_overlap),
        "equal": package_overlap == vendor_overlap == [],
    }

    package_stream_digest = hashlib.sha256()
    vendor_stream_digest = hashlib.sha256()
    package_loader_digest = hashlib.sha256()
    vendor_loader_digest = hashlib.sha256()
    split_results: dict[str, object] = {}
    checked_samples = 0
    checked_batches = 0
    for split, package_dataset in package_datasets.items():
        vendor_wrapper = vendor_wrappers[split]
        if len(package_dataset) != len(vendor_wrapper):
            raise RuntimeError(f"first divergence at {split}.length")
        shuffle = split == "train"
        torch.manual_seed(args.seed)
        torch.empty((), dtype=torch.int64).random_()
        batch_sampler = cast(
            Iterable[Sequence[int]], package_loaders[split].batch_sampler
        )
        expected_order = [
            int(index) for batch_indices in batch_sampler for index in batch_indices
        ]
        torch.manual_seed(args.seed)
        package_iterator = iter(package_loaders[split])
        package_loader_rng = capture_rng_state()
        torch.manual_seed(args.seed)
        vendor_iterator = iter(vendor_loaders[split])
        vendor_loader_rng = capture_rng_state()
        split_checked_samples = 0
        split_checked_batches = 0
        split_first_divergence: str | None = None
        for batch_index in range(min(steps, len(package_loaders[split]))):
            restore_rng_state(package_loader_rng)
            package_batch = next(package_iterator)
            package_after_loader_rng = capture_rng_state()
            restore_rng_state(vendor_loader_rng)
            vendor_batch = next(vendor_iterator)
            vendor_after_loader_rng = capture_rng_state()
            if not torch.equal(
                package_after_loader_rng.torch_cpu,
                vendor_after_loader_rng.torch_cpu,
            ):
                split_first_divergence = f"{split}.loader.batch[{batch_index}].rng"
                break
            package_loader_rng = package_after_loader_rng
            vendor_loader_rng = vendor_after_loader_rng
            start = batch_index * args.batch_size
            indexes = expected_order[start : start + args.batch_size]
            if len(indexes) != len(cast(Tensor, package_batch["input_ids"])):
                split_first_divergence = f"{split}.batch[{batch_index}].size"
                break
            expected_package_batch = collate_training_batch(
                [package_dataset[index] for index in indexes]
            )
            difference = _first_difference(
                package_batch,
                expected_package_batch,
                f"{split}.loader.batch[{batch_index}]",
            )
            if difference is not None:
                split_first_divergence = difference
                break
            expected_vendor_ids = [
                str(vendor_dataset[split][index]["id"]) for index in indexes
            ]
            actual_vendor_ids = [str(value) for value in vendor_batch["id"]]
            if actual_vendor_ids != expected_vendor_ids:
                split_first_divergence = (
                    f"{split}.vendor_loader.batch[{batch_index}].id"
                )
                break
            package_loader_digest.update(
                _serialized_sha256(_copy_batch_to_cpu(package_batch)).encode("utf-8")
            )
            vendor_loader_digest.update(
                _serialized_sha256(_copy_batch_to_cpu(vendor_batch)).encode("utf-8")
            )
            for position, index in enumerate(indexes):
                package_item = package_dataset[index]
                vendor_item = vendor_wrapper[index]
                package_record = _package_record(
                    package_item,
                    package_dataset.samples[index].get("id", index),
                )
                vendor_retrieved_item = cast(
                    list[Mapping[str, object]], vendor_item["retrieved"]
                )[0]
                vendor_record = _vendor_record(
                    vendor_batch,
                    position,
                    vendor_retrieved_item["index"],
                )
                difference = _first_difference(
                    package_record,
                    vendor_record,
                    f"{split}.loader.batch[{batch_index}].sample[{position}]",
                )
                if difference is not None:
                    split_first_divergence = difference
                    break
                package_digest = _serialized_sha256(package_record)
                vendor_digest = _serialized_sha256(vendor_record)
                package_stream_digest.update(
                    f"{split}:{index}:".encode("utf-8") + package_digest.encode()
                )
                vendor_stream_digest.update(
                    f"{split}:{index}:".encode("utf-8") + vendor_digest.encode()
                )
                split_checked_samples += 1
            if split_first_divergence is not None:
                break
            split_checked_batches += 1
        if split_first_divergence is not None:
            raise RuntimeError(f"first divergence at {split_first_divergence}")
        checked_samples += split_checked_samples
        checked_batches += split_checked_batches
        split_results[split] = {
            "shuffle": shuffle,
            "seed": args.seed,
            "checked_samples": split_checked_samples,
            "checked_batches": split_checked_batches,
            "dataset_count": len(package_dataset),
            "first_divergence": None,
        }

    return {
        "status": "PASS",
        "checked_samples": checked_samples,
        "checked_batches": checked_batches,
        "first_divergence": None,
        "seed": args.seed,
        "splits": split_results,
        "split_membership": split_membership,
        "package_stream_sha256": package_stream_digest.hexdigest(),
        "vendor_stream_sha256": vendor_stream_digest.hexdigest(),
        "serialized_sha256": {
            "package_loader": package_loader_digest.hexdigest(),
            "vendor_loader": vendor_loader_digest.hexdigest(),
            "package_canonical_stream": package_stream_digest.hexdigest(),
            "vendor_canonical_stream": vendor_stream_digest.hexdigest(),
        },
        "stream_contract": {
            "comparison": "exact tensor equality after vendor-effective transforms and padding",
            "package_loader": "RalfDataModule.train_dataloader/val_dataloader",
            "vendor_loader": "vendor train.py DataLoader with collate_fn",
            "vendor_transforms": ["image", "sort_label", "sort_lexicographic"],
            "retrieval": "fixed table indexes, top_k=16, random_retrieval=false",
            "validation_split": "val",
        },
    }


def main() -> int:
    args = _parse_args()
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("PARITY_REQUIRE=1 is required; refusing an all-skip result")
    device = _device()
    config, data, context = _load_context(args)
    if args.stage == "S0":
        result = _s0(args, config, data, context, device)
    elif args.stage == "S1":
        result = _s1(args, config, context, device)
    elif args.stage == "S2":
        result = _s2(args, config, context, device)
    elif args.stage == "S3":
        result = _s3(
            args, config, data, device, args.steps or DEFAULT_STEPS[args.stage]
        )
    else:
        result = _s4(
            args, config, data, context, args.steps or DEFAULT_STEPS[args.stage]
        )
    candidate = _candidate_metadata()
    vendor = _vendor_metadata()
    result.update(
        {
            "stage": args.stage,
            "dataset": args.dataset,
            "command": " ".join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "runtime_environment": {
                "python": sys.executable,
                "cuda_runtime": torch.version.cuda,
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            },
            "cache_dir": str(args.cache_dir),
            "package_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "candidate": candidate,
            "effective_config_digest": _effective_config_digest(config, args.cache_dir),
            "effective_config_digest_scope": (
                "canonical RalfConfig.to_dict JSON with runtime asset paths relative to cache_dir"
            ),
            "vendor_revision": vendor["revision"],
            "vendor_submodule_status": vendor["submodule_status"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
