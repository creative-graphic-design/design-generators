"""Generate ordered LayoutGAN++ training-reproduction evidence."""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TypeAlias, cast
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from jaxtyping import Float, Int, Shaped
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset

from laygen.common.randomness import randn, resolve_torch_generator
from traingen_parity.compare import (
    BatchStreamReport,
    OptimizerStepReport,
    StepReport,
    compare_batch_stream,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import (
    DeterminismConfig,
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
)
from traingen_parity.trace import (
    build_step_trace,
    summarize_tensor,
    tensor_sha256,
)

from layoutganpp import LayoutGANPPModel
from layoutganpp.training import LayoutGANPPTrainingModule
from layoutganpp.training.dataset import (
    LayoutGANPPDataset,
    LayoutRow,
    collate_layoutganpp,
    load_rows,
)
from layoutganpp.training.step import gan_forward_trace, run_gan_iteration

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / ".cache" / "layoutganpp" / "stage-evidence"
DATA_ROOT = ROOT / ".cache" / "layoutganpp" / "data" / "magazine"
SOURCE_MANIFEST = DATA_ROOT / "source-manifest.json"
VENDOR_ROOT = ROOT / "vendor" / "const-layout"
VENDOR_WORK = ROOT / ".cache" / "layoutganpp" / "vendor-work"
AUDIT_VENV = ROOT / ".cache" / "layoutganpp" / "runtime" / "audit-venv"
AUDIT_FREEZE = ROOT / ".cache" / "layoutganpp" / "runtime" / "pip-freeze.txt"
WHEEL_ROOT = ROOT / ".cache" / "layoutganpp" / "runtime" / "wheels"
CHECKPOINT = (
    ROOT / ".cache" / "layoutganpp" / "original" / "layoutganpp_magazine.pth.tar"
)
CONVERTED = ROOT / ".cache" / "layoutganpp" / "converted" / "layoutganpp-magazine"
VENDOR_CHECKPOINT = VENDOR_WORK / "pretrained" / "layoutganpp_magazine.pth.tar"
VENDOR_LAYOUTNET = VENDOR_WORK / "pretrained" / "layoutnet_magazine.pth.tar"
VENDOR_BATCH_SIZE = 64
WEIGHT_BASE_URL = "https://esslab.jp/~kotaro/files/const_layout"
CHECKPOINT_URL = f"{WEIGHT_BASE_URL}/layoutganpp_magazine.pth.tar"
LAYOUTNET_URL = f"{WEIGHT_BASE_URL}/layoutnet_magazine.pth.tar"
LATENT_SIZE = 4
INIT_SEED = 42975
S1_LATENT_SEED = 42001
S2_LATENT_SEED = 42002
S3_BATCH_SEED = 42003
S3_LATENT_SEED = 42004
S4_EVALUATION_SEED = 42005
S3_STEPS = 300
S3_REPEATS = 2
TRACE_ATOL = 1.0e-6

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


VendorData = Data


def _source_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError(
            "the evidence source tree is dirty; commit the harness before running stages"
        )

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_record(path: Path, url: str, acquisition: str) -> dict[str, JsonValue]:
    available = path.exists()
    return {
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "available": available,
        "size_bytes": path.stat().st_size if available else None,
        "sha256": _sha256(path) if available else None,
        "acquisition": acquisition,
    }


def _audit_runtime() -> dict[str, JsonValue]:
    freeze_hash = _sha256(AUDIT_FREEZE) if AUDIT_FREEZE.exists() else "missing"
    wheels = {
        path.name: _sha256(path)
        for path in sorted(WHEEL_ROOT.glob("torch*.whl"))
        if path.name.startswith(("torch-", "torchvision-"))
    }
    from importlib.metadata import version

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "cuda_tag": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "wheel_sha256": wheels,
        "pip_freeze_sha256": freeze_hash,
        "pip_freeze_path": ".cache/layoutganpp/runtime/pip-freeze.txt",
    }


def _source_manifest() -> tuple[dict[str, JsonValue], str]:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"missing Magazine source manifest: {SOURCE_MANIFEST}")
    manifest = cast(dict[str, JsonValue], json.loads(SOURCE_MANIFEST.read_text()))
    required = {"source_id", "source_revision", "acquisition_command"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"source manifest is missing {missing}")
    return manifest, _sha256(SOURCE_MANIFEST)


def _record(stage: str, started: float, **values: JsonValue) -> dict[str, JsonValue]:
    return {
        "stage": stage,
        "source_commit": _source_commit(),
        "runtime": _audit_runtime(),
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        **values,
    }


def _write(stage: str, payload: dict[str, JsonValue]) -> Path:
    output_dir = OUTPUT_ROOT / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _require_previous(stage: str, previous: str | None) -> None:
    if previous is None:
        return
    path = OUTPUT_ROOT / previous / "summary.json"
    if not path.exists():
        raise RuntimeError(f"{stage} requires the committed {previous} artifact")
    record = cast(dict[str, JsonValue], json.loads(path.read_text()))
    if record.get("source_commit") != _source_commit():
        raise RuntimeError(
            f"{stage} found a previous artifact from another source commit"
        )

    if record.get("result") != "PASS":
        raise RuntimeError(f"{stage} found a previous artifact without PASS result")


def _tensor_summary(value: Shaped[torch.Tensor, "..."]) -> dict[str, JsonValue]:
    summary = summarize_tensor(value)
    return {
        "shape": list(summary.shape),
        "dtype": summary.dtype,
        "device": summary.device,
        "sha256": summary.sha256,
        "min": summary.min,
        "max": summary.max,
        "mean": summary.mean,
    }


def _module_hash(module: torch.nn.Module) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{name}:{tensor_sha256(value)}"
            for name, value in module.state_dict().items()
        ).encode()
    ).hexdigest()


def _state_values_equal(
    left: Mapping[str, Shaped[torch.Tensor, "..."]],
    right: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> bool:
    return list(left) == list(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _state_map(
    module: torch.nn.Module,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _gradient_map(
    module: torch.nn.Module,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    return float(
        torch.sqrt(
            sum(
                (
                    parameter.grad.detach().square().sum()
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ),
                torch.zeros((), device=next(module.parameters()).device),
            )
        ).item()
    )


def _max_parameter_difference(
    module: torch.nn.Module,
    previous: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> float:
    current = module.state_dict()
    return max(
        (
            float((current[name] - previous[name]).abs().max().item())
            for name in current
        ),
        default=0.0,
    )


def _optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {"tensor_count": 0}
    for state_name in ("exp_avg", "exp_avg_sq"):
        total = torch.zeros((), device=next(module.parameters()).device)
        count = 0
        for parameter in module.parameters():
            value = optimizer.state[parameter].get(state_name)
            if isinstance(value, torch.Tensor):
                total = total + value.detach().square().sum()
                count += 1
        summary[f"{state_name}_norm"] = float(torch.sqrt(total).item())
        summary["tensor_count"] = int(summary["tensor_count"]) + count
    return summary


def _optimizer_map(
    optimizer: torch.optim.Optimizer, module: torch.nn.Module
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    return {
        f"{name}.{state_name}": value.detach().clone()
        for name, parameter in module.named_parameters()
        for state_name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }


def _trajectory_record(
    *,
    step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    generator_loss: float,
    discriminator_loss: float,
    previous_generator: Mapping[str, Shaped[torch.Tensor, "..."]],
    previous_discriminator: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> dict[str, JsonValue]:
    return {
        "step": step,
        "generator_loss": generator_loss,
        "discriminator_loss": discriminator_loss,
        "generator_gradient_norm": _gradient_norm(generator),
        "discriminator_gradient_norm": _gradient_norm(discriminator),
        "generator_max_parameter_difference": _max_parameter_difference(
            generator, previous_generator
        ),
        "discriminator_max_parameter_difference": _max_parameter_difference(
            discriminator, previous_discriminator
        ),
        "generator_optimizer_state": _optimizer_state_summary(optimizer_g, generator),
        "discriminator_optimizer_state": _optimizer_state_summary(
            optimizer_d, discriminator
        ),
        "generator_learning_rate": float(optimizer_g.param_groups[0]["lr"]),
        "discriminator_learning_rate": float(optimizer_d.param_groups[0]["lr"]),
    }


def _report_json(report: StepReport | OptimizerStepReport) -> dict[str, JsonValue]:
    first = next((item for item in report.comparisons if not item.passed), None)
    return {
        "passed": report.passed,
        "missing": list(report.missing),
        "first_divergence": None
        if first is None
        else {
            "tensor": first.name,
            "max_abs": first.max_abs_diff,
            "max_rel": first.max_rel_diff,
            "message": first.message,
        },
        "comparisons": [
            {
                "tensor": item.name,
                "passed": item.passed,
                "max_abs": item.max_abs_diff,
                "max_rel": item.max_rel_diff,
            }
            for item in report.comparisons
        ],
    }


def _first_report_divergence(
    reports: Iterable[StepReport | OptimizerStepReport],
) -> dict[str, JsonValue] | None:
    for report in reports:
        value = _report_json(report)["first_divergence"]
        if value is not None:
            return cast(dict[str, JsonValue], value)
    return None


def _float_field(record: Mapping[str, JsonValue], key: str) -> float:
    value = record[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} is not numeric")
    return float(value)


def _vendor_classes() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    if not (VENDOR_ROOT / "train.py").exists():
        raise FileNotFoundError("vendor/const-layout is not initialized")
    sys.path.insert(0, str(VENDOR_ROOT))
    from model.layoutganpp import Discriminator, Generator

    return Generator, Discriminator


def _build_models(
    device: torch.device,
    *,
    copy_vendor_weights: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, LayoutGANPPTrainingModule]:
    generator_cls, discriminator_cls = _vendor_classes()
    torch.manual_seed(INIT_SEED)
    vendor_generator = generator_cls(4, 5, d_model=256, nhead=4, num_layers=8).to(
        device
    )
    vendor_discriminator = discriminator_cls(5, d_model=256, nhead=4, num_layers=8).to(
        device
    )
    torch.manual_seed(INIT_SEED)
    package_module = LayoutGANPPTrainingModule(
        dataset_name="magazine",
        latent_size=4,
        generator_d_model=256,
        generator_nhead=4,
        generator_num_layers=8,
        discriminator_d_model=256,
        discriminator_nhead=4,
        discriminator_num_layers=8,
        discriminator_max_elements=50,
        learning_rate=1.0e-5,
    ).to(device)
    if copy_vendor_weights:
        package_module.generator.load_state_dict(
            vendor_generator.state_dict(), strict=True
        )
        package_module.discriminator.load_state_dict(
            vendor_discriminator.state_dict(), strict=True
        )
    return vendor_generator, vendor_discriminator, package_module


def _package_batch(
    rows: list[LayoutRow], device: torch.device
) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
    batch = collate_layoutganpp(rows)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _vendor_forward_trace(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
    *,
    detach: bool = True,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    bbox = cast(torch.Tensor, batch["bbox"])
    labels = cast(torch.Tensor, batch["labels"])
    mask = cast(torch.Tensor, batch["mask"])
    padding_mask = ~mask
    latent_noise = randn(
        labels.shape[0], labels.shape[1], LATENT_SIZE, device=labels.device
    )
    bbox_fake = generator(latent_noise, labels, padding_mask)
    discriminator_fake_for_g = discriminator(bbox_fake, labels, padding_mask)
    loss_g = torch.nn.functional.softplus(-discriminator_fake_for_g).mean()
    discriminator_fake = discriminator(bbox_fake.detach(), labels, padding_mask)
    loss_d_fake = torch.nn.functional.softplus(discriminator_fake).mean()
    discriminator_real, logits_cls, bbox_reconstruction = discriminator(
        bbox, labels, padding_mask, reconst=True
    )
    loss_d_real = torch.nn.functional.softplus(-discriminator_real).mean()
    loss_d_reconstruction_labels = torch.nn.functional.cross_entropy(
        logits_cls, labels[mask]
    )
    loss_d_reconstruction_boxes = torch.nn.functional.mse_loss(
        bbox_reconstruction, bbox[mask]
    )
    values = {
        "latent_noise": latent_noise,
        "draw_latent_noise": latent_noise,
        "condition_labels": labels,
        "condition_mask": mask,
        "padding_mask": padding_mask,
        "generator_bbox": bbox_fake,
        "generator_discriminator_logits": discriminator_fake_for_g,
        "generator_loss": loss_g.reshape(1),
        "discriminator_fake_bbox": bbox_fake,
        "discriminator_fake_logits": discriminator_fake,
        "discriminator_real_logits": discriminator_real,
        "discriminator_class_logits": logits_cls,
        "discriminator_bbox_reconstruction": bbox_reconstruction,
        "discriminator_fake_loss": loss_d_fake.reshape(1),
        "discriminator_real_loss": loss_d_real.reshape(1),
        "discriminator_label_reconstruction_loss": loss_d_reconstruction_labels.reshape(
            1
        ),
        "discriminator_bbox_reconstruction_loss": loss_d_reconstruction_boxes.reshape(
            1
        ),
        "discriminator_loss": (
            loss_d_real
            + loss_d_fake
            + loss_d_reconstruction_labels
            + 10.0 * loss_d_reconstruction_boxes
        ).reshape(1),
    }
    if detach:
        return {key: value.detach().clone() for key, value in values.items()}
    return values


def _vendor_iteration(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    trace = _vendor_forward_trace(generator, discriminator, batch, detach=False)
    optimizer_g.zero_grad()
    trace["generator_loss"].mean().backward()
    optimizer_g.step()
    optimizer_d.zero_grad()
    trace["discriminator_loss"].mean().backward()
    optimizer_d.step()
    result = {key: value.detach().clone() for key, value in trace.items()}
    result["update_order"] = torch.tensor([0, 1], device=result["latent_noise"].device)
    return result


def _stage_s0(device: torch.device) -> Path:
    started = time.time()
    source, source_hash = _source_manifest()
    vendor_generator, vendor_discriminator, package = _build_models(
        device, copy_vendor_weights=False
    )
    vendor_rows = {split: _vendor_rows(split) for split in ("train", "val", "test")}
    package_rows = {
        split: load_rows("magazine", DATA_ROOT, split)
        for split in ("train", "val", "test")
    }
    vendor_g = torch.optim.Adam(vendor_generator.parameters(), lr=1.0e-5)
    vendor_d = torch.optim.Adam(vendor_discriminator.parameters(), lr=1.0e-5)
    package_g = torch.optim.Adam(package.generator.parameters(), lr=1.0e-5)
    package_d = torch.optim.Adam(package.discriminator.parameters(), lr=1.0e-5)
    vendor_generator_state = _state_map(vendor_generator)
    package_generator_state = _state_map(package.generator)
    vendor_discriminator_state = _state_map(vendor_discriminator)
    package_discriminator_state = _state_map(package.discriminator)
    initialization_equal = _state_values_equal(
        vendor_generator_state, package_generator_state
    ) and _state_values_equal(vendor_discriminator_state, package_discriminator_state)
    optimizer_defaults_equal = (
        vendor_g.defaults == package_g.defaults
        and vendor_d.defaults == package_d.defaults
    )
    optimizer_state_equal = (
        vendor_g.state_dict()["state"] == package_g.state_dict()["state"] == {}
        and vendor_d.state_dict()["state"] == package_d.state_dict()["state"] == {}
    )
    dataset_equal = all(
        len(vendor_rows[split]) == len(package_rows[split])
        and [str(row.attr["name"]) for row in vendor_rows[split]]
        == [row.name for row in package_rows[split]]
        for split in vendor_rows
    )
    payload = _record(
        "s0-static",
        started,
        result="PASS"
        if dataset_equal
        and initialization_equal
        and optimizer_defaults_equal
        and optimizer_state_equal
        and sum(p.numel() for p in vendor_generator.parameters())
        == sum(p.numel() for p in package.generator.parameters())
        and sum(p.numel() for p in vendor_discriminator.parameters())
        == sum(p.numel() for p in package.discriminator.parameters())
        else "FAIL",
        source_manifest=".cache/layoutganpp/data/magazine/source-manifest.json",
        source_manifest_sha256=source_hash,
        source=source,
        initialization={
            "seed": INIT_SEED,
            "vendor_generator_state_sha256": _module_hash(vendor_generator),
            "package_generator_state_sha256": _module_hash(package.generator),
            "vendor_discriminator_state_sha256": _module_hash(vendor_discriminator),
            "package_discriminator_state_sha256": _module_hash(package.discriminator),
            "compared_as": "independently constructed state metadata; no weights copied",
            "values_equal": initialization_equal,
        },
        parameter_counts={
            "vendor_generator": sum(p.numel() for p in vendor_generator.parameters()),
            "package_generator": sum(p.numel() for p in package.generator.parameters()),
            "vendor_discriminator": sum(
                p.numel() for p in vendor_discriminator.parameters()
            ),
            "package_discriminator": sum(
                p.numel() for p in package.discriminator.parameters()
            ),
        },
        state_dict_key_map={
            "generator_equal": list(vendor_generator.state_dict())
            == list(package.generator.state_dict()),
            "discriminator_equal": list(vendor_discriminator.state_dict())
            == list(package.discriminator.state_dict()),
        },
        optimizer_static_state={
            "class": "torch.optim.Adam",
            "vendor_defaults": vendor_g.defaults,
            "package_defaults": package_g.defaults,
            "vendor_discriminator_defaults": vendor_d.defaults,
            "package_discriminator_defaults": package_d.defaults,
            "defaults_equal": optimizer_defaults_equal,
            "state_empty": optimizer_state_equal,
        },
        dataset_static={
            split: {
                "vendor_count": len(vendor_rows[split]),
                "package_count": len(package_rows[split]),
                "vendor_names_sha256": hashlib.sha256(
                    "\n".join(
                        str(row.attr["name"]) for row in vendor_rows[split]
                    ).encode()
                ).hexdigest(),
                "package_names_sha256": hashlib.sha256(
                    "\n".join(row.name for row in package_rows[split]).encode()
                ).hexdigest(),
            }
            for split in vendor_rows
        },
        initial_state_equal=initialization_equal,
        optimizer_static_equal=optimizer_defaults_equal and optimizer_state_equal,
        inactive_rules=["scheduler", "EMA", "AMP", "multi-worker randomness in S0"],
        first_divergence=None,
    )
    return _write("s0-static", payload)


def _stage_batch(
    device: torch.device,
) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
    return _package_batch(
        load_rows("magazine", DATA_ROOT, "train")[:VENDOR_BATCH_SIZE], device
    )


def _stage_s1(device: torch.device) -> Path:
    started = time.time()
    _require_previous("s1-fixed-batch", "s0-static")
    batch = _stage_batch(device)
    vendor_generator, vendor_discriminator, package = _build_models(
        device, copy_vendor_weights=True
    )
    torch.manual_seed(S1_LATENT_SEED)
    state = capture_rng_state()
    vendor_trace = _vendor_forward_trace(vendor_generator, vendor_discriminator, batch)
    restore_rng_state(state)
    package_trace = gan_forward_trace(package.generator, package.discriminator, batch)
    report = compare_step_trace(
        build_step_trace("vendor", vendor_trace),
        build_step_trace("package", package_trace),
    )
    artifact = OUTPUT_ROOT / "s1-fixed-batch" / "trace.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vendor": vendor_trace, "package": package_trace}, artifact)
    payload = _record(
        "s1-fixed-batch",
        started,
        result="PASS" if report.passed else "FAIL",
        trace_artifact=str(artifact.relative_to(ROOT)),
        draws={
            "vendor": "vendor/const-layout/train.py:124 torch.randn in the vendor adapter",
            "package": "layoutganpp.training.step._resolve_latent_noise",
            "latent_seed": S1_LATENT_SEED,
        },
        named_tensors={
            key: _tensor_summary(value) for key, value in package_trace.items()
        },
        comparison=_report_json(report),
        first_divergence=_report_json(report)["first_divergence"],
    )
    return _write("s1-fixed-batch", payload)


def _stage_s2(device: torch.device) -> Path:
    started = time.time()
    _require_previous("s2-one-step", "s1-fixed-batch")
    batch = _stage_batch(device)
    vendor_generator, vendor_discriminator, package = _build_models(
        device, copy_vendor_weights=True
    )
    vendor_g = torch.optim.Adam(vendor_generator.parameters(), lr=1.0e-5)
    vendor_d = torch.optim.Adam(vendor_discriminator.parameters(), lr=1.0e-5)
    package_g = torch.optim.Adam(package.generator.parameters(), lr=1.0e-5)
    package_d = torch.optim.Adam(package.discriminator.parameters(), lr=1.0e-5)
    torch.manual_seed(S2_LATENT_SEED)
    state = capture_rng_state()
    vendor_trace = _vendor_iteration(
        vendor_generator, vendor_discriminator, batch, vendor_g, vendor_d
    )
    vendor_gradients = {
        "generator": _gradient_map(vendor_generator),
        "discriminator": _gradient_map(vendor_discriminator),
    }
    vendor_states = {
        "generator": _optimizer_map(vendor_g, vendor_generator),
        "discriminator": _optimizer_map(vendor_d, vendor_discriminator),
    }
    restore_rng_state(state)
    package_trace = run_gan_iteration(
        package.generator, package.discriminator, batch, package_g, package_d
    )
    package_gradients = {
        "generator": _gradient_map(package.generator),
        "discriminator": _gradient_map(package.discriminator),
    }
    package_states = {
        "generator": _optimizer_map(package_g, package.generator),
        "discriminator": _optimizer_map(package_d, package.discriminator),
    }
    trace_report = compare_step_trace(
        build_step_trace("vendor", vendor_trace),
        build_step_trace("package", package_trace),
    )
    gradient_reports = {
        name: compare_optimizer_step(vendor_gradients[name], package_gradients[name])
        for name in vendor_gradients
    }
    state_reports = {
        name: compare_optimizer_step(vendor_states[name], package_states[name])
        for name in vendor_states
    }
    parameter_reports = {
        "generator": compare_optimizer_step(
            _state_map(vendor_generator), _state_map(package.generator)
        ),
        "discriminator": compare_optimizer_step(
            _state_map(vendor_discriminator), _state_map(package.discriminator)
        ),
    }
    reports = [
        trace_report,
        *gradient_reports.values(),
        *state_reports.values(),
        *parameter_reports.values(),
    ]
    payload = _record(
        "s2-one-step",
        started,
        result="PASS" if all(report.passed for report in reports) else "FAIL",
        draws={
            "vendor": "vendor adapter torch.randn path",
            "package": "package step path",
            "latent_seed": S2_LATENT_SEED,
        },
        update_order={
            "vendor": vendor_trace["update_order"].tolist(),
            "package": package_trace["update_order"].tolist(),
        },
        gradients={
            name: _report_json(report) for name, report in gradient_reports.items()
        },
        optimizer_state={
            name: _report_json(report) for name, report in state_reports.items()
        },
        post_step_parameters={
            name: _report_json(report) for name, report in parameter_reports.items()
        },
        trace_comparison=_report_json(trace_report),
        first_divergence=_first_report_divergence(reports),
    )
    artifact = OUTPUT_ROOT / "s2-one-step" / "trace.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vendor": vendor_trace, "package": package_trace}, artifact)
    payload["trace_artifact"] = str(artifact.relative_to(ROOT))
    return _write("s2-one-step", payload)


def _vendor_rows(split: str) -> list[Data]:
    VENDOR_WORK.mkdir(parents=True, exist_ok=True)
    dataset_root = VENDOR_WORK / "data" / "dataset" / "magazine"
    raw_root = dataset_root / "raw"
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if raw_root.exists() and not raw_root.is_symlink():
        raise RuntimeError(f"vendor raw path is not the expected symlink: {raw_root}")
    if not raw_root.exists():
        raw_root.symlink_to(DATA_ROOT.resolve(), target_is_directory=True)
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_env = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.chdir(VENDOR_WORK)
    sys.path.insert(0, str(VENDOR_ROOT))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        from data.magazine import Magazine
        from data.util import LexicographicSort

        transform = LexicographicSort() if split == "train" else None
        return cast(list[Data], list(Magazine(split, transform=transform)))
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if old_env is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_env


def _dense_vendor_batch(
    data: Data, device: torch.device
) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
    from torch_geometric.utils import to_dense_batch

    batch = data.batch
    labels, mask = to_dense_batch(data.y, batch)
    bbox, _ = to_dense_batch(data.x, batch)
    names = [str(item.attr["name"]) for item in data.to_data_list()]
    return {
        "bbox": bbox.to(device),
        "labels": labels.to(device),
        "mask": mask.to(device),
        "names": names,
    }


def _package_loader(split: str, seed: int) -> DataLoader[LayoutRow]:
    dataset = LayoutGANPPDataset(
        dataset_name="magazine",
        split=split,
        data_root=DATA_ROOT,
        synthetic_size=0,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=VENDOR_BATCH_SIZE,
        shuffle=split == "train",
        num_workers=0,
        collate_fn=collate_layoutganpp,
        generator=resolve_torch_generator(seed=seed),
    )


def _vendor_loader(split: str, seed: int) -> Iterable[Data]:
    from torch_geometric.data import DataLoader as VendorLoader

    rows = _vendor_rows(split)
    return VendorLoader(
        rows,
        batch_size=VENDOR_BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=split == "train",
        generator=resolve_torch_generator(seed=seed),
    )


def _loader_stream(
    loader: Iterable[Data | dict[str, Shaped[torch.Tensor, "..."] | list[str]]],
) -> Iterator[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]:
    while True:
        for raw in loader:
            if isinstance(raw, dict):
                yield cast(dict[str, Shaped[torch.Tensor, "..."] | list[str]], raw)
            else:
                yield _dense_vendor_batch(cast(Data, raw), torch.device("cpu"))


def _compare_loader_streams(
    device: torch.device,
    steps: int,
    *,
    split: str = "train",
    seed: int = S3_BATCH_SEED,
) -> BatchStreamReport:
    package_stream = (
        {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        for batch in _loader_stream(_package_loader(split, seed))
    )
    vendor_stream = (
        {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        for batch in _loader_stream(_vendor_loader(split, seed))
    )
    del device
    return compare_batch_stream(vendor_stream, package_stream, steps=steps)


def _loader_stream_summary(
    device: torch.device, split: str, seed: int
) -> dict[str, JsonValue]:
    row_count = len(load_rows("magazine", DATA_ROOT, split))
    expected_batches = (row_count + VENDOR_BATCH_SIZE - 1) // VENDOR_BATCH_SIZE
    report = _compare_loader_streams(device, expected_batches, split=split, seed=seed)
    return {
        "passed": report.passed and report.checked_steps == expected_batches,
        "checked_batches": report.checked_steps,
        "expected_batches": expected_batches,
        "row_count": row_count,
        "seed": seed,
        "shuffle": split == "train",
        "first_mismatch": report.first_mismatch,
    }


def _all_loader_streams(device: torch.device, seed: int) -> dict[str, JsonValue]:
    return {
        split: _loader_stream_summary(device, split, seed)
        for split in ("train", "val", "test")
    }


class _BatchDataset(Dataset[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]):
    def __init__(
        self,
        batches: list[dict[str, Shaped[torch.Tensor, "..."] | list[str]]],
    ) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(
        self, index: int
    ) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
        return self.batches[index]


def _identity_batch(
    batch: list[dict[str, Shaped[torch.Tensor, "..."] | list[str]]],
) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
    if len(batch) != 1:
        raise ValueError("the production Trainer must receive one pre-collated batch")
    return batch[0]


def _script_entry_point(path: Path) -> str:
    tree = ast.parse(path.read_text())
    functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if len(functions) != 1:
        raise RuntimeError(f"expected one main entry point in {path}")
    return f"{path.relative_to(ROOT)}:{functions[0]}"


def _relative_command(command: list[str]) -> list[str]:
    relative: list[str] = []
    for item in command:
        path = Path(item)
        if path.is_absolute():
            try:
                relative.append(str(path.relative_to(ROOT)))
                continue
            except ValueError:
                pass
        relative.append(item)
    return relative


def _command_option(command: list[str], option: str, converter: type[int] = int) -> int:
    try:
        value = command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"executed command lacks {option}") from error
    return converter(value)


def _optimizer_parameter_names(
    optimizer: torch.optim.Optimizer,
    modules: Mapping[str, torch.nn.Module],
) -> list[list[str]]:
    names = {
        id(parameter): f"{module_name}.{name}"
        for module_name, module in modules.items()
        for name, parameter in module.named_parameters()
    }
    return [
        [names[id(parameter)] for parameter in group["params"]]
        for group in optimizer.param_groups
    ]


def _package_training_trajectory(
    device: torch.device,
    steps: int,
    seed: int,
    initial_generator: Mapping[str, Shaped[torch.Tensor, "..."]],
    initial_discriminator: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue]]:
    batches = []
    loader = _loader_stream(_package_loader("train", S3_BATCH_SEED))
    for _ in range(steps):
        batches.append(next(loader))
    _, _, package = _build_models(device, copy_vendor_weights=False)
    package.generator.load_state_dict(initial_generator, strict=True)
    package.discriminator.load_state_dict(initial_discriminator, strict=True)

    class Capture(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[dict[str, JsonValue]] = []
            self.previous_generator = _state_map(package.generator)
            self.previous_discriminator = _state_map(package.discriminator)

        def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
            batch_idx: int,
        ) -> None:
            del outputs, batch
            optimizers = trainer.optimizers
            if len(optimizers) != 2:
                raise RuntimeError(
                    "LayoutGAN++ production training needs two optimizers"
                )
            optimizer_g, optimizer_d = optimizers
            package_module = cast(LayoutGANPPTrainingModule, pl_module)
            trace = package_module.latest_step_trace
            record = _trajectory_record(
                step=batch_idx,
                generator=package_module.generator,
                discriminator=package_module.discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                generator_loss=float(trace["generator_loss"].item()),
                discriminator_loss=float(trace["discriminator_loss"].item()),
                previous_generator=self.previous_generator,
                previous_discriminator=self.previous_discriminator,
            )
            self.records.append(record)
            self.previous_generator = _state_map(package_module.generator)
            self.previous_discriminator = _state_map(package_module.discriminator)

    capture = Capture()
    data_loader = DataLoader(
        _BatchDataset(batches),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_identity_batch,
    )
    torch.manual_seed(seed)
    trainer = Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        max_epochs=1,
        limit_train_batches=steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        gradient_clip_val=0.0,
        callbacks=[capture],
        default_root_dir=str(OUTPUT_ROOT / "s3-lockstep" / "production"),
    )
    trainer.fit(package, train_dataloaders=data_loader)
    if len(capture.records) != steps:
        raise RuntimeError(
            f"package Trainer recorded {len(capture.records)} steps, expected {steps}"
        )
    package_optimizers = trainer.optimizers
    metadata: dict[str, JsonValue] = {
        "entry_points": {
            "training_step": (
                f"{package.training_step.__self__.__class__.__module__}:"
                f"{package.training_step.__qualname__}"
            ),
            "configure_optimizers": (
                f"{package.configure_optimizers.__self__.__class__.__module__}:"
                f"{package.configure_optimizers.__qualname__}"
            ),
        },
        "trainer": {
            "gradient_clip_val": trainer.gradient_clip_val,
            "gradient_clip_algorithm": str(trainer.gradient_clip_algorithm),
            "optimizer_count": len(package_optimizers),
            "optimizer_parameter_names": _optimizer_parameter_names(
                package_optimizers[0],
                {
                    "generator": package.generator,
                    "discriminator": package.discriminator,
                },
            ),
            "fit_loop_batches": trainer.fit_loop.max_batches,
            "model_training": package.training,
        },
        "fit_calls": 1,
        "latent_seed": seed,
    }
    return capture.records, metadata


def _vendor_training_trajectory(
    device: torch.device,
    system: str,
    steps: int,
    seed: int,
    initial_generator: Mapping[str, Shaped[torch.Tensor, "..."]],
    initial_discriminator: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue]]:
    if system != "vendor":
        raise ValueError("the vendor entry point helper only accepts the vendor system")
    generator_cls, discriminator_cls = _vendor_classes()
    captured_models: dict[str, torch.nn.Module] = {}
    constructed_states: dict[str, dict[str, Shaped[torch.Tensor, "..."]]] = {}
    captured_optimizers: list["CapturingAdam"] = []
    captured_losses: list[float] = []
    records: list[dict[str, JsonValue]] = []
    previous_generator = dict(initial_generator)
    previous_discriminator = dict(initial_discriminator)

    class CapturingGenerator(generator_cls):
        def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            super().__init__(*args, **kwargs)
            captured_models["generator"] = self

    class CapturingDiscriminator(discriminator_cls):
        def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            super().__init__(*args, **kwargs)
            captured_models["discriminator"] = self
            constructed_states["generator"] = _state_map(captured_models["generator"])
            constructed_states["discriminator"] = _state_map(self)
            torch.manual_seed(seed)

    class CapturingAdam:
        def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1.0e-3,
            betas: tuple[float, float] = (0.9, 0.999),
        ) -> None:
            self.raw = torch.optim.Adam(params, lr=lr, betas=betas)
            captured_optimizers.append(self)

        def zero_grad(self) -> None:
            self.raw.zero_grad()

        def step(
            self, closure: Callable[[], float] | None = None
        ) -> int | float | None:
            result = self.raw.step(closure=closure)
            if len(captured_optimizers) == 2 and self is captured_optimizers[0]:
                if len(captured_losses) < 2:
                    raise RuntimeError("vendor training step did not record two losses")
                generator = captured_models["generator"]
                discriminator = captured_models["discriminator"]
                generator_optimizer = captured_optimizers[1].raw
                discriminator_optimizer = captured_optimizers[0].raw
                records.append(
                    _trajectory_record(
                        step=len(records),
                        generator=generator,
                        discriminator=discriminator,
                        optimizer_g=generator_optimizer,
                        optimizer_d=discriminator_optimizer,
                        generator_loss=captured_losses[-2],
                        discriminator_loss=captured_losses[-1],
                        previous_generator=previous_generator,
                        previous_discriminator=previous_discriminator,
                    )
                )
                previous_generator.clear()
                previous_generator.update(_state_map(generator))
                previous_discriminator.clear()
                previous_discriminator.update(_state_map(discriminator))
                captured_losses.clear()
            return result

    class DatasetView(Dataset[Data]):
        def __init__(self, rows: list[Data], length: int) -> None:
            self.rows = rows
            self.num_classes = 5
            self.colors: list[tuple[int, int, int]] = []
            self.length = length

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int) -> Data:
            return self.rows[index % len(self.rows)]

    class Sequence:
        def __init__(self, batches: list[Data]) -> None:
            self.batches = batches

        def __len__(self) -> int:
            return len(self.batches)

        def __iter__(self) -> Iterator[Data]:
            return iter(self.batches)

    train_rows = _vendor_rows("train")
    val_rows = _vendor_rows("val")
    train_dataset = DatasetView(train_rows, steps * VENDOR_BATCH_SIZE)
    val_dataset = DatasetView(val_rows, len(val_rows))
    train_batches = list(_vendor_loader("train", S3_BATCH_SEED))
    train_batches = (
        train_batches * ((steps + len(train_batches) - 1) // len(train_batches))
    )[:steps]
    val_batches = list(_vendor_loader("val", S3_BATCH_SEED))
    captured_module = importlib.util.spec_from_file_location(
        "layoutganpp_vendor_train", VENDOR_ROOT / "train.py"
    )
    if captured_module is None or captured_module.loader is None:
        raise RuntimeError("could not load the vendor training entry point")
    vendor_train = importlib.util.module_from_spec(captured_module)
    captured_module.loader.exec_module(vendor_train)
    vendor_module: ModuleType = vendor_train
    vendor_module.__dict__["Generator"] = CapturingGenerator
    vendor_module.__dict__["Discriminator"] = CapturingDiscriminator
    vendor_module.__dict__["optim"] = SimpleNamespace(Adam=CapturingAdam)
    vendor_module.__dict__["get_dataset"] = lambda name, split, transform=None: (
        train_dataset if split == "train" else val_dataset
    )
    vendor_module.__dict__["DataLoader"] = (
        lambda dataset, batch_size, num_workers, pin_memory, shuffle: Sequence(
            train_batches if shuffle else val_batches
        )
    )

    class NoopWriter:
        def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            del args, kwargs

        def add_scalar(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            del args, kwargs

        def add_scalars(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            del args, kwargs

    class NoopFID:
        def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
            del args, kwargs

        def collect_features(
            self,
            *args: Shaped[torch.Tensor, "..."] | bool,
            **kwargs: Shaped[torch.Tensor, "..."] | bool,
        ) -> None:
            del args, kwargs

        def compute_score(self) -> float:
            return 0.0

    vendor_module.__dict__["SummaryWriter"] = NoopWriter
    vendor_module.__dict__["LayoutFID"] = NoopFID
    vendor_module.__dict__["save_image"] = lambda *args, **kwargs: None
    vendor_module.__dict__["save_checkpoint"] = lambda *args, **kwargs: None
    original_backward = torch.Tensor.backward
    original_argv = sys.argv
    original_cwd = Path.cwd()
    args = [
        "--name",
        f"s3-{seed}",
        "--dataset",
        "magazine",
        "--batch_size",
        str(VENDOR_BATCH_SIZE),
        "--iteration",
        str(steps),
        "--seed",
        str(seed),
    ]

    def capture_backward(
        self: Shaped[torch.Tensor, "..."],
        *args: Shaped[torch.Tensor, "..."] | bool,
        **kwargs: Shaped[torch.Tensor, "..."] | bool,
    ) -> None:
        captured_losses.append(float(self.detach().item()))
        original_backward(self, *args, **kwargs)  # type: ignore[arg-type]

    try:
        torch.Tensor.backward = capture_backward  # type: ignore[method-assign]
        sys.argv = [str(VENDOR_ROOT / "train.py"), *args]
        os.chdir(VENDOR_WORK)
        cast(Callable[[], None], vendor_module.__dict__["main"])()
    finally:
        torch.Tensor.backward = original_backward  # type: ignore[method-assign]
        sys.argv = original_argv
        os.chdir(original_cwd)
    if len(records) != steps:
        raise RuntimeError(
            f"vendor train.py recorded {len(records)} steps, expected {steps}"
        )
    if not _state_values_equal(initial_generator, constructed_states["generator"]):
        raise RuntimeError("vendor entry point changed the initialized generator")
    if not _state_values_equal(
        initial_discriminator, constructed_states["discriminator"]
    ):
        raise RuntimeError("vendor entry point changed the initialized discriminator")
    metadata: dict[str, JsonValue] = {
        "entry_point": _script_entry_point(VENDOR_ROOT / "train.py"),
        "command": shlex.join(
            ["python", str((VENDOR_ROOT / "train.py").relative_to(ROOT)), *args]
        ),
        "training_module_calls": len(records),
        "optimizer_creation_order": [
            _optimizer_parameter_names(
                optimizer.raw,
                {
                    "generator": captured_models["generator"],
                    "discriminator": captured_models["discriminator"],
                },
            )
            for optimizer in captured_optimizers
        ],
        "model_training": {
            name: module.training for name, module in captured_models.items()
        },
        "latent_seed": seed,
    }
    del device
    return records, metadata


def _trajectory(
    device: torch.device,
    system: str,
    steps: int,
    seed: int,
    initial_generator: Mapping[str, Shaped[torch.Tensor, "..."]],
    initial_discriminator: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue]]:
    if system == "package":
        return _package_training_trajectory(
            device,
            steps,
            seed,
            initial_generator,
            initial_discriminator,
        )
    return _vendor_training_trajectory(
        device,
        system,
        steps,
        seed,
        initial_generator,
        initial_discriminator,
    )


def _stage_s3(device: torch.device) -> Path:
    started = time.time()
    _require_previous("s3-lockstep", "s2-one-step")
    stream = _compare_loader_streams(device, S3_STEPS)
    initial_generator, initial_discriminator, _ = _build_models(
        device, copy_vendor_weights=False
    )
    initial_g = {
        name: value.detach().clone()
        for name, value in initial_generator.state_dict().items()
    }
    initial_d = {
        name: value.detach().clone()
        for name, value in initial_discriminator.state_dict().items()
    }
    natural: dict[str, list[list[dict[str, JsonValue]]]] = {"vendor": [], "package": []}
    production_runs: list[dict[str, JsonValue]] = []
    for system in natural:
        for repeat in range(S3_REPEATS):
            trajectory, metadata = _trajectory(
                device,
                system,
                S3_STEPS,
                S3_LATENT_SEED + repeat,
                initial_g,
                initial_d,
            )
            natural[system].append(trajectory)
            if system == "package":
                production_runs.append(metadata)
    first: dict[str, JsonValue] | None = None
    for step in range(S3_STEPS):
        for name in natural["vendor"][0][step]:
            left = natural["vendor"][0][step][name]
            right = natural["package"][0][step][name]
            if left != right:
                first = {"step": step, "field": name, "vendor": left, "package": right}
                break
        if first is not None:
            break
    repeat_envelope = {
        system: {
            "repeat_count": S3_REPEATS,
            "max_abs_generator_loss": max(
                abs(
                    _float_field(natural[system][0][step], "generator_loss")
                    - _float_field(natural[system][repeat][step], "generator_loss")
                )
                for repeat in range(1, S3_REPEATS)
                for step in range(S3_STEPS)
            )
            if S3_REPEATS > 1
            else 0.0,
            "max_abs_discriminator_loss": max(
                abs(
                    _float_field(natural[system][0][step], "discriminator_loss")
                    - _float_field(natural[system][repeat][step], "discriminator_loss")
                )
                for repeat in range(1, S3_REPEATS)
                for step in range(S3_STEPS)
            )
            if S3_REPEATS > 1
            else 0.0,
        }
        for system in natural
    }
    natural_path = OUTPUT_ROOT / "s3-lockstep" / "natural.json"
    natural_path.parent.mkdir(parents=True, exist_ok=True)
    natural_path.write_text(json.dumps(natural, indent=2, sort_keys=True) + "\n")
    production_path = OUTPUT_ROOT / "s3-lockstep" / "production-wiring.json"
    command = [
        sys.executable,
        "-m",
        "traingen.lightning.cli",
        "fit",
        "--config",
        "models/layoutganpp/configs/training/layoutganpp_magazine.yaml",
        "--trainer.limit_train_batches=1",
        "--trainer.limit_val_batches=0",
        "--trainer.default_root_dir=.cache/layoutganpp/stage-evidence/s3-lockstep/production",
    ]
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    production_path.write_text(process.stdout + process.stderr)
    summary = _record(
        "s3-lockstep",
        started,
        result="PASS"
        if stream.passed and first is None and process.returncode == 0
        else "FAIL",
        natural_artifact=str(natural_path.relative_to(ROOT)),
        natural_steps=S3_STEPS,
        batch_stream={
            "checked_steps": stream.checked_steps,
            "passed": stream.passed,
            "first_mismatch": stream.first_mismatch,
            "batch_seed": S3_BATCH_SEED,
        },
        latent_seeds=[S3_LATENT_SEED + repeat for repeat in range(S3_REPEATS)],
        latent_seed_policy={
            "per_repeat_run": [
                {
                    "repeat": repeat,
                    "seed": S3_LATENT_SEED + repeat,
                    "systems": list(natural),
                }
                for repeat in range(S3_REPEATS)
            ],
            "draw_paths": {
                "vendor": "vendor adapter torch.randn path",
                "package": "package step path",
            },
        },
        repeat_run_envelope=repeat_envelope,
        synchronized_layer={
            "status": "not-needed" if first is None else "required",
            "first_natural_divergence": first,
        },
        production_wiring={
            "command": " ".join(command),
            "returncode": process.returncode,
            "artifact": str(production_path.relative_to(ROOT)),
            "package_trainer_runs": production_runs,
        },
        first_divergence=first,
    )
    return _write("s3-lockstep", summary)


def _require_checkpoint_assets() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Magazine checkpoint is required; run models/layoutganpp/scripts/download_original_weights.py"
        )
    if not VENDOR_LAYOUTNET.exists():
        raise FileNotFoundError(
            "vendor LayoutFID requires pretrained/layoutnet_magazine.pth.tar"
        )
    VENDOR_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not VENDOR_CHECKPOINT.exists():
        VENDOR_CHECKPOINT.symlink_to(CHECKPOINT.resolve())
    if not CONVERTED.exists():
        subprocess.run(
            [
                sys.executable,
                "models/layoutganpp/scripts/convert_original_checkpoint.py",
                "--input-checkpoint",
                str(CHECKPOINT.relative_to(ROOT)),
                "--output-dir",
                str(CONVERTED.relative_to(ROOT)),
            ],
            cwd=ROOT,
            check=True,
        )


def _clear_vendor_processed_cache() -> bool:
    processed = VENDOR_WORK / "data" / "dataset" / "magazine" / "processed"
    existed = processed.exists()
    if processed.exists():
        shutil.rmtree(processed)
    return existed


def _parse_vendor_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in ("FID", "Max. IoU"):
        match = re.search(
            rf"^\s*{re.escape(name)}:\s+([-+0-9.eE]+)",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"vendor evaluator did not report {name}")
        metrics[name] = float(match.group(1))
    return metrics


def _compare_prediction_rows(
    vendor: list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
    package: list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
) -> tuple[float, dict[str, JsonValue] | None]:
    if len(vendor) != len(package):
        return float("inf"), {
            "field": "prediction_count",
            "vendor": len(vendor),
            "package": len(package),
        }

    maximum = 0.0
    for index, (vendor_row, package_row) in enumerate(zip(vendor, package)):
        vendor_boxes = torch.as_tensor(vendor_row[0], dtype=torch.float32)
        package_boxes = torch.as_tensor(package_row[0], dtype=torch.float32)
        if vendor_boxes.shape != package_boxes.shape:
            return float("inf"), {
                "index": index,
                "field": "prediction_shape",
                "vendor": list(vendor_boxes.shape),
                "package": list(package_boxes.shape),
            }
        difference = (vendor_boxes - package_boxes).abs()
        maximum = max(
            maximum, float(difference.max().item()) if difference.numel() else 0.0
        )
        if not torch.equal(
            torch.as_tensor(vendor_row[1]), torch.as_tensor(package_row[1])
        ):
            return maximum, {"index": index, "field": "labels"}

    if maximum > TRACE_ATOL:
        return maximum, {"field": "prediction_values", "max_abs": maximum}
    return maximum, None


def _out_of_bounds_counts(
    values: list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
) -> dict[str, int]:
    box_count = 0
    layout_count = 0
    for boxes, _ in values:
        tensor = torch.as_tensor(boxes)
        if not tensor.numel():
            invalid = tensor.new_zeros(0, dtype=torch.bool)
        else:
            components_outside = ((tensor < 0) | (tensor > 1)).any(dim=1)
            left = tensor[:, 0] - tensor[:, 2] / 2
            top = tensor[:, 1] - tensor[:, 3] / 2
            right = tensor[:, 0] + tensor[:, 2] / 2
            bottom = tensor[:, 1] + tensor[:, 3] / 2
            edges_outside = (left < 0) | (top < 0) | (right > 1) | (bottom > 1)
            invalid = components_outside | edges_outside

        box_count += int(invalid.sum().item())
        layout_count += int(invalid.any().item())
    return {
        "out_of_bounds_box_count": box_count,
        "out_of_bounds_layout_count": layout_count,
    }


def _stage_s4(device: torch.device) -> Path:
    started = time.time()
    _require_previous("s4-loader-eval", "s3-lockstep")
    processed_cache_existed = _clear_vendor_processed_cache()
    vendor_rows = _vendor_rows("test")
    package_rows = load_rows("magazine", DATA_ROOT, "test")
    package_names = [row.name for row in package_rows]
    vendor_names = [str(row.attr["name"]) for row in vendor_rows]
    if vendor_names != package_names:
        raise RuntimeError(
            "vendor and package TEST rows diverge after cache regeneration"
        )
    evaluator_dir = OUTPUT_ROOT / "s4-loader-eval"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    source, source_hash = _source_manifest()
    loader_streams = _all_loader_streams(device, S4_EVALUATION_SEED)
    evaluation_path = evaluator_dir / "evaluation-path.json"

    try:
        _require_checkpoint_assets()
    except FileNotFoundError as error:
        retry_summary = os.environ.get("LAYOUTGANPP_CHECKPOINT_RETRY_SUMMARY")
        blocker = retry_summary or str(error)
        vendor_source_commit = subprocess.check_output(
            ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        blocked_evaluation = {
            "status": f"blocked ({blocker})",
            "source_commit": _source_commit(),
            "runtime": _audit_runtime(),
            "source_manifest": source,
            "source_manifest_sha256": source_hash,
            "reason": blocker,
            "weights": {
                "trained_checkpoint": {
                    "path": str(CHECKPOINT.relative_to(ROOT)),
                    "available": CHECKPOINT.exists(),
                    "sha256": _sha256(CHECKPOINT) if CHECKPOINT.exists() else None,
                },
                "layoutnet_fid": {
                    "path": str(VENDOR_LAYOUTNET.relative_to(ROOT)),
                    "available": VENDOR_LAYOUTNET.exists(),
                    "sha256": (
                        _sha256(VENDOR_LAYOUTNET) if VENDOR_LAYOUTNET.exists() else None
                    ),
                },
            },
            "downloads": {
                "trained_checkpoint": _asset_record(
                    CHECKPOINT,
                    CHECKPOINT_URL,
                    "models/layoutganpp/scripts/download_original_weights.py via GET",
                ),
                "layoutnet_fid": _asset_record(
                    VENDOR_LAYOUTNET,
                    LAYOUTNET_URL,
                    "official URL via GET",
                ),
            },
            "inputs": {
                "dataset": "magazine",
                "split": "test",
                "count": len(package_rows),
                "source_manifest_sha256": source_hash,
            },
            "evaluator": {
                "status": "not-run",
                "entry_points": {
                    "generate": _script_entry_point(VENDOR_ROOT / "generate.py"),
                    "eval": _script_entry_point(VENDOR_ROOT / "eval.py"),
                },
            },
            "prediction_files": {"vendor": None, "package": None},
            "per_system": {
                "vendor": {
                    "status": "not-run",
                    "prediction_count": None,
                    "out_of_bounds_counts": None,
                    "metrics": None,
                },
                "package": {
                    "status": "not-run",
                    "prediction_count": None,
                    "out_of_bounds_counts": None,
                    "metrics": None,
                },
            },
            "evaluator_source_commits": {
                "vendor_generate": vendor_source_commit,
                "vendor_eval": vendor_source_commit,
                "package": _source_commit(),
            },
        }
        evaluation_path.write_text(
            json.dumps(blocked_evaluation, indent=2, sort_keys=True) + "\n"
        )
        return _write(
            "s4-loader-eval",
            _record(
                "s4-loader-eval",
                started,
                result="PARTIAL",
                source_manifest=source,
                source_manifest_sha256=source_hash,
                loader_streams=loader_streams,
                evaluation_path_parity={
                    "status": "blocked",
                    "artifact": str(evaluation_path.relative_to(ROOT)),
                    "reason": blocker,
                },
                first_divergence=None,
            ),
        )

    vendor_pickle = evaluator_dir / "vendor-predictions.pkl"
    package_pickle = evaluator_dir / "package-predictions.pkl"
    input_file = evaluator_dir / "test-inputs.pt"
    vendor_command = [
        sys.executable,
        str((VENDOR_ROOT / "generate.py").resolve()),
        str(VENDOR_CHECKPOINT.resolve()),
        "--batch_size",
        str(VENDOR_BATCH_SIZE),
        "--seed",
        str(S4_EVALUATION_SEED),
        "--out_path",
        str(vendor_pickle.resolve()),
    ]
    recorded_vendor_command = shlex.join(_relative_command(vendor_command))
    vendor_environment = os.environ.copy()
    vendor_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    subprocess.run(
        vendor_command,
        cwd=VENDOR_WORK,
        check=True,
        env=vendor_environment,
    )
    package_model = (
        cast(LayoutGANPPModel, LayoutGANPPModel.from_pretrained(CONVERTED))
        .to(device)
        .eval()
    )
    package_loader = DataLoader(
        LayoutGANPPDataset(
            dataset_name="magazine",
            split="test",
            data_root=DATA_ROOT,
            synthetic_size=0,
            seed=0,
        ),
        batch_size=VENDOR_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_layoutganpp,
    )
    torch.manual_seed(S4_EVALUATION_SEED)
    package_predictions: list[
        tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]
    ] = []
    input_batches: list[dict[str, Shaped[torch.Tensor, "..."]]] = []
    with torch.no_grad():
        for batch in package_loader:
            labels = cast(torch.Tensor, batch["labels"]).to(device)
            mask = cast(torch.Tensor, batch["mask"]).to(device)
            bbox = cast(torch.Tensor, batch["bbox"]).to(device)
            latent = randn(labels.shape[0], labels.shape[1], LATENT_SIZE, device=device)
            output = package_model(latents=latent, labels=labels, padding_mask=~mask)
            for index in range(labels.shape[0]):
                valid = mask[index].bool()
                package_predictions.append(
                    (
                        output.bbox[index][valid].cpu().numpy(),
                        labels[index][valid].cpu().numpy(),
                    )
                )
            input_batches.append(
                {"bbox": bbox.cpu(), "labels": labels.cpu(), "mask": mask.cpu()}
            )
    package_pickle.write_bytes(pickle.dumps(package_predictions))
    torch.save({"names": package_names, "batches": input_batches}, input_file)
    vendor_predictions = cast(
        list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
        pickle.loads(vendor_pickle.read_bytes()),
    )
    package_predictions = cast(
        list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
        pickle.loads(package_pickle.read_bytes()),
    )
    predictions = {"vendor": vendor_predictions, "package": package_predictions}
    evaluator_outputs: dict[str, str] = {}
    evaluator_metrics: dict[str, dict[str, float]] = {}
    evaluator_commands: dict[str, str] = {}
    evaluator_command_argv: dict[str, list[str]] = {}
    for system, prediction_file in (
        ("vendor", vendor_pickle),
        ("package", package_pickle),
    ):
        command = [
            sys.executable,
            str((VENDOR_ROOT / "eval.py").resolve()),
            "magazine",
            str(prediction_file.resolve()),
            "--batch_size",
            str(VENDOR_BATCH_SIZE),
        ]
        evaluator = subprocess.run(
            command,
            cwd=VENDOR_WORK,
            text=True,
            capture_output=True,
            check=True,
            env=vendor_environment,
        )
        output = (evaluator.stdout + evaluator.stderr).replace(f"{ROOT}/", "")
        evaluator_outputs[system] = output
        evaluator_metrics[system] = _parse_vendor_metrics(output)
        evaluator_commands[system] = shlex.join(_relative_command(command))
        evaluator_command_argv[system] = command
        (evaluator_dir / f"{system}-evaluator.txt").write_text(output)

    maximum_difference, first_divergence = _compare_prediction_rows(
        vendor_predictions, package_predictions
    )
    out_of_bounds = {
        system: _out_of_bounds_counts(values) for system, values in predictions.items()
    }
    package_weight_files = sorted(CONVERTED.glob("*.safetensors"))
    if len(package_weight_files) != 1:
        raise RuntimeError(
            "converted Magazine checkpoint must contain one safetensors file"
        )
    vendor_source_commit = subprocess.check_output(
        ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    passed = maximum_difference <= TRACE_ATOL and first_divergence is None
    prediction_files = {
        "vendor": {
            "path": str(vendor_pickle.relative_to(ROOT)),
            "sha256": _sha256(vendor_pickle),
        },
        "package": {
            "path": str(package_pickle.relative_to(ROOT)),
            "sha256": _sha256(package_pickle),
        },
        "inputs": {
            "path": str(input_file.relative_to(ROOT)),
            "sha256": _sha256(input_file),
        },
    }
    checkpoint_args = cast(
        dict[str, JsonValue],
        torch.load(VENDOR_CHECKPOINT, map_location="cpu", weights_only=False)["args"],
    )
    evaluation_payload = {
        "status": "PASS" if passed else "FAIL",
        "source_commit": _source_commit(),
        "runtime": _audit_runtime(),
        "source_manifest": source,
        "source_manifest_sha256": source_hash,
        "weights": {
            "trained_checkpoint": {
                "path": str(CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(CHECKPOINT),
            },
            "vendor": {
                "path": str(VENDOR_CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(VENDOR_CHECKPOINT),
            },
            "package": {
                "path": str(package_weight_files[0].relative_to(ROOT)),
                "sha256": _sha256(package_weight_files[0]),
                "converted_directory": str(CONVERTED.relative_to(ROOT)),
            },
            "layoutnet_fid": {
                "path": str(VENDOR_LAYOUTNET.relative_to(ROOT)),
                "sha256": _sha256(VENDOR_LAYOUTNET),
            },
        },
        "downloads": {
            "trained_checkpoint": _asset_record(
                CHECKPOINT,
                CHECKPOINT_URL,
                "models/layoutganpp/scripts/download_original_weights.py via GET",
            ),
            "layoutnet_fid": _asset_record(
                VENDOR_LAYOUTNET,
                LAYOUTNET_URL,
                "official URL via GET",
            ),
        },
        "inputs": {
            "path": str(input_file.relative_to(ROOT)),
            "sha256": _sha256(input_file),
            "dataset": "magazine",
            "split": "test",
            "count": len(package_rows),
            "names_sha256": hashlib.sha256(
                "\n".join(package_names).encode()
            ).hexdigest(),
            "coordinate_frame": "original normalized xywh frame",
        },
        "evaluator_settings": {
            "generate_command": {
                "batch_size": _command_option(vendor_command, "--batch_size"),
                "latent_size": int(
                    cast(int | float | str, checkpoint_args["latent_size"])
                ),
                "sampling_seed": _command_option(vendor_command, "--seed"),
            },
            "eval_commands": {
                system: {
                    "batch_size": _command_option(
                        evaluator_command_argv[system], "--batch_size"
                    )
                    for system in evaluator_commands
                },
            },
        },
        "sampling_seeds": {"evaluation_seed": S4_EVALUATION_SEED},
        "evaluator": {
            "entry_points": {
                "generate": _script_entry_point(VENDOR_ROOT / "generate.py"),
                "eval": _script_entry_point(VENDOR_ROOT / "eval.py"),
            },
            "commands": evaluator_commands,
            "metrics_output_by_system": evaluator_outputs,
        },
        "prediction_files": prediction_files,
        "per_system": {
            system: {
                "prediction_count": len(values),
                "out_of_bounds_counts": out_of_bounds[system],
                "metrics": evaluator_metrics[system],
            }
            for system, values in predictions.items()
        },
        "evaluator_source_commits": {
            "vendor_generate": vendor_source_commit,
            "vendor_eval_for_vendor": vendor_source_commit,
            "vendor_eval_for_package": vendor_source_commit,
            "package": _source_commit(),
        },
        "prediction_comparison": {
            "passed": passed,
            "population": f"{len(package_rows)} TEST layouts and all valid element boxes",
            "max_abs_difference": maximum_difference,
            "limit": TRACE_ATOL,
            "first_divergence": first_divergence,
        },
    }
    evaluation_path.write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True) + "\n"
    )
    payload = _record(
        "s4-loader-eval",
        started,
        result="PASS" if passed else "FAIL",
        source_manifest=source,
        source_manifest_sha256=source_hash,
        loader_streams=loader_streams,
        evaluation_path_parity=str(evaluation_path.relative_to(ROOT)),
        weights={
            "source_checkpoint": {
                "path": str(CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(CHECKPOINT),
            },
            "vendor": {
                "path": str(VENDOR_CHECKPOINT.relative_to(ROOT)),
                "sha256": _sha256(VENDOR_CHECKPOINT),
            },
            "package": {
                "path": str(package_weight_files[0].relative_to(ROOT)),
                "sha256": _sha256(package_weight_files[0]),
                "converted_directory": str(CONVERTED.relative_to(ROOT)),
            },
        },
        loader={
            "package": "layoutganpp training dataset DataLoader + collate_layoutganpp",
            "test_split": "test",
            "batch_size": VENDOR_BATCH_SIZE,
            "shuffle": False,
            "vendor_processed_cache": {
                "path": str(
                    (
                        VENDOR_WORK / "data" / "dataset" / "magazine" / "processed"
                    ).relative_to(ROOT)
                ),
                "existed_before_clear": processed_cache_existed,
                "exists_after_vendor_load": (
                    VENDOR_WORK / "data" / "dataset" / "magazine" / "processed"
                ).exists(),
            },
        },
        evaluator={
            "entry_points": {
                "generate": _script_entry_point(VENDOR_ROOT / "generate.py"),
                "eval": _script_entry_point(VENDOR_ROOT / "eval.py"),
            },
            "generate_command": recorded_vendor_command,
            "commands_by_system": evaluator_commands,
            "metrics_output_by_system": evaluator_outputs,
        },
        sampling_seeds={"evaluation_seed": S4_EVALUATION_SEED},
        prediction_files=prediction_files,
        per_system={
            system: {
                "prediction_count": len(values),
                **out_of_bounds[system],
                "metrics": evaluator_metrics[system],
            }
            for system, values in predictions.items()
        },
        evaluator_source_commits={
            "vendor_generate": vendor_source_commit,
            "vendor_eval_for_vendor": vendor_source_commit,
            "vendor_eval_for_package": vendor_source_commit,
            "package_source": _source_commit(),
        },
        coordinate_frame="original normalized xywh frame",
        input_count=len(package_rows),
        input_names_sha256=hashlib.sha256(
            "\n".join(package_names).encode()
        ).hexdigest(),
        prediction_comparison={
            "passed": passed,
            "population": f"{len(package_rows)} TEST layouts and all valid element boxes",
            "max_abs_difference": maximum_difference,
            "limit": TRACE_ATOL,
            "first_divergence": first_divergence,
        },
        first_divergence=first_divergence,
    )
    return _write("s4-loader-eval", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "s0-static",
            "s1-fixed-batch",
            "s2-one-step",
            "s3-lockstep",
            "s4-loader-eval",
        ),
    )
    parser.add_argument("--steps", type=int, default=S3_STEPS)
    args = parser.parse_args()
    apply_determinism(DeterminismConfig(seed=INIT_SEED))
    if not torch.cuda.is_available() and args.stage in {
        "s3-lockstep",
        "s4-loader-eval",
    }:
        raise RuntimeError("GPU is required for S3 and S4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "s0-static":
        path = _stage_s0(device)
    elif args.stage == "s1-fixed-batch":
        path = _stage_s1(device)
    elif args.stage == "s2-one-step":
        path = _stage_s2(device)
    elif args.stage == "s3-lockstep":
        path = _stage_s3(device)
    else:
        path = _stage_s4(device)

    record = cast(dict[str, JsonValue], json.loads(path.read_text()))
    if record.get("result") != "PASS":
        raise SystemExit(f"{args.stage} wrote non-PASS result: {record.get('result')}")

    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
