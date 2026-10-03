"""Generate ordered LayoutGAN++ training-reproduction evidence."""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch
from jaxtyping import Float, Int, Shaped
from torch_geometric.data import Data
from torch.utils.data import DataLoader

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


def _optimizer_map(
    optimizer: torch.optim.Optimizer, module: torch.nn.Module
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    return {
        f"{name}.{state_name}": value.detach().clone()
        for name, parameter in module.named_parameters()
        for state_name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
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
    package_g = torch.optim.Adam(package.generator.parameters(), lr=1.0e-5)
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
        and sum(p.numel() for p in vendor_generator.parameters())
        == sum(p.numel() for p in package.generator.parameters())
        and sum(p.numel() for p in vendor_discriminator.parameters())
        == sum(p.numel() for p in package.discriminator.parameters())
        else "FAIL",
        source_manifest=".cache/layoutganpp/data/magazine/source-manifest.json",
        source_manifest_sha256=source_hash,
        source=source,
        initialization={
            "vendor_generator_state_sha256": _module_hash(vendor_generator),
            "package_generator_state_sha256": _module_hash(package.generator),
            "vendor_discriminator_state_sha256": _module_hash(vendor_discriminator),
            "package_discriminator_state_sha256": _module_hash(package.discriminator),
            "compared_as": "independently constructed state metadata; no weights copied",
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
            "defaults_equal": vendor_g.defaults == package_g.defaults,
            "state_empty": vendor_g.state_dict()["state"]
            == package_g.state_dict()["state"]
            == {},
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
            "rng_control": "one captured state restored before the package trace; no injected latent",
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
            "rng_control": "one restore between systems; no injected latent",
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


def _trajectory(
    device: torch.device,
    system: str,
    steps: int,
    seed: int,
    initial_generator: Mapping[str, Shaped[torch.Tensor, "..."]],
    initial_discriminator: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> list[dict[str, JsonValue]]:
    vendor_generator, vendor_discriminator, package = _build_models(
        device, copy_vendor_weights=False
    )
    if system == "vendor":
        generator, discriminator = vendor_generator, vendor_discriminator
    else:
        package.generator.load_state_dict(initial_generator, strict=True)
        package.discriminator.load_state_dict(initial_discriminator, strict=True)
        generator, discriminator = package.generator, package.discriminator
    generator.load_state_dict(initial_generator, strict=True)
    discriminator.load_state_dict(initial_discriminator, strict=True)
    generator.train()
    discriminator.train()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1.0e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1.0e-5)
    loader = _loader_stream(
        _vendor_loader("train", S3_BATCH_SEED)
        if system == "vendor"
        else _package_loader("train", S3_BATCH_SEED)
    )
    torch.manual_seed(seed)
    records: list[dict[str, JsonValue]] = []
    for step in range(steps):
        raw_batch = next(loader)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        trace = (
            _vendor_iteration(
                generator,
                discriminator,
                batch,
                optimizer_g,
                optimizer_d,
            )
            if system == "vendor"
            else run_gan_iteration(
                generator,
                discriminator,
                batch,
                optimizer_g,
                optimizer_d,
            )
        )
        records.append(
            {
                "step": step,
                "generator_loss": float(trace["generator_loss"].item()),
                "discriminator_loss": float(trace["discriminator_loss"].item()),
                "generator_gradient_sha256": hashlib.sha256(
                    "\n".join(
                        tensor_sha256(value)
                        for value in _gradient_map(generator).values()
                    ).encode()
                ).hexdigest(),
                "discriminator_gradient_sha256": hashlib.sha256(
                    "\n".join(
                        tensor_sha256(value)
                        for value in _gradient_map(discriminator).values()
                    ).encode()
                ).hexdigest(),
                "generator_optimizer_state_sha256": hashlib.sha256(
                    "\n".join(
                        tensor_sha256(value)
                        for value in _optimizer_map(optimizer_g, generator).values()
                    ).encode()
                ).hexdigest(),
                "discriminator_optimizer_state_sha256": hashlib.sha256(
                    "\n".join(
                        tensor_sha256(value)
                        for value in _optimizer_map(optimizer_d, discriminator).values()
                    ).encode()
                ).hexdigest(),
                "generator_learning_rate": optimizer_g.param_groups[0]["lr"],
                "discriminator_learning_rate": optimizer_d.param_groups[0]["lr"],
                "generator_parameter_sha256": _module_hash(generator),
                "discriminator_parameter_sha256": _module_hash(discriminator),
            }
        )
    return records


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
    for system in natural:
        for repeat in range(S3_REPEATS):
            natural[system].append(
                _trajectory(
                    device,
                    system,
                    S3_STEPS,
                    S3_LATENT_SEED + repeat,
                    initial_g,
                    initial_d,
                )
            )
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
        natural_control="each repeat run assigned one latent seed consumed by both systems through their own draw paths; no per-step RNG restore and no injected latent",
        natural_steps=S3_STEPS,
        batch_stream={
            "checked_steps": stream.checked_steps,
            "passed": stream.passed,
            "first_mismatch": stream.first_mismatch,
            "batch_seed": S3_BATCH_SEED,
        },
        latent_seeds=[S3_LATENT_SEED + repeat for repeat in range(S3_REPEATS)],
        latent_seed_policy={
            "per_repeat_run": [S3_LATENT_SEED + repeat for repeat in range(S3_REPEATS)],
            "systems_per_run": ["vendor", "package"],
            "same_seed_for_both_systems": True,
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


def _clear_vendor_processed_cache() -> None:
    processed = VENDOR_WORK / "data" / "dataset" / "magazine" / "processed"
    if processed.exists():
        shutil.rmtree(processed)


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
        invalid = (
            ((tensor < 0) | (tensor > 1)).any(dim=1)
            if tensor.numel()
            else tensor.new_zeros(0, dtype=torch.bool)
        )
        box_count += int(invalid.sum().item())
        layout_count += int(invalid.any().item())
    return {
        "out_of_bounds_box_count": box_count,
        "out_of_bounds_layout_count": layout_count,
    }


def _stage_s4(device: torch.device) -> Path:
    started = time.time()
    _require_previous("s4-loader-eval", "s3-lockstep")
    _clear_vendor_processed_cache()
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
            "evaluator_settings": {
                "batch_size": VENDOR_BATCH_SIZE,
                "default_batch_size": VENDOR_BATCH_SIZE,
                "latent_size": LATENT_SIZE,
                "sampling_seed": S4_EVALUATION_SEED,
                "model_mode": "eval",
                "coordinate_frame": "original normalized xywh frame",
            },
            "evaluator": {
                "status": "not-run",
                "vendor_generate": "vendor/const-layout/generate.py:main",
                "vendor_eval": "vendor/const-layout/eval.py:main",
                "vendor_metric_functions": [
                    "vendor/const-layout/metric.py:compute_alignment",
                    "vendor/const-layout/metric.py:compute_overlap",
                    "vendor/const-layout/metric.py:LayoutFID",
                ],
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
            "sampling_seeds": {"evaluation_seed": S4_EVALUATION_SEED},
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
        output = evaluator.stdout + evaluator.stderr
        evaluator_outputs[system] = output
        evaluator_metrics[system] = _parse_vendor_metrics(output)
        evaluator_commands[system] = " ".join(command)
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
            "batch_size": VENDOR_BATCH_SIZE,
            "default_batch_size": VENDOR_BATCH_SIZE,
            "latent_size": LATENT_SIZE,
            "sampling_seed": S4_EVALUATION_SEED,
            "model_mode": "eval",
            "shuffle": False,
        },
        "evaluator": {
            "vendor_generate": "vendor/const-layout/generate.py:main",
            "vendor_eval": "vendor/const-layout/eval.py:main",
            "commands": evaluator_commands,
            "metric_functions": [
                "vendor/const-layout/metric.py:compute_alignment",
                "vendor/const-layout/metric.py:compute_overlap",
                "vendor/const-layout/metric.py:LayoutFID",
            ],
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
            "vendor_generate": "vendor/const-layout/generate.py:39-66 DataLoader + torch_geometric.to_dense_batch",
            "vendor_eval": "vendor/const-layout/eval.py:44-63 and :99-133 DataLoader + torch_geometric.to_dense_batch",
            "package": "layoutganpp training dataset DataLoader + collate_layoutganpp",
            "test_split": "test",
            "batch_size": VENDOR_BATCH_SIZE,
            "shuffle": False,
            "vendor_processed_cache_cleared": True,
        },
        evaluator={
            "vendor_command": " ".join(vendor_command),
            "commands_by_system": evaluator_commands,
            "source": "vendor/const-layout/eval.py:main; one invocation per prediction file",
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
        metrics_recorded_by_vendor=True,
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
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
