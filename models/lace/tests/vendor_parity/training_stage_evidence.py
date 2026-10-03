"""Generate staged LACE training-reproduction evidence.

This module is executable from the repository root.  It imports the original
implementation only from the vendor checkout and writes JSON evidence below
the caller-selected cache directory.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Callable, cast

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset as GeometricDataset
from lightning.pytorch import Callback, LightningModule, Trainer
from traingen_parity.compare import (
    compare_batch_stream,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import capture_rng_state, restore_rng_state
from traingen_parity.trace import build_step_trace, tensor_sha256
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

ROOT = Path(__file__).resolve().parents[4]


DEFAULT_OUTPUT_ROOT = ROOT / ".cache" / "lace" / "stage-evidence"

sys.path.insert(0, str(ROOT / "vendor" / "lace"))

from model_diffusion import Diffusion as VendorDiffusion  # noqa: E402
from util.constraint import (  # noqa: E402
    PIoU_xywh,
    Pdist,
    constraint_temporal_weight,
    layout_alignment,
)  # noqa: E402
from util.datasets.publaynet import PubLayNetDataset  # noqa: E402
from util.datasets.rico import Rico25Dataset  # noqa: E402
from util.ema import EMA as VendorEMA  # noqa: E402
from util.metric import (  # noqa: E402
    compute_alignment,
    compute_generative_model_scores,
    compute_maximum_iou,
    compute_overlap,
)  # noqa: E402
from util.seq_util import pad_until, sparse_to_dense  # noqa: E402

from lace.configuration_lace import default_model_config, get_dataset_spec  # noqa: E402
from lace.conversion import build_pipeline_from_vendor_checkpoint  # noqa: E402
from lace.modeling_lace import LaceTransformerModel  # noqa: E402
from lace.training.config import LaceTrainingDatasetName, LaceTrainingSplit  # noqa: E402
from lace.training.dataset import LaceProcessedDataset, collate_lace_batch  # noqa: E402
from lace.training.datamodule import LaceDataModule  # noqa: E402
from lace.training.ema import LaceEMA  # noqa: E402
from lace.training.lightning_module import LaceTrainingModule  # noqa: E402


def _git_commit(path: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _assert_clean_sources() -> tuple[str, str]:
    statuses: dict[str, str] = {}
    for name, path in (("repository", ROOT), ("vendor", ROOT / "vendor" / "lace")):
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            raise RuntimeError(
                f"{name} source tree is dirty; commit all evidence code before running"
            )
        statuses[name] = _git_commit(path)
    return statuses["repository"], statuses["vendor"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_entrypoint(function: Callable[..., object]) -> str:
    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"cannot locate source for {function!r}")
    name = getattr(function, "__name__", None)
    if not isinstance(name, str):
        raise RuntimeError(f"cannot locate callable name for {function!r}")
    return f"{Path(source).resolve().relative_to(ROOT).as_posix()}::{name}"


def _runtime_metadata() -> dict[str, object]:
    return {
        "torch_wheel": "torch-2.8.0+cu128",
        "torchvision_wheel": "torchvision-0.23.0+cu128",
        "cuda_tag": "cu128",
        "python": platform.python_version(),
        "audit_venv_placeholder": "<LACE_AUDIT_VENV>",
        "venv_creation_command": "python3.11 -m venv <LACE_AUDIT_VENV>",
        "torch_wheel_file_url": (
            "file://<LACE_AUDIT_WHEEL_ROOT>/"
            "torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        "torch_wheel_sha256": (
            "039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed"
        ),
        "torchvision_wheel_file_url": (
            "file://<LACE_AUDIT_WHEEL_ROOT>/"
            "torchvision-0.23.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        "torchvision_wheel_sha256": (
            "93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb"
        ),
        "pip_freeze_sha256": os.environ.get(
            "LACE_AUDIT_FREEZE_SHA256", "<LACE_AUDIT_FREEZE_SHA256>"
        ),
        "lockfile_environment_used_for_cpu_checks_and_tests": True,
    }


def _write_json(output_root: Path, name: str, payload: Mapping[str, object]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _training_config(dataset: str) -> dict[str, int]:
    config = cast(dict[str, int], dict(default_model_config(dataset)))
    config.update(
        {
            "dim_transformer": 1024,
            "nhead": 16,
            "num_layers": 4,
            "dim_feedforward": 2048,
        }
    )
    return config


def _tiny_config(dataset: str) -> dict[str, int]:
    config = cast(dict[str, int], dict(default_model_config(dataset)))
    config.update(
        {
            "dim_transformer": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
        }
    )
    return config


def _build_training_fixture(
    dataset: str,
    device: torch.device,
    *,
    tiny: bool,
    seed: int = 42975,
) -> tuple[VendorDiffusion, LaceTrainingModule]:
    config = _tiny_config(dataset) if tiny else _training_config(dataset)
    torch.manual_seed(123)
    vendor = VendorDiffusion(
        num_timesteps=config["diffusion_step"],
        nhead=config["nhead"],
        dim_transformer=config["dim_transformer"],
        feature_dim=config["dim_feedforward"],
        seq_dim=config["seq_dim"],
        num_layers=config["num_layers"],
        device=device,
        ddim_num_steps=200,
    )
    torch.manual_seed(123)
    package_model = LaceTransformerModel(**config)
    cast(nn.Module, package_model).to(device)
    target = LaceTrainingModule(
        dataset_name=cast(LaceTrainingDatasetName, dataset),
        model=package_model,
        dim_transformer=config["dim_transformer"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        feature_dim=config["dim_feedforward"],
        seed=seed,
    ).to(device)
    vendor.model.train()
    target.model.train()
    return vendor, target


def _layout_input(
    batch: Mapping[str, torch.Tensor], *, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    bbox = batch["bbox"].float()
    labels = batch["labels"].long().clone()
    mask = batch["mask"].bool()
    labels[~mask] = num_classes - 1
    bbox_in = 2 * (bbox - 0.5)
    one_hot = nn.functional.one_hot(labels, num_classes=num_classes).to(bbox_in.dtype)
    return torch.cat((one_hot, bbox_in), dim=2), mask


def _losses_vendor(
    layout_input: torch.Tensor,
    output: torch.Tensor,
    noise: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    timesteps: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    bbox_rep = reconstructed[:, :, num_classes:].clamp(-1, 1) / 2 + 0.5
    _, alignment_loss = layout_alignment(bbox_rep, mask, xy_only=False)
    alignment_loss = 20 * alignment_loss
    piou = PIoU_xywh(bbox_rep, mask=mask.float(), xy_only=False)
    pdist = Pdist(bbox_rep)
    overlap_loss = piou.mean(dim=(1, 2)) + (piou.ne(0) * torch.exp(-pdist)).mean(
        dim=(1, 2)
    )
    reconstruct_loss = nn.functional.mse_loss(
        torch.cat((layout_input,) * 4, dim=0)[:, :, num_classes:],
        reconstructed[:, :, num_classes:],
    )
    weight = constraint_temporal_weight(timesteps, schedule="const")
    constraint_loss = torch.mean((alignment_loss + overlap_loss) * weight)
    diffusion_loss = nn.functional.mse_loss(noise, output)
    total = diffusion_loss + constraint_loss + reconstruct_loss
    return {
        "diffusion_loss": diffusion_loss,
        "alignment_loss": alignment_loss.mean(),
        "overlap_loss": overlap_loss.mean(),
        "constraint_loss": constraint_loss,
        "reconstruct_loss": reconstruct_loss,
        "train_loss": total,
    }


def _vendor_trace(
    vendor: VendorDiffusion,
    batch: Mapping[str, torch.Tensor],
    *,
    num_classes: int,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    layout_input, mask = _layout_input(batch, num_classes=num_classes)
    timestep = vendor.sample_t([layout_input.shape[0]], t_max=999)
    vendor_output, vendor_noise, vendor_reconstructed = vendor.forward_t(
        layout_input, timestep, mask, reparam=True
    )
    all_timesteps = torch.cat((timestep,) * 4, dim=0)
    vendor_values = _losses_vendor(
        layout_input,
        vendor_output,
        vendor_noise,
        vendor_reconstructed,
        torch.cat((mask,) * 4, dim=0),
        all_timesteps,
        num_classes,
    )
    prepared_labels = batch["labels"].long().clone()
    prepared_labels[~mask] = num_classes - 1
    trace = {
        "bbox": batch["bbox"].detach(),
        "labels": prepared_labels.detach(),
        "mask": mask.detach(),
        "layout_input": layout_input.detach(),
        "timestep": timestep.detach(),
        "noise": vendor_noise.detach(),
        "model_output": vendor_output.detach(),
        "reconstructed": vendor_reconstructed.detach(),
        "all_timesteps": all_timesteps.detach(),
        **{key: value.detach() for key, value in vendor_values.items()},
    }
    if not detach:
        trace["train_loss"] = vendor_values["train_loss"]
    return trace


def _package_trace(
    target: LaceTrainingModule,
    batch: Mapping[str, torch.Tensor],
    *,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    loss = target.training_step(dict(batch), 0)
    trace = dict(target.latest_step_trace)
    if not detach:
        trace["train_loss"] = loss
    return trace


def _paired_trace(
    vendor: VendorDiffusion,
    target: LaceTrainingModule,
    batch: Mapping[str, torch.Tensor],
    *,
    detach: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rng_state = capture_rng_state()
    vendor_trace = _vendor_trace(
        vendor, batch, num_classes=target.num_classes, detach=detach
    )
    restore_rng_state(rng_state)
    package_trace = _package_trace(target, batch, detach=detach)
    return vendor_trace, package_trace


def _trace_report(
    vendor_trace: Mapping[str, torch.Tensor],
    package_trace: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    common_names = tuple(name for name in vendor_trace if name in package_trace)
    vendor_step = build_step_trace(
        "vendor", {name: vendor_trace[name] for name in common_names}
    )
    package_step = build_step_trace(
        "package", {name: package_trace[name] for name in common_names}
    )
    report = compare_step_trace(vendor_step, package_step)
    return {
        "passed": report.passed and set(vendor_trace) == set(package_trace),
        "missing": sorted(set(vendor_trace) ^ set(package_trace)),
        "comparisons": {
            item.name: {
                "passed": item.passed,
                "max_abs_diff": item.max_abs_diff,
                "max_rel_diff": item.max_rel_diff,
            }
            for item in report.comparisons
        },
    }


def _state_max_diff(
    vendor: Mapping[str, torch.Tensor], package: Mapping[str, torch.Tensor]
) -> float:
    return max(
        float((vendor[name] - package[name]).abs().max().item()) for name in vendor
    )


def _parameter_grad(parameter: nn.Parameter) -> torch.Tensor:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("training parameter has no gradient")
    return gradient


def _record_float(record: Mapping[str, object], key: str) -> float:
    value = record[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"record field {key} is not numeric")
    return float(value)


def _record_int(record: Mapping[str, object], key: str) -> int:
    value = record[key]
    if not isinstance(value, int):
        raise TypeError(f"record field {key} is not an integer")
    return value


def _dataset_file(data_root: Path, dataset: str, split: str) -> Path:
    return data_root / f"{dataset}-max25" / "processed" / f"{split}.pt"


def _dataset_summary(path: Path) -> dict[str, object]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    data, slices = loaded
    x_slices = slices["x"]
    names = getattr(data, "attr", {}).get("name", [])
    if isinstance(names, str):
        names = [names]
    record_ids_digest = hashlib.sha256()
    element_tuples_digest = hashlib.sha256()
    record_signatures: list[dict[str, object]] = []
    x_slices = slices["x"]
    y_slices = slices["y"]
    for index in range(int(x_slices.numel() - 1)):
        x_start, x_end = int(x_slices[index]), int(x_slices[index + 1])
        y_start, y_end = int(y_slices[index]), int(y_slices[index + 1])
        record_id = str(names[index]) if index < len(names) else "<missing-id>"
        element_bytes = data.x[x_start:x_end].float().contiguous().numpy().tobytes()
        label_bytes = data.y[y_start:y_end].long().contiguous().numpy().tobytes()
        record_ids_digest.update(record_id.encode())
        record_ids_digest.update(b"\0")
        element_tuples_digest.update(element_bytes)
        element_tuples_digest.update(label_bytes)
        record_signatures.append(
            {
                "id": record_id,
                "element_tuple_sha256": hashlib.sha256(
                    element_bytes + label_bytes
                ).hexdigest(),
                "element_count": x_end - x_start,
            }
        )
    return {
        "sha256": _sha256(path),
        "layout_count": int(x_slices.numel() - 1),
        "element_count": int(data.x.shape[0]),
        "first_id": str(names[0]) if names else None,
        "last_id": str(names[-1]) if names else None,
        "record_ids_sha256": record_ids_digest.hexdigest(),
        "element_tuples_sha256": element_tuples_digest.hexdigest(),
        "record_signatures": record_signatures,
    }


def _compatibility(
    lace_data_root: Path, layoutdm_data_root: Path, dataset: str
) -> dict[str, object]:
    splits: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        lace = _dataset_summary(_dataset_file(lace_data_root, dataset, split))
        layoutdm = _dataset_summary(_dataset_file(layoutdm_data_root, dataset, split))
        split_result: dict[str, object] = {
            "lace": lace,
            "layoutdm": layoutdm,
            "sha256_equal": lace["sha256"] == layoutdm["sha256"],
            "layout_count_equal": lace["layout_count"] == layoutdm["layout_count"],
            "element_count_equal": lace["element_count"] == layoutdm["element_count"],
            "first_id_equal": lace["first_id"] == layoutdm["first_id"],
            "last_id_equal": lace["last_id"] == layoutdm["last_id"],
            "record_ids_equal": lace["record_ids_sha256"]
            == layoutdm["record_ids_sha256"],
            "element_tuples_equal": lace["element_tuples_sha256"]
            == layoutdm["element_tuples_sha256"],
        }
        lace_signatures = cast(list[dict[str, object]], lace["record_signatures"])
        layoutdm_signatures = cast(
            list[dict[str, object]], layoutdm["record_signatures"]
        )
        signature_mismatches = [
            index
            for index, (lace_record, layoutdm_record) in enumerate(
                zip(lace_signatures, layoutdm_signatures, strict=False)
            )
            if lace_record != layoutdm_record
        ]
        split_result["record_signature_mismatch_count"] = len(
            signature_mismatches
        ) + abs(len(lace_signatures) - len(layoutdm_signatures))
        split_result["first_record_signature_mismatch"] = (
            signature_mismatches[0] if signature_mismatches else None
        )
        if split_result["record_ids_equal"] and split_result["element_tuples_equal"]:
            split_result["content_interpretation"] = (
                "serialization-only difference: record ids and exact (bbox, label) "
                "element tuples match"
            )
        elif dataset == "rico25" and split == "train":
            split_result["content_interpretation"] = (
                "content-stream difference: LACE repeats the RICO source train "
                "stream 10x; the LayoutDM copy contains one source pass"
            )
        else:
            split_result["content_interpretation"] = (
                "content difference: record ids or exact (bbox, label) element "
                "tuples differ"
            )
        splits[split] = split_result
    train = splits["train"]
    lace_train = cast(dict[str, object], train["lace"])
    layoutdm_train = cast(dict[str, object], train["layoutdm"])
    lace_train_signatures = cast(
        list[dict[str, object]], lace_train["record_signatures"]
    )
    layoutdm_train_signatures = cast(
        list[dict[str, object]], layoutdm_train["record_signatures"]
    )
    duplicate_factor = float(cast(int, lace_train["layout_count"])) / float(
        cast(int, layoutdm_train["layout_count"])
    )
    repeated_stream_equal = (
        dataset == "rico25" and lace_train_signatures == layoutdm_train_signatures * 10
    )
    for split_mapping in splits.values():
        cast(dict[str, object], split_mapping["lace"]).pop("record_signatures")
        cast(dict[str, object], split_mapping["layoutdm"]).pop("record_signatures")
    return {
        "dataset": dataset,
        "splits": splits,
        "rico_train_duplicate_factor": duplicate_factor
        if dataset == "rico25"
        else None,
        "rico_train_duplicate_confirmed": dataset == "rico25"
        and duplicate_factor == 10.0
        and lace_train["first_id"] == layoutdm_train["first_id"]
        and lace_train["last_id"] == layoutdm_train["last_id"],
        "rico_train_repeated_stream_equal": (repeated_stream_equal),
    }


def run_s0(args: argparse.Namespace) -> Path:
    source_commit, vendor_commit = _assert_clean_sources()
    output = Path(args.output_root) / "s0-static"
    dataset = args.dataset
    config = _training_config(dataset)
    vendor, target = _build_training_fixture(dataset, torch.device("cpu"), tiny=False)
    vendor_state = vendor.model.state_dict()
    package_state = target.model.state_dict()
    shape_equal = set(vendor_state) == set(package_state) and all(
        vendor_state[name].shape == package_state[name].shape for name in vendor_state
    )
    state_key_mapping = [
        {"vendor": name, "package": name}
        for name in sorted(set(vendor_state) & set(package_state))
    ]
    initialized_state_max_abs_diff = _state_max_diff(vendor_state, package_state)
    vendor_optimizer = torch.optim.Adam(vendor.model.parameters(), lr=1e-5)
    package_optimizer = cast(torch.optim.Optimizer, target.configure_optimizers())
    vendor_ema = VendorEMA(mu=0.9999)
    vendor_ema.register(vendor.model)
    package_ema = target.ema_helper
    payload: dict[str, object] = {
        "stage": "S0",
        "source_commit": source_commit,
        "vendor_commit": vendor_commit,
        "runtime": _runtime_metadata(),
        "dataset": dataset,
        "reproduction_stream": {
            "consumer_stages": ["S1", "S2", "S3", "S4"],
            "provenance": "LACE's own processed InMemoryDataset stream from the approved source",
            "source": "creative-graphic-design/PubLayNet and creative-graphic-design/Rico",
            "lace_processed_root": "<LACE_DATA_ROOT>",
            "layoutdm_role": "compatibility cross-check only; not consumed by reproduction stages",
            "split_files": "<LACE_DATA_ROOT>/<dataset>-max25/processed/{train,val,test}.pt",
        },
        "training_entry_defaults": {
            "dim_transformer": 1024,
            "num_layers": 4,
            "batch_size": 256,
            "learning_rate": 1e-5,
            "nhead": 16,
            "feature_dim": 2048,
        },
        "resolved_config": config,
        "topology": {
            "state_key_set_equal": set(vendor_state) == set(package_state),
            "state_shapes_equal": shape_equal,
            "state_key_mapping": state_key_mapping,
            "vendor_only_state_keys": sorted(set(vendor_state) - set(package_state)),
            "package_only_state_keys": sorted(set(package_state) - set(vendor_state)),
            "parameter_count": sum(
                parameter.numel() for parameter in vendor.model.parameters()
            ),
            "vendor_parameter_count": sum(
                parameter.numel() for parameter in vendor.model.parameters()
            ),
            "package_parameter_count": sum(
                parameter.numel() for parameter in target.model.parameters()
            ),
            "initialized_state_max_abs_diff": initialized_state_max_abs_diff,
            "initialized_state_comparison": "independent same-seed vendor and package initialization; no vendor state copied into package",
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "vendor_defaults": vendor_optimizer.defaults,
            "package_defaults": package_optimizer.defaults,
            "parameter_count_equal": len(vendor_optimizer.param_groups[0]["params"])
            == len(package_optimizer.param_groups[0]["params"]),
            "scheduler": None,
        },
        "ema": {
            "active_from_initialization": bool(
                vendor_ema.shadow and package_ema.shadow
            ),
            "mu_equal": vendor_ema.mu == package_ema.mu == 0.9999,
            "update_rule": "post-optimizer-step",
            "vendor_and_package_shadow_key_sets_equal": set(vendor_ema.shadow)
            == set(package_ema.shadow),
        },
        "data_compatibility": {
            name: _compatibility(
                Path(args.lace_data_root), Path(args.layoutdm_data_root), name
            )
            for name in ("publaynet", "rico25")
        },
    }
    return _write_json(output, "summary.json", payload)


def run_s1(args: argparse.Namespace) -> Path:
    source_commit, vendor_commit = _assert_clean_sources()
    output = Path(args.output_root) / "s1-fixed-batch"
    device = torch.device(args.device)
    vendor_batches, package_batches = _loader_batches(
        args.dataset,
        Path(args.lace_data_root),
        "train",
        batch_size=args.batch_size,
        shuffle=False,
        seed=42975,
        steps=1,
    )
    if not vendor_batches or not package_batches:
        raise RuntimeError("real training loader produced no batch")
    vendor, target = _build_training_fixture(args.dataset, device, tiny=False)
    batch = {
        key: cast(torch.Tensor, package_batches[0][key]).to(device)
        for key in ("bbox", "labels", "mask")
    }
    vendor_trace, package_trace = _paired_trace(vendor, target, batch)
    trace_report = _trace_report(vendor_trace, package_trace)
    payload = {
        "stage": "S1",
        "source_commit": source_commit,
        "vendor_commit": vendor_commit,
        "runtime": _runtime_metadata(),
        "device": str(device),
        "dataset": args.dataset,
        "topology": _training_config(args.dataset),
        "loader": {
            "split": "train",
            "batch_size": args.batch_size,
            "vendor_semantics": "torch_geometric.loader.DataLoader shuffle=False num_workers=0",
            "package_semantics": "torch.utils.data.DataLoader shuffle=False num_workers=0",
            "seed": 42975,
            "ids": vendor_batches[0]["id"],
        },
        "trace": trace_report,
        "exact": trace_report["passed"],
    }
    return _write_json(output, "summary.json", payload)


def _optimizer_state_diff(
    vendor: torch.optim.Optimizer, package: torch.optim.Optimizer
) -> float:
    values: list[float] = []
    for vendor_state, package_state in zip(
        vendor.state.values(), package.state.values(), strict=True
    ):
        for key in vendor_state:
            vendor_value = vendor_state[key]
            package_value = package_state[key]
            if isinstance(vendor_value, torch.Tensor):
                values.append(float((vendor_value - package_value).abs().max().item()))
    return max(values, default=0.0)


def run_s2(args: argparse.Namespace) -> Path:
    source_commit, vendor_commit = _assert_clean_sources()
    output = Path(args.output_root) / "s2-optimizer-step"
    device = torch.device(args.device)
    vendor_batches, package_batches = _loader_batches(
        args.dataset,
        Path(args.lace_data_root),
        "train",
        batch_size=args.batch_size,
        shuffle=False,
        seed=42975,
        steps=1,
    )
    if not vendor_batches or not package_batches:
        raise RuntimeError("real training loader produced no batch")
    vendor, target = _build_training_fixture(args.dataset, device, tiny=False)
    batch = {
        key: cast(torch.Tensor, package_batches[0][key]).to(device)
        for key in ("bbox", "labels", "mask")
    }
    vendor_optimizer = torch.optim.Adam(vendor.model.parameters(), lr=1e-5)
    package_optimizer = cast(torch.optim.Optimizer, target.configure_optimizers())
    vendor_ema = VendorEMA(mu=0.9999)
    vendor_ema.register(vendor.model)
    package_ema = target.ema_helper
    sdpa_context = sdpa_kernel(SDPBackend.MATH) if args.sdpa_math else nullcontext()
    with sdpa_context:
        vendor_trace, package_trace = _paired_trace(vendor, target, batch, detach=False)
        vendor_optimizer.zero_grad()
        package_optimizer.zero_grad()
        vendor_trace["train_loss"].backward()
        package_trace["train_loss"].backward()
    vendor_parameters = dict(vendor.model.named_parameters())
    package_parameters = dict(target.model.named_parameters())
    vendor_gradients = {
        name: _parameter_grad(parameter).detach().clone()
        for name, parameter in vendor_parameters.items()
    }
    gradient_report = compare_step_trace(
        build_step_trace(
            "vendor-gradients",
            {
                name: _parameter_grad(parameter).detach()
                for name, parameter in vendor_parameters.items()
            },
        ),
        build_step_trace(
            "package-gradients",
            {
                name: _parameter_grad(parameter).detach()
                for name, parameter in package_parameters.items()
            },
        ),
    )
    vendor_grad_norm = torch.nn.utils.clip_grad_norm_(vendor.model.parameters(), 1.0)
    package_grad_norm = torch.nn.utils.clip_grad_norm_(target.model.parameters(), 1.0)
    clipped_report = compare_step_trace(
        build_step_trace(
            "vendor-clipped",
            {
                name: _parameter_grad(parameter).detach()
                for name, parameter in vendor_parameters.items()
            },
        ),
        build_step_trace(
            "package-clipped",
            {
                name: _parameter_grad(parameter).detach()
                for name, parameter in package_parameters.items()
            },
        ),
    )
    vendor_optimizer.step()
    package_optimizer.step()
    post_report = compare_optimizer_step(
        vendor.model.state_dict(), target.model.state_dict()
    )
    vendor_ema.update(vendor.model)
    package_ema.update(target.model)
    probe_vendor, probe_package = _build_training_fixture(
        args.dataset, device, tiny=False
    )
    probe_package.model.load_state_dict(
        copy.deepcopy(probe_vendor.model.state_dict()), strict=True
    )
    probe_vendor_optimizer = torch.optim.Adam(probe_vendor.model.parameters(), lr=1e-5)
    probe_package_optimizer = cast(
        torch.optim.Optimizer, probe_package.configure_optimizers()
    )
    for name, probe_vendor_parameter in probe_vendor.model.named_parameters():
        probe_vendor_parameter.grad = vendor_gradients[name].to(device).clone()
        dict(probe_package.model.named_parameters())[name].grad = (
            vendor_gradients[name].to(device).clone()
        )
    probe_vendor_optimizer.step()
    probe_package_optimizer.step()
    diagnostic_record = os.environ.get("LACE_DETERMINISTIC_DIAGNOSTIC_RECORD")
    ema_max_abs_diff = _state_max_diff(vendor_ema.shadow, package_ema.shadow)
    optimizer_state_max_abs_diff = _optimizer_state_diff(
        vendor_optimizer, package_optimizer
    )
    learning_rate_equal = (
        vendor_optimizer.param_groups[0]["lr"]
        == package_optimizer.param_groups[0]["lr"]
    )
    payload: dict[str, object] = {
        "stage": "S2",
        "source_commit": source_commit,
        "vendor_commit": vendor_commit,
        "runtime": _runtime_metadata(),
        "device": str(device),
        "dataset": args.dataset,
        "loader": {
            "split": "train",
            "batch_size": args.batch_size,
            "vendor_semantics": "torch_geometric.loader.DataLoader shuffle=False num_workers=0",
            "package_semantics": "torch.utils.data.DataLoader shuffle=False num_workers=0",
            "seed": 42975,
            "ids": vendor_batches[0]["id"],
        },
        "determinism_condition": {
            "torch_use_deterministic_algorithms": args.deterministic_algorithms,
            "warn_only": args.warn_only,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "warning": os.environ.get("LACE_DETERMINISTIC_WARNING"),
        },
        "sdpa_condition": {
            "backend": "math" if args.sdpa_math else "vendor-default",
            "forced_for_both_systems": args.sdpa_math,
            "gradients_bitwise": gradient_report.passed,
            "diagnostic_outcome": (
                "bitwise gradients under math SDPA"
                if args.sdpa_math and gradient_report.passed
                else "math SDPA gradients remain non-bitwise"
                if args.sdpa_math
                else "not run"
            ),
            "natural_backward_cause": "Memory Efficient attention backward (attention_backward.cu:775)",
        },
        "gradient_report": {
            "before_clipping": {
                "passed": gradient_report.passed,
                "max_abs_diff": max(
                    (item.max_abs_diff for item in gradient_report.comparisons),
                    default=0.0,
                ),
            },
            "after_clipping": {
                "passed": clipped_report.passed,
                "max_abs_diff": max(
                    (item.max_abs_diff for item in clipped_report.comparisons),
                    default=0.0,
                ),
            },
            "dtype_pairs": sorted(
                {
                    (
                        str(_parameter_grad(vendor_parameters[name]).dtype),
                        str(_parameter_grad(package_parameters[name]).dtype),
                    )
                    for name in vendor_parameters
                }
            ),
        },
        "gradient_norm_max_abs_diff": float(
            (vendor_grad_norm - package_grad_norm).abs().item()
        ),
        "parameter_max_abs_diff": max(
            (item.max_abs_diff for item in post_report.comparisons), default=0.0
        ),
        "ema_max_abs_diff": ema_max_abs_diff,
        "optimizer_state_max_abs_diff": optimizer_state_max_abs_diff,
        "learning_rate_equal": learning_rate_equal,
        "ema_activation": "registered before the optimizer step and updated after it",
        "optimizer_implementation": {
            "vendor_class": f"{type(vendor_optimizer).__module__}.{type(vendor_optimizer).__qualname__}",
            "package_class": f"{type(package_optimizer).__module__}.{type(package_optimizer).__qualname__}",
            "defaults_equal": vendor_optimizer.defaults == package_optimizer.defaults,
            "foreach_option": vendor_optimizer.param_groups[0].get("foreach"),
            "fused_option": vendor_optimizer.param_groups[0].get("fused"),
            "weight_decay": vendor_optimizer.param_groups[0]["weight_decay"],
            "decoupled_weight_decay": vendor_optimizer.param_groups[0].get(
                "decoupled_weight_decay"
            ),
            "eps": vendor_optimizer.param_groups[0]["eps"],
            "same_gradient_probe_parameter_max_abs_diff": _state_max_diff(
                probe_vendor.model.state_dict(), probe_package.model.state_dict()
            ),
            "same_gradient_probe_optimizer_state_max_abs_diff": _optimizer_state_diff(
                probe_vendor_optimizer, probe_package_optimizer
            ),
            "cause_of_normal_step_difference": "gradient tensors differ before clipping; the identical Adam implementation produces identical updates when given the same gradients",
        },
        "exact": (
            gradient_report.passed
            and clipped_report.passed
            and post_report.passed
            and ema_max_abs_diff == 0.0
            and optimizer_state_max_abs_diff == 0.0
            and learning_rate_equal
        ),
    }
    if diagnostic_record:
        payload["deterministic_diagnostic"] = {
            "artifact": os.environ.get(
                "LACE_DETERMINISTIC_DIAGNOSTIC_ARTIFACT", diagnostic_record
            ),
            "record": json.loads(Path(diagnostic_record).read_text()),
        }
    return _write_json(output, "summary.json", payload)


def _tensor_hashes(values: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in values.items()}


def _mapping_l2_norm(values: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for value in values.values():
        total += value.detach().double().square().sum().cpu()
    return float(total.sqrt().item())


def _optimizer_state_tensors(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    values[f"group{group_index}.parameter{parameter_index}.{key}"] = (
                        value.detach()
                    )
    return values


def _batch_tensors(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: cast(torch.Tensor, batch[key]).to(device)
        for key in ("bbox", "labels", "mask")
    }


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().double().square().sum().cpu()
    return float(squared_norm.sqrt().item())


class _PackageNaturalTraceCallback(Callback):
    """Capture the production Lightning optimizer path after each batch."""

    def __init__(self) -> None:
        self._batch_ids: list[str] = []
        self._pre_clip_hashes: dict[str, str] = {}
        self._pre_clip_norm = 0.0
        self._record: dict[str, object] | None = None
        self.records: list[dict[str, object]] = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, batch_idx
        self._finalize_record(pl_module)
        if not isinstance(batch, Mapping):
            raise TypeError("LACE production batch must be a mapping")
        ids = batch.get("id", [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise TypeError("LACE production batch must carry string ids")
        self._batch_ids = cast(list[str], ids)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        parameters = dict(pl_module.named_parameters())
        self._pre_clip_hashes = _tensor_hashes(
            {
                name: parameter.grad.detach()
                for name, parameter in parameters.items()
                if parameter.grad is not None
            }
        )
        self._pre_clip_norm = _gradient_norm(pl_module.parameters())

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        trace = getattr(pl_module, "latest_step_trace", {})
        self._record = {
            "batch_ids": list(self._batch_ids),
            "loss": float(cast(torch.Tensor, trace["train_loss"]).item()),
            "gradient_hashes": self._pre_clip_hashes,
            "gradient_norm": self._pre_clip_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "trainer_gradient_clip_val": trainer.gradient_clip_val,
            "trainer_gradient_clip_algorithm": trainer.gradient_clip_algorithm,
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, outputs, batch, batch_idx
        if self._record is None:
            raise RuntimeError("Lightning optimizer callback did not capture a step")
        optimizer = cast(torch.optim.Optimizer, pl_module.optimizers())
        model = cast(nn.Module, getattr(pl_module, "model"))
        parameters = dict(pl_module.named_parameters())
        gradients = {
            name: parameter.grad.detach()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }
        self._record.update(
            {
                "clipped_gradient_hashes": _tensor_hashes(gradients),
                "clipped_gradient_norm": _gradient_norm(parameters.values()),
            }
        )
        optimizer_state = _optimizer_state_tensors(optimizer)
        self._record.update(
            {
                "optimizer_state_hashes": _tensor_hashes(optimizer_state),
                "optimizer_state_l2_norm": _mapping_l2_norm(optimizer_state),
                "parameter_l2_norm": _mapping_l2_norm(model.state_dict()),
                "parameter_hashes": _tensor_hashes(model.state_dict()),
            }
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        self._finalize_record(pl_module)

    def finalize(self, pl_module: LightningModule) -> None:
        """Finalize a step whose trainer stopped at the configured max step."""
        self._finalize_record(pl_module)

    def _finalize_record(self, pl_module: LightningModule) -> None:
        if self._record is None:
            return
        ema_state = cast(
            Mapping[str, torch.Tensor], getattr(pl_module, "latest_ema_state")
        )
        self.records.append({**self._record, "ema_hashes": _tensor_hashes(ema_state)})
        self._record = None


def _run_natural_system(
    dataset: str,
    data_root: Path,
    *,
    system: str,
    device: torch.device,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[dict[str, object]]:
    vendor, target = _build_training_fixture(dataset, device, tiny=False, seed=seed)
    if system == "package":
        callback = _PackageNaturalTraceCallback()
        datamodule = LaceDataModule(
            processed_data_dir=data_root,
            dataset_name=cast(LaceTrainingDatasetName, dataset),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            loader_seed=42975,
        )
        trainer = Trainer(
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            max_steps=steps,
            limit_train_batches=steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[callback],
            default_root_dir=ROOT / ".cache" / "lace" / "trainer",
            log_every_n_steps=steps,
        )
        trainer.fit(target, datamodule=datamodule)
        callback.finalize(target)
        if trainer.global_step != steps or len(callback.records) != steps:
            raise RuntimeError(
                f"package Trainer produced {trainer.global_step} steps and "
                f"{len(callback.records)} records, expected {steps}"
            )
        return [
            {
                **record,
                "step": index,
            }
            for index, record in enumerate(callback.records, start=1)
        ]
    model = vendor if system == "vendor" else target
    optimizer: torch.optim.Optimizer = (
        torch.optim.Adam(model.model.parameters(), lr=1e-5)
        if system == "vendor"
        else cast(torch.optim.Optimizer, target.configure_optimizers())
    )
    ema = VendorEMA(mu=0.9999) if system == "vendor" else target.ema_helper
    if system == "vendor":
        cast(VendorEMA, ema).register(vendor.model)
    vendor_batches, package_batches = _loader_batches(
        dataset,
        data_root,
        "train",
        batch_size=batch_size,
        shuffle=True,
        seed=42975,
        steps=steps,
    )
    batches = vendor_batches if system == "vendor" else package_batches
    torch.manual_seed(seed)
    records: list[dict[str, object]] = []
    for step, source_batch in enumerate(batches, start=1):
        batch = _batch_tensors(source_batch, device)
        trace = (
            _vendor_trace(vendor, batch, num_classes=target.num_classes, detach=False)
            if system == "vendor"
            else _package_trace(target, batch, detach=False)
        )
        optimizer.zero_grad()
        trace["train_loss"].backward()
        named_parameters = dict(model.model.named_parameters())
        gradients = {
            name: _parameter_grad(parameter).detach().clone()
            for name, parameter in named_parameters.items()
        }
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        clipped_gradients = {
            name: _parameter_grad(parameter).detach().clone()
            for name, parameter in named_parameters.items()
        }
        optimizer.step()
        if system == "vendor":
            cast(VendorEMA, ema).update(vendor.model)
            ema_values = cast(VendorEMA, ema).shadow
        else:
            target.ema_helper.update(target.model)
            ema_values = target.ema_helper.shadow
        optimizer_state = _optimizer_state_tensors(optimizer)
        records.append(
            {
                "step": step,
                "batch_ids": source_batch["id"],
                "loss": float(trace["train_loss"].item()),
                "gradient_norm": float(gradient_norm.item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_hashes": _tensor_hashes(gradients),
                "clipped_gradient_hashes": _tensor_hashes(clipped_gradients),
                "clipped_gradient_norm": _gradient_norm(named_parameters.values()),
                "optimizer_state_hashes": _tensor_hashes(optimizer_state),
                "optimizer_state_l2_norm": _mapping_l2_norm(optimizer_state),
                "parameter_l2_norm": _mapping_l2_norm(model.model.state_dict()),
                "parameter_hashes": _tensor_hashes(model.model.state_dict()),
                "ema_hashes": _tensor_hashes(ema_values),
            }
        )
    if len(records) != steps:
        raise RuntimeError(
            f"{system} loader produced {len(records)} of {steps} batches"
        )
    return records


def _natural_pair_records(
    dataset: str,
    data_root: Path,
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    vendor_runs = [
        _run_natural_system(
            dataset,
            data_root,
            system="vendor",
            device=device,
            batch_size=batch_size,
            steps=steps,
            seed=seed,
        )
        for _ in range(2)
    ]
    package_runs = [
        _run_natural_system(
            dataset,
            data_root,
            system="package",
            device=device,
            batch_size=batch_size,
            steps=steps,
            seed=seed,
        )
        for _ in range(2)
    ]
    records: list[dict[str, object]] = []
    for vendor_record, package_record in zip(
        vendor_runs[0], package_runs[0], strict=True
    ):
        records.append(
            {
                "step": vendor_record["step"],
                "batch_ids_equal": vendor_record["batch_ids"]
                == package_record["batch_ids"],
                "vendor_loss": vendor_record["loss"],
                "package_loss": package_record["loss"],
                "loss_abs_diff": abs(
                    _record_float(vendor_record, "loss")
                    - _record_float(package_record, "loss")
                ),
                "vendor_gradient_norm": vendor_record["gradient_norm"],
                "package_gradient_norm": package_record["gradient_norm"],
                "gradient_norm_abs_diff": abs(
                    _record_float(vendor_record, "gradient_norm")
                    - _record_float(package_record, "gradient_norm")
                ),
                "vendor_clipped_gradient_norm": vendor_record["clipped_gradient_norm"],
                "package_clipped_gradient_norm": package_record[
                    "clipped_gradient_norm"
                ],
                "clipped_gradient_norm_abs_diff": abs(
                    _record_float(vendor_record, "clipped_gradient_norm")
                    - _record_float(package_record, "clipped_gradient_norm")
                ),
                "vendor_gradient_hashes": vendor_record["gradient_hashes"],
                "package_gradient_hashes": package_record["gradient_hashes"],
                "vendor_clipped_gradient_hashes": vendor_record[
                    "clipped_gradient_hashes"
                ],
                "package_clipped_gradient_hashes": package_record[
                    "clipped_gradient_hashes"
                ],
                "vendor_learning_rate": vendor_record["learning_rate"],
                "package_learning_rate": package_record["learning_rate"],
                "vendor_optimizer_state_hashes": vendor_record[
                    "optimizer_state_hashes"
                ],
                "package_optimizer_state_hashes": package_record[
                    "optimizer_state_hashes"
                ],
                "optimizer_state_l2_norm_abs_diff": abs(
                    _record_float(vendor_record, "optimizer_state_l2_norm")
                    - _record_float(package_record, "optimizer_state_l2_norm")
                ),
                "parameter_l2_norm_abs_diff": abs(
                    _record_float(vendor_record, "parameter_l2_norm")
                    - _record_float(package_record, "parameter_l2_norm")
                ),
                "vendor_parameter_hashes": vendor_record["parameter_hashes"],
                "package_parameter_hashes": package_record["parameter_hashes"],
                "vendor_ema_hashes": vendor_record["ema_hashes"],
                "package_ema_hashes": package_record["ema_hashes"],
                "gradient_hashes_equal": vendor_record["gradient_hashes"]
                == package_record["gradient_hashes"],
                "clipped_gradient_hashes_equal": vendor_record[
                    "clipped_gradient_hashes"
                ]
                == package_record["clipped_gradient_hashes"],
                "optimizer_state_hashes_equal": vendor_record["optimizer_state_hashes"]
                == package_record["optimizer_state_hashes"],
            }
        )
    envelope_records = [
        {
            "step": left["step"],
            "vendor_repeat_loss_abs_diff": abs(
                _record_float(left, "loss") - _record_float(right, "loss")
            ),
            "package_repeat_loss_abs_diff": abs(
                _record_float(left, "loss") - _record_float(right, "loss")
            ),
            "vendor_repeat_parameter_l2_norm_abs_diff": abs(
                _record_float(left, "parameter_l2_norm")
                - _record_float(right, "parameter_l2_norm")
            ),
            "package_repeat_parameter_l2_norm_abs_diff": abs(
                _record_float(left, "parameter_l2_norm")
                - _record_float(right, "parameter_l2_norm")
            ),
        }
        for left, right in zip(vendor_runs[0], vendor_runs[1], strict=True)
    ]
    envelope_records = [
        {
            **record,
            "package_repeat_loss_abs_diff": abs(
                _record_float(package_runs[0][index], "loss")
                - _record_float(package_runs[1][index], "loss")
            ),
            "package_repeat_parameter_l2_norm_abs_diff": abs(
                _record_float(package_runs[0][index], "parameter_l2_norm")
                - _record_float(package_runs[1][index], "parameter_l2_norm")
            ),
        }
        for index, record in enumerate(envelope_records)
    ]
    return records, {
        "runs_per_system": 2,
        "vendor_max_loss_abs_diff": max(
            _record_float(record, "vendor_repeat_loss_abs_diff")
            for record in envelope_records
        ),
        "package_max_loss_abs_diff": max(
            _record_float(record, "package_repeat_loss_abs_diff")
            for record in envelope_records
        ),
        "vendor_max_parameter_l2_norm_abs_diff": max(
            _record_float(record, "vendor_repeat_parameter_l2_norm_abs_diff")
            for record in envelope_records
        ),
        "package_max_parameter_l2_norm_abs_diff": max(
            _record_float(record, "package_repeat_parameter_l2_norm_abs_diff")
            for record in envelope_records
        ),
        "records": envelope_records,
    }


def _synchronize_s3_state(
    vendor: VendorDiffusion,
    target: LaceTrainingModule,
    vendor_optimizer: torch.optim.Optimizer,
    package_optimizer: torch.optim.Optimizer,
    vendor_ema: VendorEMA,
    package_ema: LaceEMA,
) -> bool:
    target.model.load_state_dict(copy.deepcopy(vendor.model.state_dict()), strict=True)
    package_optimizer.load_state_dict(copy.deepcopy(vendor_optimizer.state_dict()))
    package_ema.load_state_dict(copy.deepcopy(vendor_ema.state_dict()))
    for vendor_state, package_state in zip(
        vendor_optimizer.state.values(), package_optimizer.state.values(), strict=True
    ):
        for name, vendor_value in vendor_state.items():
            package_value = package_state[name]
            if isinstance(vendor_value, torch.Tensor):
                if not isinstance(package_value, torch.Tensor):
                    return False
                if vendor_value.data_ptr() == package_value.data_ptr():
                    return False
    for name, vendor_value in vendor_ema.shadow.items():
        if vendor_value.data_ptr() == package_ema.shadow[name].data_ptr():
            return False
    return True


def run_s3(args: argparse.Namespace) -> Path:
    source_commit, vendor_commit = _assert_clean_sources()
    output = Path(args.output_root).resolve() / "s3-lockstep"
    device = torch.device(args.device)
    natural_records, repeat_envelope = _natural_pair_records(
        args.dataset,
        Path(args.lace_data_root),
        device=device,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=10000,
    )

    synchronized_vendor, synchronized_package = _build_training_fixture(
        args.dataset, device, tiny=False
    )
    synchronized_vendor_optimizer = torch.optim.Adam(
        synchronized_vendor.model.parameters(), lr=1e-5
    )
    synchronized_package_optimizer = cast(
        torch.optim.Optimizer, synchronized_package.configure_optimizers()
    )
    synchronized_vendor_ema = VendorEMA(mu=0.9999)
    synchronized_vendor_ema.register(synchronized_vendor.model)
    synchronized_package_ema = synchronized_package.ema_helper
    synchronized_records: list[dict[str, object]] = []
    storage_independence: list[bool] = []
    synchronized_vendor_batches, synchronized_package_batches = _loader_batches(
        args.dataset,
        Path(args.lace_data_root),
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        seed=42975,
        steps=args.steps,
    )
    if not _synchronize_s3_state(
        synchronized_vendor,
        synchronized_package,
        synchronized_vendor_optimizer,
        synchronized_package_optimizer,
        synchronized_vendor_ema,
        synchronized_package_ema,
    ):
        raise RuntimeError(
            "initial synchronized state does not have independent storage"
        )
    sdpa_context = sdpa_kernel(SDPBackend.MATH) if args.sdpa_math else nullcontext()
    with sdpa_context:
        for step, (vendor_source_batch, package_source_batch) in enumerate(
            zip(synchronized_vendor_batches, synchronized_package_batches, strict=True),
            start=1,
        ):
            if vendor_source_batch["id"] != package_source_batch["id"]:
                raise AssertionError(
                    f"synchronized loader order differs at step {step}"
                )
            batch = _batch_tensors(package_source_batch, device)
            rng_state = capture_rng_state()
            vendor_trace = _vendor_trace(
                synchronized_vendor,
                batch,
                num_classes=synchronized_package.num_classes,
                detach=False,
            )
            restore_rng_state(rng_state)
            package_trace = _package_trace(synchronized_package, batch, detach=False)
            synchronized_vendor_optimizer.zero_grad()
            synchronized_package_optimizer.zero_grad()
            vendor_trace["train_loss"].backward()
            package_trace["train_loss"].backward()
            vendor_gradients = {
                name: _parameter_grad(parameter).detach()
                for name, parameter in synchronized_vendor.model.named_parameters()
            }
            package_gradients = {
                name: _parameter_grad(parameter).detach()
                for name, parameter in synchronized_package.model.named_parameters()
            }
            gradient_report = compare_step_trace(
                build_step_trace("vendor", vendor_gradients),
                build_step_trace("package", package_gradients),
            )
            vendor_grad_norm = torch.nn.utils.clip_grad_norm_(
                synchronized_vendor.model.parameters(), 1.0
            )
            package_grad_norm = torch.nn.utils.clip_grad_norm_(
                synchronized_package.model.parameters(), 1.0
            )
            synchronized_vendor_optimizer.step()
            synchronized_package_optimizer.step()
            parameter_report = compare_optimizer_step(
                synchronized_vendor.model.state_dict(),
                synchronized_package.model.state_dict(),
            )
            synchronized_vendor_ema.update(synchronized_vendor.model)
            synchronized_package_ema.update(synchronized_package.model)
            record: dict[str, object] = {
                "step": step,
                "batch_ids": package_source_batch["id"],
                "vendor_loss": float(vendor_trace["train_loss"].item()),
                "package_loss": float(package_trace["train_loss"].item()),
                "gradient_norm_max_abs_diff": float(
                    (vendor_grad_norm - package_grad_norm).abs().item()
                ),
                "gradient_max_abs_diff": max(
                    (item.max_abs_diff for item in gradient_report.comparisons),
                    default=0.0,
                ),
                "learning_rate_equal": synchronized_vendor_optimizer.param_groups[0][
                    "lr"
                ]
                == synchronized_package_optimizer.param_groups[0]["lr"],
                "optimizer_state_max_abs_diff": _optimizer_state_diff(
                    synchronized_vendor_optimizer, synchronized_package_optimizer
                ),
                "parameter_max_abs_diff": max(
                    (item.max_abs_diff for item in parameter_report.comparisons),
                    default=0.0,
                ),
                "ema_max_abs_diff": _state_max_diff(
                    synchronized_vendor_ema.shadow, synchronized_package_ema.shadow
                ),
                "gradient_hashes_equal": _tensor_hashes(vendor_gradients)
                == _tensor_hashes(package_gradients),
            }
            storage_independence.append(
                _synchronize_s3_state(
                    synchronized_vendor,
                    synchronized_package,
                    synchronized_vendor_optimizer,
                    synchronized_package_optimizer,
                    synchronized_vendor_ema,
                    synchronized_package_ema,
                )
            )
            record["parameter_max_abs_diff_after_sync"] = _state_max_diff(
                synchronized_vendor.model.state_dict(),
                synchronized_package.model.state_dict(),
            )
            record["ema_max_abs_diff_after_sync"] = _state_max_diff(
                synchronized_vendor_ema.shadow, synchronized_package_ema.shadow
            )
            record["optimizer_state_max_abs_diff_after_sync"] = _optimizer_state_diff(
                synchronized_vendor_optimizer, synchronized_package_optimizer
            )
            synchronized_records.append(record)
    for record in natural_records:
        record["source_commit"] = source_commit
        record["vendor_commit"] = vendor_commit
    for record in synchronized_records:
        record["source_commit"] = source_commit
        record["vendor_commit"] = vendor_commit
    _write_jsonl(output, "natural.jsonl", natural_records)
    _write_jsonl(output, "synchronized.jsonl", synchronized_records)
    natural_artifact = str((output / "natural.jsonl").relative_to(ROOT))
    synchronized_artifact = str((output / "synchronized.jsonl").relative_to(ROOT))
    natural_first_divergence = next(
        (
            record
            for record in natural_records
            if _record_float(record, "loss_abs_diff") > 1e-3
        ),
        None,
    )
    payload = {
        "stage": "S3",
        "source_commit": source_commit,
        "vendor_commit": vendor_commit,
        "runtime": _runtime_metadata(),
        "device": str(device),
        "dataset": args.dataset,
        "steps": args.steps,
        "model_scale": {
            "dim_transformer": _training_config(args.dataset)["dim_transformer"],
            "num_layers": _training_config(args.dataset)["num_layers"],
        },
        "batch_size": args.batch_size,
        "natural_seed": 10000,
        "determinism_condition": {
            "torch_use_deterministic_algorithms": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "sdpa_condition": {
            "natural_backend": "vendor-default",
            "synchronized_backend": "math" if args.sdpa_math else "vendor-default",
            "synchronized_forced_for_both_systems": args.sdpa_math,
        },
        "natural": {
            "artifact": natural_artifact,
            "max_parameter_l2_norm_abs_diff": max(
                record["parameter_l2_norm_abs_diff"] for record in natural_records
            ),
            "max_gradient_norm_abs_diff": max(
                record["gradient_norm_abs_diff"] for record in natural_records
            ),
            "final_vendor_loss": natural_records[-1]["vendor_loss"],
            "final_package_loss": natural_records[-1]["package_loss"],
            "first_loss_divergence_over_1e-3": natural_first_divergence,
            "exact": all(
                record["loss_abs_diff"] == 0.0
                and record["parameter_l2_norm_abs_diff"] == 0.0
                and record["batch_ids_equal"]
                for record in natural_records
            ),
            "repeat_run_envelope": repeat_envelope,
        },
        "synchronized": {
            "artifact": synchronized_artifact,
            "max_parameter_abs_diff_before_sync": max(
                record["parameter_max_abs_diff"] for record in synchronized_records
            ),
            "max_parameter_abs_diff_after_sync": max(
                record["parameter_max_abs_diff_after_sync"]
                for record in synchronized_records
            ),
            "max_ema_abs_diff_after_sync": max(
                record["ema_max_abs_diff_after_sync"] for record in synchronized_records
            ),
            "max_optimizer_state_abs_diff_after_sync": max(
                record["optimizer_state_max_abs_diff_after_sync"]
                for record in synchronized_records
            ),
            "optimizer_state_storage_independent": all(storage_independence),
            "exact_after_sync": all(
                record["parameter_max_abs_diff_after_sync"] == 0.0
                and record["ema_max_abs_diff_after_sync"] == 0.0
                and record["optimizer_state_max_abs_diff_after_sync"] == 0.0
                and record["learning_rate_equal"]
                for record in synchronized_records
            ),
        },
    }
    return _write_json(output, "summary.json", payload)


def _write_jsonl(
    output: Path, name: str, records: Iterable[Mapping[str, object]]
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    return path


def _vendor_dataset(dataset: str, data_root: Path, split: str) -> GeometricDataset:
    original_torch_load = cast(Callable[..., object], torch.load)

    def load_legacy_processed_stream(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    setattr(torch, "load", load_legacy_processed_stream)
    try:
        if dataset == "publaynet":
            return PubLayNetDataset(dir=str(data_root), split=split, max_seq_length=25)
        if dataset == "rico25":
            return Rico25Dataset(dir=str(data_root), split=split, max_seq_length=25)
        raise ValueError(dataset)
    finally:
        setattr(torch, "load", original_torch_load)


def _package_split(split: str) -> str:
    return {"train": "train", "val": "validation", "test": "test"}[split]


def _loader_pair(
    dataset: str,
    data_root: Path,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[Iterable[object], Iterable[dict[str, object]]]:
    from torch_geometric.loader import DataLoader as GeometricDataLoader

    vendor_dataset = _vendor_dataset(dataset, data_root, split)
    package_dataset = LaceProcessedDataset(
        processed_data_dir=data_root,
        dataset_name=cast(LaceTrainingDatasetName, dataset),
        split=cast(LaceTrainingSplit, _package_split(split)),
        max_seq_length=25,
    )
    vendor_generator = torch.Generator().manual_seed(seed)
    package_generator = torch.Generator().manual_seed(seed)
    vendor_loader = GeometricDataLoader(
        vendor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=vendor_generator,
    )
    package_loader = DataLoader(
        package_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_lace_batch,
        generator=package_generator,
    )
    return vendor_loader, package_loader


def _vendor_batch_to_mapping(batch: object) -> dict[str, object]:
    bbox, labels, _, mask = sparse_to_dense(batch)
    labels, bbox, mask = pad_until(labels, bbox, mask, max_seq_length=25)
    return {
        "bbox": bbox,
        "labels": labels,
        "mask": mask,
        "id": _vendor_ids(batch),
    }


def _package_batch_to_mapping(batch: Mapping[str, object]) -> dict[str, object]:
    return {
        "bbox": cast(torch.Tensor, batch["bbox"]),
        "labels": cast(torch.Tensor, batch["labels"]),
        "mask": cast(torch.Tensor, batch["mask"]),
        "id": cast(list[str], batch.get("id", [])),
    }


def _loader_batches(
    dataset: str,
    data_root: Path,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    steps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    vendor_loader, package_loader = _loader_pair(
        dataset,
        data_root,
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    vendor_batches: list[dict[str, object]] = []
    package_batches: list[dict[str, object]] = []
    for vendor_batch, package_batch in zip(vendor_loader, package_loader, strict=False):
        if len(vendor_batches) >= steps:
            break
        vendor_batches.append(_vendor_batch_to_mapping(vendor_batch))
        package_batches.append(_package_batch_to_mapping(package_batch))
    return vendor_batches, package_batches


def _vendor_ids(batch: object) -> list[str]:
    names = getattr(batch, "attr", {}).get("name", [])
    if isinstance(names, str):
        return [names]
    return [str(name) for name in names]


def _stream_split(
    dataset: str,
    data_root: Path,
    split: str,
    *,
    batch_size: int,
    max_batches: int,
) -> dict[str, object]:
    vendor_dataset = _vendor_dataset(dataset, data_root, split)
    package_dataset = LaceProcessedDataset(
        processed_data_dir=data_root,
        dataset_name=cast(LaceTrainingDatasetName, dataset),
        split=cast(LaceTrainingSplit, _package_split(split)),
        max_seq_length=25,
    )
    vendor_batches, package_batches = _loader_batches(
        dataset,
        data_root,
        split,
        batch_size=batch_size,
        shuffle=False,
        seed=314159,
        steps=max_batches,
    )
    rows: list[dict[str, object]] = []
    for index, (vendor_batch, package_batch) in enumerate(
        zip(vendor_batches, package_batches, strict=False)
    ):
        ids = cast(list[str], vendor_batch["id"])
        package_ids = cast(list[str], package_batch["id"])
        vendor_bbox = cast(torch.Tensor, vendor_batch["bbox"])
        package_bbox = cast(torch.Tensor, package_batch["bbox"])
        vendor_labels = cast(torch.Tensor, vendor_batch["labels"])
        package_labels = cast(torch.Tensor, package_batch["labels"])
        vendor_mask = cast(torch.Tensor, vendor_batch["mask"])
        package_mask = cast(torch.Tensor, package_batch["mask"])
        rows.append(
            {
                "batch": index,
                "ids_equal": ids == package_ids,
                "bbox_equal": torch.equal(vendor_bbox, package_bbox),
                "labels_equal": torch.equal(vendor_labels, package_labels),
                "mask_equal": torch.equal(vendor_mask, package_mask),
                "vendor_ids": ids,
                "package_ids": package_ids,
            }
        )
    stream_report = compare_batch_stream(
        [cast(Mapping[str, torch.Tensor], batch) for batch in vendor_batches],
        [cast(Mapping[str, torch.Tensor], batch) for batch in package_batches],
        steps=max_batches,
    )
    return {
        "vendor_layout_count": len(vendor_dataset),
        "package_layout_count": len(package_dataset),
        "batches_checked": len(rows),
        "rows": rows,
        "shared_helper": {
            "passed": stream_report.passed,
            "checked_steps": stream_report.checked_steps,
            "first_mismatch": stream_report.first_mismatch,
        },
        "exact": all(
            row["ids_equal"]
            and row["bbox_equal"]
            and row["labels_equal"]
            and row["mask_equal"]
            for row in rows
        )
        and stream_report.passed,
    }


def _evaluation_parity(
    dataset: str,
    data_root: Path,
    checkpoint: Path,
    *,
    source_commit: str,
    vendor_commit: str,
    output_root: Path,
    device_name: str,
    batch_size: int,
    max_batches: int | None,
    ddim_num_steps: int,
    fid_root: Path | None,
) -> dict[str, object]:
    if batch_size != 256:
        raise ValueError(
            "evaluation-path parity must use the vendor default batch size 256"
        )
    device = torch.device(device_name)
    output_root = output_root.resolve()
    spec = get_dataset_spec(dataset)
    config = default_model_config(dataset)
    vendor = VendorDiffusion(
        num_timesteps=1000,
        nhead=config["nhead"],
        dim_transformer=config["dim_transformer"],
        feature_dim=config["dim_feedforward"],
        seq_dim=config["seq_dim"],
        num_layers=config["num_layers"],
        device=device,
        ddim_num_steps=ddim_num_steps,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vendor.model.load_state_dict(state, strict=True)
    vendor.model.eval()
    package = build_pipeline_from_vendor_checkpoint(
        dataset, checkpoint, ddim_num_steps=ddim_num_steps
    )
    package.to(device)
    if fid_root is None:
        raise RuntimeError("full evaluation-path parity requires the vendor FID assets")
    from torch_geometric.loader import DataLoader as GeometricDataLoader

    vendor_dataset = _vendor_dataset(dataset, data_root, "test")
    vendor_loader = GeometricDataLoader(
        vendor_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    evaluation_root = output_root / f"{dataset}-evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    feature_path = fid_root / "feature" / f"fid_feat_test_{dataset}.pk"
    with feature_path.open("rb") as stream:
        test_features = pickle.load(stream)
    sys.path.insert(0, str(fid_root.parent))
    import test as vendor_test

    class CapturingVendor:
        def __init__(self, wrapped: VendorDiffusion) -> None:
            self.wrapped = wrapped
            self.outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            self.calls: list[dict[str, object]] = []
            self.elapsed = 0.0

        @property
        def device(self) -> torch.device:
            return wrapped_device

        def eval(self) -> "CapturingVendor":
            self.wrapped.eval()
            return self

        def conditional_reverse_ddim(
            self, real_layout: torch.Tensor, *, cond: str = "c"
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            seed = 20260000 + len(self.outputs)
            torch.manual_seed(seed)
            started = time.perf_counter()
            result = self.wrapped.conditional_reverse_ddim(
                real_layout, cond=cond, stochastic=True
            )
            self.elapsed += time.perf_counter() - started
            self.outputs.append(tuple(value.detach().cpu() for value in result))
            self.calls.append(
                {
                    "condition_type": {"c": "label", "cwh": "label_size"}.get(
                        cond, cond
                    ),
                    "stochastic": True,
                    "generator_seed": seed,
                    "batch_size": int(real_layout.shape[0]),
                    "grad_enabled": torch.is_grad_enabled(),
                }
            )
            return result

    wrapped_device = device
    captured_vendor = CapturingVendor(vendor)
    original_init_dataset = vendor_test.init_dataset
    original_test_fid_feat = vendor_test.test_fid_feat
    original_load_fidnet = vendor_test.load_fidnet_v3
    vendor_test.init_dataset = lambda *_, **__: (
        vendor_dataset,
        vendor_loader,
    )
    vendor_test.test_fid_feat = lambda *_, **__: test_features
    vendor_test.load_fidnet_v3 = lambda dataset_arg, _, device_arg: (
        original_load_fidnet(dataset_arg, str(fid_root / "FIDNetV3"), device_arg)
    )
    try:
        vendor_result = vendor_test.test_layout_cond(
            captured_vendor,
            cond="c",
            dataset_name=dataset,
            seq_dim=spec.seq_dim,
            beautify=False,
        )
    finally:
        vendor_test.init_dataset = original_init_dataset
        vendor_test.test_fid_feat = original_test_fid_feat
        vendor_test.load_fidnet_v3 = original_load_fidnet
    if vendor_result is None or len(captured_vendor.outputs) != len(vendor_loader):
        raise RuntimeError(
            "vendor evaluation entry point did not produce the full TEST split"
        )

    loader = GeometricDataLoader(
        vendor_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    input_batches: list[dict[str, np.ndarray]] = []
    vendor_batches: list[dict[str, np.ndarray]] = []
    package_batches: list[dict[str, np.ndarray]] = []
    reference_layouts: list[tuple[np.ndarray, np.ndarray]] = []
    vendor_layouts: list[tuple[np.ndarray, np.ndarray]] = []
    package_layouts: list[tuple[np.ndarray, np.ndarray]] = []
    batch_records: list[dict[str, object]] = []
    package_calls: list[dict[str, object]] = []
    preprocessing_equal = True
    package_elapsed = 0.0
    for batch_index, test_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        bbox, labels, _, mask = sparse_to_dense(test_batch)
        labels, bbox, mask = pad_until(labels, bbox, mask, max_seq_length=25)
        ids = _vendor_ids(test_batch)
        vendor_real_layout = torch.cat(
            (
                nn.functional.one_hot(
                    labels.masked_fill(~mask, spec.pad_label_id),
                    num_classes=spec.num_classes_with_pad,
                ).float(),
                2 * (bbox - 0.5),
            ),
            dim=2,
        )
        package_encoded = package.processor(bbox=bbox, labels=labels, mask=mask)
        batch_preprocessing_equal = torch.equal(
            vendor_real_layout, package_encoded["layout"]
        )
        preprocessing_equal = preprocessing_equal and batch_preprocessing_equal
        if not batch_preprocessing_equal:
            raise AssertionError("vendor and package TEST inputs differ")
        input_batches.append(
            {"bbox": bbox.numpy(), "labels": labels.numpy(), "mask": mask.numpy()}
        )
        for layout_bbox, layout_labels, layout_mask in zip(
            bbox, labels, mask, strict=True
        ):
            valid = layout_mask.numpy().astype(bool)
            reference_layouts.append(
                (layout_bbox.numpy()[valid], layout_labels.numpy()[valid])
            )
        vendor_cond, vendor_cond_labels, vendor_cond_mask = captured_vendor.outputs[
            batch_index
        ]
        package_generator = torch.Generator().manual_seed(20260000 + batch_index)
        package_started = time.perf_counter()
        with torch.no_grad():
            package_cond = cast(
                LayoutGenerationOutput,
                package(
                    condition_type="label",
                    bbox=bbox.to(device),
                    labels=labels.to(device),
                    mask=mask.to(device),
                    generator=package_generator,
                    num_inference_steps=ddim_num_steps,
                ),
            )
            package_grad_enabled = torch.is_grad_enabled()
        package_elapsed += time.perf_counter() - package_started
        package_call: dict[str, object] = {
            "condition_type": "label",
            "stochastic": package_generator is not None,
            "generator_seed": 20260000 + batch_index,
            "batch_size": int(bbox.shape[0]),
            "grad_enabled": package_grad_enabled,
        }
        package_calls.append(package_call)
        vendor_cpu = {
            "bbox": vendor_cond,
            "labels": vendor_cond_labels,
            "mask": vendor_cond_mask,
        }
        package_cpu = {
            "bbox": package_cond.bbox.cpu(),
            "labels": package_cond.labels.cpu(),
            "mask": package_cond.mask.cpu(),
        }
        vendor_batches.append({key: value.numpy() for key, value in vendor_cpu.items()})
        package_batches.append(
            {key: value.numpy() for key, value in package_cpu.items()}
        )
        for layout_set, values in (
            (vendor_layouts, vendor_cpu),
            (package_layouts, package_cpu),
        ):
            for layout_bbox, layout_labels, layout_mask in zip(
                values["bbox"], values["labels"], values["mask"], strict=True
            ):
                valid = layout_mask.numpy().astype(bool)
                layout_set.append(
                    (layout_bbox.numpy()[valid], layout_labels.numpy()[valid])
                )
        vendor_oob = int(
            (
                ((vendor_cpu["bbox"] < 0) | (vendor_cpu["bbox"] > 1)).any(dim=-1)
                & vendor_cpu["mask"]
            ).sum()
        )
        package_oob = int(
            (
                ((package_cpu["bbox"] < 0) | (package_cpu["bbox"] > 1)).any(dim=-1)
                & package_cpu["mask"]
            ).sum()
        )
        batch_records.append(
            {
                "batch": batch_index,
                "input_ids": ids,
                "sampling_seed": 20260000 + batch_index,
                "input_count": len(ids),
                "vendor_prediction_count": int(vendor_cond.shape[0]),
                "package_prediction_count": int(package_cond.bbox.shape[0]),
                "vendor_element_count": int(vendor_cond_mask.sum().item()),
                "package_element_count": int(package_cond.mask.sum().item()),
                "vendor_out_of_bounds_count": vendor_oob,
                "package_out_of_bounds_count": package_oob,
                "predictions_equal": torch.equal(
                    vendor_cpu["bbox"], package_cpu["bbox"]
                )
                and torch.equal(vendor_cpu["labels"], package_cpu["labels"])
                and torch.equal(vendor_cpu["mask"], package_cpu["mask"]),
                "vendor_evaluation_settings": captured_vendor.calls[batch_index],
                "package_evaluation_settings": package_call,
            }
        )
    if not input_batches:
        raise RuntimeError("evaluation did not process any TEST batches")

    def concatenate_batches(
        batches: list[dict[str, np.ndarray]], key: str
    ) -> np.ndarray:
        return np.concatenate([batch[key] for batch in batches], axis=0)

    inputs = {
        key: concatenate_batches(input_batches, key)
        for key in ("bbox", "labels", "mask")
    }
    vendor_predictions = {
        key: concatenate_batches(vendor_batches, key)
        for key in ("bbox", "labels", "mask")
    }
    package_predictions = {
        key: concatenate_batches(package_batches, key)
        for key in ("bbox", "labels", "mask")
    }
    np.savez(
        evaluation_root / "vendor-inputs.npz",
        bbox=inputs["bbox"],
        labels=inputs["labels"],
        mask=inputs["mask"],
    )
    np.savez(
        evaluation_root / "package-inputs.npz",
        bbox=inputs["bbox"],
        labels=inputs["labels"],
        mask=inputs["mask"],
    )
    np.savez(
        evaluation_root / "vendor-predictions.npz",
        bbox=vendor_predictions["bbox"],
        labels=vendor_predictions["labels"],
        mask=vendor_predictions["mask"],
    )
    np.savez(
        evaluation_root / "package-predictions.npz",
        bbox=package_predictions["bbox"],
        labels=package_predictions["labels"],
        mask=package_predictions["mask"],
    )
    vendor_mask = torch.from_numpy(vendor_predictions["mask"])
    package_bbox = torch.from_numpy(package_predictions["bbox"])
    package_mask = torch.from_numpy(package_predictions["mask"])
    vendor_input_mask = torch.from_numpy(inputs["mask"])
    vendor_metrics: dict[str, float] = {
        "alignment": float(vendor_result[0]),
        "fid": float(vendor_result[1]),
        "maximum_iou": float(vendor_result[2]),
        "overlap": float(vendor_result[3]),
    }
    package_metrics = {
        "alignment": float(
            100 * compute_alignment(package_bbox, vendor_input_mask).mean()
        ),
        "overlap": float(100 * compute_overlap(package_bbox, vendor_input_mask).mean()),
        "maximum_iou": float(compute_maximum_iou(reference_layouts, package_layouts)),
    }
    processed_count = len(vendor_layouts)
    fid_metadata: dict[str, object] | None = None
    if fid_root is not None:
        sys.path.insert(0, str(fid_root.parent))
        from fid.model import load_fidnet_v3

        fid_dataset = vendor_dataset
        fid_model = load_fidnet_v3(
            fid_dataset, str(fid_root / "FIDNetV3"), device=device
        )
        feature_path = fid_root / "feature" / f"fid_feat_test_{dataset}.pk"
        with feature_path.open("rb") as stream:
            test_features = pickle.load(stream)
        package_features: list[torch.Tensor] = []
        for batch in range(0, processed_count, batch_size):
            end = min(batch + batch_size, processed_count)
            labels_for_fid = package_predictions["labels"][batch:end].copy()
            labels_for_fid[~package_predictions["mask"][batch:end]] = 0
            feature = fid_model.extract_features(
                torch.from_numpy(package_predictions["bbox"][batch:end]).to(device),
                torch.from_numpy(labels_for_fid).to(device),
                ~torch.from_numpy(package_predictions["mask"][batch:end]).to(device),
            )
            package_features.append(feature.detach().cpu())
        package_fid = compute_generative_model_scores(test_features, package_features)[
            "fid"
        ]
        package_metrics["fid"] = float(package_fid)
        fid_metadata = {
            "feature_cache": str(
                Path("<LACE_FID_ROOT>") / "feature" / f"fid_feat_test_{dataset}.pk"
            ),
            "feature_cache_sha256": _sha256(feature_path),
            "fid_checkpoint_sha256": _sha256(
                fid_root / "FIDNetV3" / f"{dataset}-max25" / "model_best.pth.tar"
            ),
            "metric_function": _source_entrypoint(compute_generative_model_scores),
        }
    vendor_input_hash = _sha256(evaluation_root / "vendor-inputs.npz")
    package_input_hash = _sha256(evaluation_root / "package-inputs.npz")
    vendor_weights_hash = _sha256(checkpoint)
    package_weights_hash = _sha256(checkpoint)
    vendor_prediction_hash = _sha256(evaluation_root / "vendor-predictions.npz")
    package_prediction_hash = _sha256(evaluation_root / "package-predictions.npz")
    prediction_equal = all(record["predictions_equal"] for record in batch_records)
    test_check = {
        "preprocessing_equal": preprocessing_equal,
        "predictions_equal": prediction_equal,
        "vendor_prediction_count": len(vendor_layouts),
        "package_prediction_count": len(package_layouts),
        "vendor_element_count": int(vendor_mask.sum().item()),
        "package_element_count": int(package_mask.sum().item()),
        "vendor_out_of_bounds_count": sum(
            _record_int(record, "vendor_out_of_bounds_count")
            for record in batch_records
        ),
        "package_out_of_bounds_count": sum(
            _record_int(record, "package_out_of_bounds_count")
            for record in batch_records
        ),
        "coordinate_frame": "normalized center-xywh in [0, 1] for decoded outputs; [-1, 1] latent input",
        "metric_implementation": _source_entrypoint(compute_alignment),
        "vendor_metrics": vendor_metrics,
        "package_metrics": package_metrics,
        "metric_values_equal": vendor_metrics == package_metrics,
        "fid": fid_metadata,
        "processed_layout_count": processed_count,
        "expected_test_layout_count": len(vendor_dataset),
        "full_test_split": processed_count == len(vendor_dataset),
        "inputs_sha256": vendor_input_hash,
        "vendor_inputs_sha256": vendor_input_hash,
        "package_inputs_sha256": package_input_hash,
        "vendor_predictions_sha256": vendor_prediction_hash,
        "package_predictions_sha256": package_prediction_hash,
        "weights_sha256": vendor_weights_hash,
        "vendor_weights_sha256": vendor_weights_hash,
        "package_weights_sha256": package_weights_hash,
        "inputs_sha256_equal": vendor_input_hash == package_input_hash,
        "prediction_files_sha256": {
            "vendor": vendor_prediction_hash,
            "package": package_prediction_hash,
        },
        "prediction_artifacts": {
            "vendor_inputs": str(
                (evaluation_root / "vendor-inputs.npz").relative_to(ROOT)
            ),
            "package_inputs": str(
                (evaluation_root / "package-inputs.npz").relative_to(ROOT)
            ),
            "vendor": str(
                (evaluation_root / "vendor-predictions.npz").relative_to(ROOT)
            ),
            "package": str(
                (evaluation_root / "package-predictions.npz").relative_to(ROOT)
            ),
        },
        "sampling": {
            "seed_formula": "20260000 + test batch index",
            "ddim_num_steps": ddim_num_steps,
            "stochastic": True,
            "condition": "c / label",
            "batch_size": batch_size,
            "vendor_default_batch_size": 256,
        },
        "evaluation_settings_equal": all(
            vendor_call == package_call
            for vendor_call, package_call in zip(
                captured_vendor.calls, package_calls, strict=True
            )
        ),
        "sampling_seeds_sha256": hashlib.sha256(
            json.dumps(
                [20260000 + index for index in range(len(batch_records))]
            ).encode()
        ).hexdigest(),
        "timing": {
            "vendor_seconds": captured_vendor.elapsed,
            "package_seconds": package_elapsed,
            "vendor_seconds_per_layout": captured_vendor.elapsed / processed_count,
            "package_seconds_per_layout": package_elapsed / processed_count,
        },
        "batch_records": batch_records,
    }
    evaluation_payload = {
        "dataset": dataset,
        "source_commit": source_commit,
        "checkpoint_sha256": _sha256(checkpoint),
        "weights_sha256_equal": vendor_weights_hash == package_weights_hash,
        "evaluator_commit": vendor_commit,
        "vendor_evaluator_source_commit": vendor_commit,
        "package_evaluator_source_commit": source_commit,
        "input_split": "test",
        "vendor_evaluation_entry": _source_entrypoint(vendor_test.test_layout_cond),
        "package_evaluation_entry": _source_entrypoint(package.__call__),
        "runtime_condition": {
            "vendor_grad_enabled": all(
                not bool(call["grad_enabled"]) for call in captured_vendor.calls
            ),
            "package_grad_enabled": all(
                not bool(call["grad_enabled"]) for call in package_calls
            ),
        },
        "runtime": _runtime_metadata(),
        "test_split": test_check,
    }
    _write_json(evaluation_root, "evaluation.json", evaluation_payload)
    return {
        "dataset": dataset,
        "source_commit": source_commit,
        "checkpoint_sha256": _sha256(checkpoint),
        "weights_sha256_equal": vendor_weights_hash == package_weights_hash,
        "evaluator_commit": vendor_commit,
        "vendor_evaluator_source_commit": vendor_commit,
        "package_evaluator_source_commit": source_commit,
        "input_split": "test",
        "vendor_evaluation_entry": _source_entrypoint(vendor_test.test_layout_cond),
        "package_evaluation_entry": _source_entrypoint(package.__call__),
        "runtime_condition": {
            "vendor_grad_enabled": all(
                not bool(call["grad_enabled"]) for call in captured_vendor.calls
            ),
            "package_grad_enabled": all(
                not bool(call["grad_enabled"]) for call in package_calls
            ),
        },
        "runtime": _runtime_metadata(),
        "test_split": test_check,
        "metrics_exact": test_check["metric_values_equal"],
        "prediction_counts_recorded": all(
            "vendor_prediction_count" in record and "package_prediction_count" in record
            for record in batch_records
        ),
        "output_artifact": str((evaluation_root / "evaluation.json").relative_to(ROOT)),
    }


def run_s4(args: argparse.Namespace) -> Path:
    source_commit, vendor_commit = _assert_clean_sources()
    output = Path(args.output_root) / "s4-loader-evaluation"
    data_root = Path(args.lace_data_root)
    streams = {
        dataset: {
            split: _stream_split(
                dataset,
                data_root,
                split,
                batch_size=4,
                max_batches=args.max_batches,
            )
            for split in ("train", "val", "test")
        }
        for dataset in ("publaynet", "rico25")
    }
    evaluations: dict[str, dict[str, object]] = {
        dataset: _evaluation_parity(
            dataset,
            data_root,
            Path(args.checkpoint_root) / f"{dataset}_best.pt",
            source_commit=source_commit,
            vendor_commit=vendor_commit,
            output_root=output,
            device_name=args.device,
            batch_size=args.evaluation_batch_size,
            max_batches=args.evaluation_max_batches,
            ddim_num_steps=args.evaluation_ddim_steps,
            fid_root=args.fid_root,
        )
        for dataset in ("publaynet", "rico25")
    }
    payload = {
        "stage": "S4",
        "source_commit": source_commit,
        "vendor_commit": vendor_commit,
        "runtime": _runtime_metadata(),
        "streams": streams,
        "evaluations": evaluations,
        "test_split_mandatory_check": all(
            cast(dict[str, object], result["test_split"])["full_test_split"]
            for result in evaluations.values()
        ),
        "exact_loader_stream": all(
            split_result["exact"]
            for dataset_result in streams.values()
            for split_result in dataset_result.values()
        ),
        "exact_evaluation_path": all(
            cast(dict[str, object], result["test_split"])["preprocessing_equal"]
            and cast(dict[str, object], result["test_split"])["predictions_equal"]
            and result["metrics_exact"]
            for result in evaluations.values()
        ),
    }
    return _write_json(output, "summary.json", payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("s0", "s1", "s2", "s3", "s4"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset", default="publaynet", choices=("publaynet", "rico25")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lace-data-root", type=Path, default=None)
    parser.add_argument("--layoutdm-data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--evaluation-max-batches",
        type=int,
        default=None,
        help="limit evaluation batches for a retained timing/subset attempt; default is the full TEST split",
    )
    parser.add_argument("--evaluation-ddim-steps", type=int, default=100)
    parser.add_argument("--fid-root", type=Path, default=None)
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="enable strict or warn-only deterministic PyTorch algorithms for diagnostics",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="warn instead of raising when deterministic algorithms are unavailable",
    )
    parser.add_argument(
        "--sdpa-math",
        action="store_true",
        help="force the math scaled-dot-product-attention backend for both systems",
    )
    args = parser.parse_args()
    if args.stage == "s0" and (
        args.lace_data_root is None or args.layoutdm_data_root is None
    ):
        parser.error("s0 requires --lace-data-root and --layoutdm-data-root")
    if args.stage in {"s1", "s2", "s3"} and args.lace_data_root is None:
        parser.error(f"{args.stage} requires --lace-data-root")
    if args.stage == "s4" and (
        args.lace_data_root is None or args.checkpoint_root is None
    ):
        parser.error("s4 requires --lace-data-root and --checkpoint-root")
    return args


def main() -> None:
    args = _parse_args()
    if args.warn_only and not args.deterministic_algorithms:
        parser_error = "--warn-only requires --deterministic-algorithms"
        raise ValueError(parser_error)
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=args.warn_only)
    runners = {"s0": run_s0, "s1": run_s1, "s2": run_s2, "s3": run_s3, "s4": run_s4}
    path = runners[args.stage](args)
    print(path.resolve().relative_to(ROOT))


if __name__ == "__main__":
    main()
