"""Run DS-GAN training and evaluation agreement evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
from contextlib import contextmanager, redirect_stdout
import csv
import gc
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator, cast  # noqa: TID251 - vendor adapter boundary is heterogeneous
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from traingen_parity.compare import (
    BatchStreamReport,
    compare_batch_stream,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import (
    DeterminismConfig,
    RNGState,
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
)
from traingen_parity.trace import build_step_trace, tensor_sha256


ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "vendor" / "posterlayout-cvpr2023"
CACHE = ROOT / ".cache" / "ds-gan"
EVIDENCE = CACHE / "stage-evidence"
TRAINING_BATCH_SIZE = 128
TEST_BATCH_SIZE = 4
MAX_ELEM = 32
SEED = 0
TRAIN_BATCHES_PER_EPOCH = 78
LOCKSTEP_STEPS = 300
S2_SELF_REPEATS = 12
_DETERMINISTIC_WARNINGS: list[str] = []


def _determinism_mode() -> str:
    mode = os.environ.get("DSGAN_DETERMINISM_MODE", "envelope")
    if mode not in {"strict", "envelope"}:
        raise ValueError(f"unsupported DSGAN_DETERMINISM_MODE: {mode}")

    return mode


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_distribution(name: str) -> dict[str, Any]:
    metadata = distribution(name)
    direct_url_text = metadata.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    return {
        "name": name,
        "version": metadata.version,
        "direct_url": direct_url,
        "location": str(metadata.locate_file("")),
    }


def _runtime() -> dict[str, Any]:
    freeze_path = EVIDENCE / "runtime" / "pip-freeze.txt"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    venv_path = Path(sys.prefix)
    try:
        venv_path_text = str(venv_path.relative_to(ROOT))
    except ValueError:
        venv_path_text = str(venv_path)

    freeze_path.write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True
        )
    )
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_distribution": _runtime_distribution("torch"),
        "torchvision": distribution("torchvision").version,
        "torchvision_distribution": _runtime_distribution("torchvision"),
        "venv_creation_command": (
            f"UV_FROZEN=1 uv venv --python 3.11 {venv_path_text}"
        ),
        "environment_basis": "lockfile environment for all CPU-only checks and tests; audited runtime only for CUDA evidence",
        "pip_freeze_path": str(freeze_path.relative_to(ROOT)),
        "pip_freeze_sha256": _sha256(freeze_path),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "determinism_mode": _determinism_mode(),
        "determinism_note": (
            "strict deterministic mode is used for the determinism probe"
            if _determinism_mode() == "strict"
            else "envelope mode is used after the strict determinism probe raised"
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _metadata() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "refusing evidence from a dirty tree; commit the PR head first"
        )
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "vendor_commit": _git("-C", str(VENDOR), "rev-parse", "HEAD"),
        "backbone_weights": _backbone_manifest(),
        "runtime": _runtime(),
        "deterministic_warning": _warning_record(),
        "strict_determinism_probe": _determinism_probe_record(),
    }


def _write(stage: str, payload: dict[str, Any]) -> Path:
    target = EVIDENCE / stage / "run.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def _write_jsonl(stage: str, name: str, rows: list[dict[str, Any]]) -> Path:
    target = EVIDENCE / stage / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return target


def _set_determinism(seed: int = SEED) -> None:
    apply_determinism(
        DeterminismConfig(
            seed=seed,
            deterministic_algorithms=False,
            cudnn_benchmark=False,
            allow_tf32=False,
            cublas_workspace_config=":4096:8",
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.use_deterministic_algorithms(
            True, warn_only=_determinism_mode() != "strict"
        )
    _DETERMINISTIC_WARNINGS.extend(str(item.message) for item in caught)
    torch.backends.cudnn.deterministic = True


@contextmanager
def _capture_deterministic_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield messages
    messages.extend(str(item.message) for item in caught)
    _DETERMINISTIC_WARNINGS.extend(messages)


def _warning_record() -> dict[str, Any]:
    return {
        "messages": list(dict.fromkeys(_DETERMINISTIC_WARNINGS)),
        "contains_nll_loss2d": any(
            "nll_loss2d" in message for message in _DETERMINISTIC_WARNINGS
        ),
    }


def _determinism_probe_record() -> dict[str, Any] | None:
    log_value = os.environ.get("DSGAN_STRICT_ATTEMPT_LOG")
    if not log_value:
        return None

    log_path = Path(log_value)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    record: dict[str, Any] = {
        "path": str(log_path.relative_to(ROOT)),
        "exists": log_path.is_file(),
    }
    if log_path.is_file():
        record.update({"bytes": log_path.stat().st_size, "sha256": _sha256(log_path)})

    exit_value = os.environ.get("DSGAN_STRICT_ATTEMPT_EXIT")
    if exit_value:
        exit_path = Path(exit_value)
        if not exit_path.is_absolute():
            exit_path = ROOT / exit_path
        record["exit_path"] = str(exit_path.relative_to(ROOT))
        if exit_path.is_file():
            record["exit_code"] = int(exit_path.read_text().strip())

    return record


def _rng_digest(state: RNGState) -> dict[str, str]:
    import hashlib

    return {
        "torch_cpu": tensor_sha256(state.torch_cpu),
        "torch_cuda": hashlib.sha256(
            b"".join(value.cpu().numpy().tobytes() for value in state.torch_cuda)
        ).hexdigest()
        if state.torch_cuda
        else "none",
        "python": hashlib.sha256(repr(state.python).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(state.numpy).encode()).hexdigest(),
    }


@contextmanager
def _vendor_backbone_loader() -> Iterator[None]:
    global _VENDOR_TORCH_LOAD
    original_load = torch.load
    _VENDOR_TORCH_LOAD = original_load
    torch.load = _fake_vendor_backbone  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        torch.load = original_load
        _VENDOR_TORCH_LOAD = None


def _fake_vendor_backbone(
    path: str, *args: Any, **kwargs: Any
) -> dict[str, torch.Tensor]:
    del args, kwargs
    if _VENDOR_TORCH_LOAD is None:
        raise RuntimeError("vendor torch.load hook is not active")

    weight_path = _backbone_path("resnet50" if "resnet50" in path else "resnet18")
    loaded = _VENDOR_TORCH_LOAD(weight_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"vendor backbone weights are not a state dict: {weight_path}")

    return loaded


_VENDOR_CLASSES: tuple[type[nn.Module], type[nn.Module], Any, Any] | None = None
_VENDOR_MAIN: Any | None = None
_VENDOR_TORCH_LOAD: Any | None = None


def _backbone_path(backbone: str) -> Path:
    environment_name = (
        "DSGAN_RESNET50_WEIGHTS" if backbone == "resnet50" else "DSGAN_RESNET18_WEIGHTS"
    )
    value = os.environ.get(environment_name)
    if not value:
        raise RuntimeError(f"{environment_name} is required for training evidence")

    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"backbone weights not found: {path}")

    return path


def _backbone_manifest() -> dict[str, dict[str, str | int]]:
    urls = {
        "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet50": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
    }
    return {
        backbone: {
            "filename": path.name,
            "url": urls[backbone],
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "path_env": (
                "DSGAN_RESNET50_WEIGHTS"
                if backbone == "resnet50"
                else "DSGAN_RESNET18_WEIGHTS"
            ),
        }
        for backbone in ("resnet18", "resnet50")
        for path in (_backbone_path(backbone),)
    }


def _vendor_classes() -> tuple[type[nn.Module], type[nn.Module], Any, Any]:
    global _VENDOR_CLASSES
    if _VENDOR_CLASSES is not None:
        return _VENDOR_CLASSES
    sys.path.insert(0, str(VENDOR))
    with _vendor_backbone_loader():
        from RecLoss import HungarianMatcher as vendor_matcher
        from RecLoss import SetCriterion as vendor_criterion
        from model import discriminator as vendor_discriminator
        from model import generator as vendor_generator

    _VENDOR_CLASSES = (
        vendor_generator,
        vendor_discriminator,
        vendor_criterion,
        vendor_matcher,
    )
    return _VENDOR_CLASSES


def _vendor_main() -> Any:
    global _VENDOR_MAIN
    if _VENDOR_MAIN is None:
        sys.path.insert(0, str(VENDOR))
        import main as vendor_main

        vendor_main.coef = [0.1, 0.8, 1, 1]
        _VENDOR_MAIN = vendor_main
    return _VENDOR_MAIN


def _vendor_cross_entropy_operator() -> dict[str, str | int]:
    path = VENDOR / "RecLoss.py"
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if "F.cross_entropy(" in line:
            return {
                "path": str(path.relative_to(ROOT)),
                "line": line_number,
                "expression": line.strip(),
            }

    raise RuntimeError("vendor cross_entropy operator was not found")


def _configs() -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    from ds_gan import DSGANConfig

    generator_config = DSGANConfig(
        backbone="resnet50",
        max_elem=MAX_ELEM,
        hidden_size=MAX_ELEM * 8,
        num_layers=4,
        image_size=(350, 240),
        backbone_feature_size=330,
    )
    discriminator_config = DSGANConfig(
        backbone="resnet18",
        max_elem=MAX_ELEM,
        hidden_size=MAX_ELEM * 8,
        num_layers=2,
        image_size=(350, 240),
        backbone_feature_size=330,
    )
    common = {
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": MAX_ELEM * 8,
        "output_size": 8,
        "max_elem": MAX_ELEM,
    }
    return (
        generator_config,
        discriminator_config,
        {"backbone": "resnet50", "num_layers": 4, **common},
        {"backbone": "resnet18", "num_layers": 2, **common},
    )


def _models(device: torch.device) -> tuple[Any, Any, Any, Any, Any, Any]:
    from ds_gan import DSGANModel
    from ds_gan.training.discriminator import DSGANDiscriminator

    vendor_generator_class, vendor_discriminator_class, _, _ = _vendor_classes()
    generator_config, discriminator_config, generator_args, discriminator_args = (
        _configs()
    )
    with _vendor_backbone_loader():
        torch.manual_seed(SEED)
        vendor_generator = vendor_generator_class(generator_args).to(device)
        torch.manual_seed(SEED)
        vendor_discriminator = vendor_discriminator_class(discriminator_args).to(device)
    torch.manual_seed(SEED)
    package_generator = DSGANModel(
        generator_config, backbone_weights=_backbone_path("resnet50")
    ).to(device)
    torch.manual_seed(SEED)
    package_discriminator = DSGANDiscriminator(
        discriminator_config, backbone_weights=_backbone_path("resnet18")
    ).to(device)
    return (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    )


def _independent_models(system: str, device: torch.device) -> tuple[Any, Any, Any, Any]:
    """Construct one system from its own seed-controlled initialization path."""
    from ds_gan import DSGANModel
    from ds_gan.training.discriminator import DSGANDiscriminator

    vendor_generator_class, vendor_discriminator_class, _, _ = _vendor_classes()
    _, _, generator_args, discriminator_args = _configs()
    generator_config, discriminator_config, _, _ = _configs()
    _set_determinism(SEED)
    if system == "vendor":
        with _vendor_backbone_loader():
            vendor_generator = vendor_generator_class(generator_args).to(device)
            vendor_discriminator = vendor_discriminator_class(discriminator_args).to(
                device
            )
        return (
            vendor_generator,
            vendor_discriminator,
            generator_config,
            discriminator_config,
        )
    if system == "package":
        package_generator = DSGANModel(
            generator_config, backbone_weights=_backbone_path("resnet50")
        ).to(device)
        package_discriminator = DSGANDiscriminator(
            discriminator_config, backbone_weights=_backbone_path("resnet18")
        ).to(device)
        return (
            package_generator,
            package_discriminator,
            generator_config,
            discriminator_config,
        )
    raise ValueError(f"unsupported natural system: {system}")


def _copy_module_state(destination: nn.Module, source: nn.Module) -> None:
    destination.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)


def _optimizers(
    generator: nn.Module, discriminator: nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    def groups(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        backbone: list[nn.Parameter] = []
        head: list[nn.Parameter] = []
        for name, parameter in module.named_parameters():
            (backbone if name.startswith("resnet_fpn") else head).append(parameter)
        return backbone, head

    generator_backbone, generator_head = groups(generator)
    discriminator_backbone, discriminator_head = groups(discriminator)
    return (
        torch.optim.Adam(
            [
                {"params": generator_head, "lr": 1e-4},
                {"params": generator_backbone, "lr": 1e-5},
            ]
        ),
        torch.optim.Adam(
            [
                {"params": discriminator_head, "lr": 1e-3},
                {"params": discriminator_backbone, "lr": 1e-4},
            ]
        ),
    )


def _schedulers(
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
) -> tuple[torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR]:
    generator, discriminator = optimizers
    return (
        torch.optim.lr_scheduler.MultiStepLR(
            generator, milestones=[0, 50, 100, 150, 200, 250], gamma=0.8
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            discriminator,
            milestones=[0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
            gamma=0.8,
        ),
    )


def _package_optimizers_and_schedulers(
    module: Any,
) -> tuple[
    tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    tuple[torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR],
    list[dict[str, Any]],
]:
    configured = cast(
        tuple[list[torch.optim.Optimizer], list[dict[str, Any]]],
        module.configure_optimizers(),
    )
    optimizers = cast(
        tuple[torch.optim.Optimizer, torch.optim.Optimizer], tuple(configured[0])
    )
    schedulers = tuple(
        cast(torch.optim.lr_scheduler.MultiStepLR, config["scheduler"])
        for config in configured[1]
    )
    if len(optimizers) != 2 or len(schedulers) != 2:
        raise RuntimeError(
            "DS-GAN production configuration must define two optimizers and schedulers"
        )

    return optimizers, schedulers, configured[1]


def _targets(batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    return [
        {"labels": labels.long(), "boxes": boxes.float()}
        for labels, boxes in zip(batch["labels"], batch["boxes"], strict=True)
    ]


def _package_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _vendor_batch(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    images, layout = batch
    labels = layout[:, :, 0].argmax(dim=-1)
    return {
        "pixel_values": images.to(device),
        "layout": layout.to(device),
        "labels": labels.to(device),
        "boxes": layout[:, :, 1].to(device),
        "mask": labels.ne(0).to(device),
    }


def _source_root() -> Path:
    return Path(
        os.environ.get(
            "DSGAN_HF_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
        )
    )


def _bridge_paths() -> dict[str, Path]:
    bridge = CACHE / "bridge"
    return {
        "root": bridge,
        "train_images": bridge / "train" / "inpainted_poster",
        "train_pfpn": bridge / "train" / "saliencymaps_pfpn",
        "train_basnet": bridge / "train" / "saliencymaps_basnet",
        "train_csv": bridge / "train_csv_9973.csv",
        "test_images": bridge / "test" / "image_canvas",
        "test_pfpn": bridge / "test" / "saliencymaps_pfpn",
        "test_basnet": bridge / "test" / "saliencymaps_basnet",
        "manifest": bridge / "pku_posterlayout_manifest.json",
    }


def _aggregate_files(paths: list[Path], root: Path) -> dict[str, Any]:
    entries = [(str(path.relative_to(root)), _sha256(path)) for path in sorted(paths)]
    digest = (
        __import__("hashlib")
        .sha256(json.dumps(entries, separators=(",", ":")).encode())
        .hexdigest()
    )
    return {"count": len(entries), "sha256": digest}


def _materialize_bridge() -> dict[str, Any]:
    from ds_gan.training.dataset import (
        SOURCE_TO_VENDOR_LABEL,
        load_cached_dataset,
        manifest_from_cached_dataset,
    )

    source = load_cached_dataset(_source_root())
    paths = _bridge_paths()
    for path in (
        paths["train_images"],
        paths["train_pfpn"],
        paths["train_basnet"],
        paths["test_images"],
        paths["test_pfpn"],
        paths["test_basnet"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    paths["train_csv"].parent.mkdir(parents=True, exist_ok=True)
    train_rows = source["train"]
    test_rows = source["test"]
    invalid_only_rows = 0
    train_images_complete = len(list(paths["train_images"].glob("*.png"))) == len(
        train_rows
    )
    train_pfpn_complete = len(list(paths["train_pfpn"].glob("*.png"))) == len(
        train_rows
    )
    train_basnet_complete = len(list(paths["train_basnet"].glob("*.png"))) == len(
        train_rows
    )
    with paths["train_csv"].open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["poster_path", "total_elem", "cls_elem", "box_elem"])
        annotation_rows = train_rows.select_columns(["annotations"]).with_format(
            "python"
        )
        for index, row in enumerate(annotation_rows):
            filename = f"{index:05d}_mask.png"
            poster_path = f"train/{index:05d}.png"
            annotations = row["annotations"]
            if not (
                train_images_complete and train_pfpn_complete and train_basnet_complete
            ):
                source_row = train_rows[index]
                source_row["inpainted_poster"].save(paths["train_images"] / filename)
                source_row["pfpn_saliency_map"].save(
                    paths["train_pfpn"] / filename.replace(".png", "_pred.png")
                )
                source_row["basnet_saliency_map"].save(paths["train_basnet"] / filename)
            mapped_rows = 0
            for label, box in zip(
                annotations["cls_elem"], annotations["box_elem"], strict=True
            ):
                vendor_label = SOURCE_TO_VENDOR_LABEL[int(label)]
                if vendor_label == 0:
                    continue
                writer.writerow(
                    [poster_path, len(annotations["cls_elem"]), vendor_label, repr(box)]
                )
                mapped_rows += 1
            if mapped_rows == 0:
                invalid_only_rows += 1
                writer.writerow(
                    [poster_path, len(annotations["cls_elem"]), 0, "[0, 0, 0, 0]"]
                )
    test_images_complete = len(list(paths["test_images"].glob("*.png"))) == len(
        test_rows
    )
    test_pfpn_complete = len(list(paths["test_pfpn"].glob("*.png"))) == len(test_rows)
    test_basnet_complete = len(list(paths["test_basnet"].glob("*.png"))) == len(
        test_rows
    )
    if not (test_images_complete and test_pfpn_complete and test_basnet_complete):
        for index, row in enumerate(test_rows):
            filename = f"{index:05d}.png"
            row["canvas"].save(paths["test_images"] / filename)
            row["pfpn_saliency_map"].save(
                paths["test_pfpn"] / filename.replace(".png", "_pred.png")
            )
            row["basnet_saliency_map"].save(paths["test_basnet"] / filename)
    manifest = manifest_from_cached_dataset(_source_root())
    manifest["source_fields"] = {
        "train.inpainted_poster": "inpainted_poster",
        "train.saliency_pfpnet": "pfpn_saliency_map",
        "train.saliency_basnet": "basnet_saliency_map",
        "train.annotations": "annotations",
        "test.canvas": "canvas",
        "test.saliency_pfpnet": "pfpn_saliency_map",
        "test.saliency_basnet": "basnet_saliency_map",
    }
    manifest["bridge"] = {
        "train.inpainted_poster": _aggregate_files(
            list(paths["train_images"].glob("*.png")), paths["root"]
        ),
        "train.saliency_pfpnet": _aggregate_files(
            list(paths["train_pfpn"].glob("*.png")), paths["root"]
        ),
        "train.saliency_basnet": _aggregate_files(
            list(paths["train_basnet"].glob("*.png")), paths["root"]
        ),
        "train.annotations": {
            "count": train_rows.num_rows,
            "sha256": _sha256(paths["train_csv"]),
            "invalid_only_rows": invalid_only_rows,
        },
        "test.canvas": _aggregate_files(
            list(paths["test_images"].glob("*.png")), paths["root"]
        ),
        "test.saliency_pfpnet": _aggregate_files(
            list(paths["test_pfpn"].glob("*.png")), paths["root"]
        ),
        "test.saliency_basnet": _aggregate_files(
            list(paths["test_basnet"].glob("*.png")), paths["root"]
        ),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _make_bridge_read_only() -> None:
    paths = _bridge_paths()
    for path in sorted(paths["root"].rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    paths["root"].chmod(0o555)


def _source_artifact_hashes() -> dict[str, str]:
    root = _source_root()
    metadata = sorted(
        root.glob(
            "creative-graphic-design___pku-poster_layout/default/0.0.0/*/dataset_info.json"
        )
    )[-1]
    dataset_dir = metadata.parent
    source_paths = [
        metadata,
        *sorted(dataset_dir.glob("pku-poster_layout-train-*.arrow")),
        dataset_dir / "pku-poster_layout-test.arrow",
    ]
    return {path.name: _sha256(path) for path in source_paths}


def _vendor_loaders(seed: int, *, overlay_root: Path | None = None) -> tuple[Any, Any]:
    sys.path.insert(0, str(VENDOR))
    from dataloader import canvas, canvasLayout

    paths = _bridge_paths()
    train_dataset = canvasLayout(
        str(paths["train_images"]),
        str(paths["train_pfpn"]),
        str(paths["train_basnet"]),
        str(paths["train_csv"]),
        MAX_ELEM,
    )
    loader_overlay = overlay_root or EVIDENCE / "loader-overlays" / str(os.getpid())
    loader_overlay.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(loader_overlay)
        test_dataset = canvas(
            str(paths["test_images"]),
            str(paths["test_pfpn"]),
            str(paths["test_basnet"]),
            train=False,
        )
    finally:
        os.chdir(old_cwd)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )

    def pad_test(items: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            items
            if len(items) == TEST_BATCH_SIZE
            else items + [items[0]] * (TEST_BATCH_SIZE - len(items))
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=pad_test,
    )
    return train_loader, test_loader


def _package_loaders(seed: int) -> tuple[Any, Any]:
    from ds_gan.training.datamodule import DSGANDataModule

    module = DSGANDataModule(
        cache_dir=str(_source_root()),
        batch_size=TRAINING_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        max_elem=MAX_ELEM,
        num_workers=16,
        seed=seed,
    )
    module.setup("fit")
    production_train_loader = module.train_dataloader()
    train_loader = DataLoader(
        production_train_loader.dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    return train_loader, module.test_dataloader()


def _batch_stream_report(
    package_batch: dict[str, torch.Tensor], vendor_batch: dict[str, torch.Tensor]
) -> tuple[BatchStreamReport, dict[str, Any]]:
    report = compare_batch_stream([vendor_batch], [package_batch], steps=1)
    return report, {
        "passed": report.passed,
        "checked_steps": report.checked_steps,
        "first_mismatch": report.first_mismatch,
    }


def _sample_ids(batch: torch.Tensor) -> list[str]:
    return [tensor_sha256(sample) for sample in batch]


def _comparison_dict(
    report: Any,
    reference: dict[str, torch.Tensor] | None = None,
    target: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    first = next((item for item in report.comparisons if not item.passed), None)
    first_difference = asdict(first) if first is not None else None
    max_abs_difference = max(
        (item.max_abs_diff for item in report.comparisons), default=0.0
    )
    max_relative_difference = max(
        (item.max_rel_diff for item in report.comparisons), default=0.0
    )
    if first is not None and reference is not None and target is not None:
        assert first_difference is not None
        expected = reference[first.name].detach().float()
        actual = target[first.name].detach().float()
        difference = (actual - expected).abs()
        index = (
            tuple(
                int(value)
                for value in torch.unravel_index(difference.argmax(), difference.shape)
            )
            if difference.numel()
            else ()
        )
        first_difference = {
            **first_difference,
            "index": index,
            "vendor_value": float(expected[index].item())
            if index
            else float(expected.item()),
            "package_value": float(actual[index].item())
            if index
            else float(actual.item()),
        }
    return {
        "passed": bool(report.passed),
        "missing": list(report.missing),
        "comparison_count": len(report.comparisons),
        "failed_count": sum(not item.passed for item in report.comparisons),
        "max_abs_difference": max_abs_difference,
        "max_relative_difference": max_relative_difference,
        "first_difference": first_difference,
    }


def _trace_compare(
    vendor: dict[str, torch.Tensor], package: dict[str, torch.Tensor]
) -> dict[str, Any]:
    report = compare_step_trace(
        build_step_trace("vendor", vendor), build_step_trace("package", package)
    )
    return _comparison_dict(report, vendor, package)


def _state_compare(
    vendor: dict[str, torch.Tensor], package: dict[str, torch.Tensor]
) -> dict[str, Any]:
    report = compare_optimizer_step(vendor, package)
    return _comparison_dict(report, vendor, package)


def _vendor_criterion(device: torch.device) -> nn.Module:
    _, _, criterion_class, matcher_class = _vendor_classes()
    return criterion_class(
        3,
        matcher_class(2, 5, 2),
        {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2},
        [0.1, 0.8, 1, 1],
        ["labels", "boxes"],
    ).to(device)


class _DeferredScheduler:
    def step(self) -> None:
        return None


def _vendor_step(
    generator: nn.Module,
    discriminator: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    initial_layout: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    vendor_main = _vendor_main()
    vendor_main.device = batch["pixel_values"].device
    raw_batch = (batch["pixel_values"].detach().cpu(), batch["layout"].detach().cpu())
    captured_layout: list[torch.Tensor] = []
    captured_generator: list[tuple[torch.Tensor, torch.Tensor]] = []
    captured_scores: list[torch.Tensor] = []
    captured_reconstruction: list[torch.Tensor] = []
    original_random_init = vendor_main.random_init
    original_generator_forward = generator.forward
    original_discriminator_forward = discriminator.forward
    original_criterion_forward = criterion.forward

    def capture_random_init(batch_size: int, max_elem: int) -> torch.Tensor:
        if initial_layout is not None:
            layout = initial_layout.detach().cpu().clone()
            captured_layout.append(layout)
            return layout

        layout = original_random_init(batch_size, max_elem)
        captured_layout.append(layout.detach().clone())
        return layout

    def capture_generator(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = original_generator_forward(*args, **kwargs)
        captured_generator.append((result[0].detach(), result[1].detach()))
        return result

    def capture_discriminator(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_discriminator_forward(*args, **kwargs)
        captured_scores.append(result.detach())
        return result

    def capture_criterion(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        losses = original_criterion_forward(*args, **kwargs)
        if not captured_reconstruction:
            captured_reconstruction.append(sum(losses.values()).detach())
        return losses

    setattr(vendor_main, "random_init", capture_random_init)
    setattr(generator, "forward", capture_generator)
    setattr(discriminator, "forward", capture_discriminator)
    setattr(criterion, "forward", capture_criterion)
    try:
        with redirect_stdout(io.StringIO()):
            vendor_main.train(
                generator,
                discriminator,
                [raw_batch],
                criterion,
                nn.HingeEmbeddingLoss(),
                min(1.0, max(0, epoch - 1) / 100),
                optimizer_g,
                optimizer_d,
                _DeferredScheduler(),
                _DeferredScheduler(),
                epoch,
                MAX_ELEM,
            )
    finally:
        setattr(vendor_main, "random_init", original_random_init)
        setattr(generator, "forward", original_generator_forward)
        setattr(discriminator, "forward", original_discriminator_forward)
        setattr(criterion, "forward", original_criterion_forward)

    if (
        len(captured_layout) != 1
        or len(captured_generator) != 1
        or len(captured_reconstruction) != 1
    ):
        raise RuntimeError("vendor train entry point did not produce one captured step")
    if len(captured_scores) != 3:
        raise RuntimeError(
            "vendor train entry point did not produce three discriminator calls"
        )
    initial_layout = captured_layout[0].to(batch["pixel_values"].device)
    classes, boxes = captured_generator[0]
    classes = classes.to(batch["pixel_values"].device)
    boxes = boxes.to(batch["pixel_values"].device)
    generated_score, discriminator_fake, discriminator_real = captured_scores
    generated_score = generated_score.to(batch["pixel_values"].device)
    discriminator_fake = discriminator_fake.to(batch["pixel_values"].device)
    discriminator_real = discriminator_real.to(batch["pixel_values"].device)
    batch_size = batch["pixel_values"].shape[0]
    real = torch.ones(batch_size, device=batch["pixel_values"].device)
    fake = torch.full((batch_size,), -1.0, device=batch["pixel_values"].device)
    loss_reconstruction = captured_reconstruction[0].to(batch["pixel_values"].device)
    loss_g_adv = nn.functional.hinge_embedding_loss(generated_score.reshape(-1), real)
    loss_g = min(1.0, max(0, epoch - 1) / 100) * loss_g_adv + loss_reconstruction
    loss_d_fake = nn.functional.hinge_embedding_loss(
        discriminator_fake.reshape(-1), fake
    )
    loss_d_real = nn.functional.hinge_embedding_loss(
        discriminator_real.reshape(-1), real
    )
    loss_d = min(1.0, max(0, epoch - 1) / 100) * (loss_d_real + loss_d_fake)
    return {
        "initial_layout": initial_layout.detach(),
        "class_probs": classes.detach(),
        "bbox": boxes.detach(),
        "discriminator_generated": generated_score.detach(),
        "discriminator_fake": discriminator_fake.detach(),
        "discriminator_real": discriminator_real.detach(),
        "loss_g_adv": loss_g_adv.detach(),
        "loss_reconstruction": loss_reconstruction.detach(),
        "loss_g": loss_g.detach(),
        "loss_d_fake": loss_d_fake.detach(),
        "loss_d_real": loss_d_real.detach(),
        "loss_d": loss_d.detach(),
    }


def _package_step(
    module: Any,
    batch: dict[str, torch.Tensor],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    initial_layout: torch.Tensor | None = None,
    numpy_rng: np.random.RandomState | None = None,
    torch_generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    module.step_with_optimizers(
        batch,
        optimizer_g,
        optimizer_d,
        initial_layout=initial_layout,
        numpy_rng=numpy_rng,
        torch_generator=torch_generator,
        epoch=epoch,
    )
    return dict(module.latest_step_trace)


def _run_s2_self_repeat(system: str, repeat: int, json_path: Path) -> None:
    if system not in {"vendor", "package"}:
        raise ValueError(f"unsupported S2 self-repeat system: {system}")

    _set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(device)
    _copy_module_state(package_generator, vendor_generator)
    _copy_module_state(package_discriminator, vendor_discriminator)
    if system == "vendor":
        generator = vendor_generator
        discriminator = vendor_discriminator
    else:
        generator = package_generator
        discriminator = package_discriminator
    optimizers = _optimizers(generator, discriminator)
    initial_parameters = {
        name: value.cpu()
        for name, value in {
            **_named_parameters(generator, "generator"),
            **_named_parameters(discriminator, "discriminator"),
        }.items()
    }
    if system == "vendor":
        _schedulers(optimizers)
        batch, _, initial_layout, batch_meta = _fixed_batch(SEED, device)
        with _capture_deterministic_warnings():
            trace = _vendor_step(
                generator,
                discriminator,
                batch,
                *optimizers,
                _vendor_criterion(device),
                1,
                initial_layout,
            )
    else:
        from ds_gan.training.lightning_module import DSGANTrainingModule

        module = DSGANTrainingModule(
            config=generator_config,
            discriminator_config=discriminator_config,
            generator=generator,
            discriminator=discriminator,
        ).to(device)
        optimizers, _, _ = _package_optimizers_and_schedulers(module)
        _, batch, initial_layout, batch_meta = _fixed_batch(SEED, device)
        with _capture_deterministic_warnings():
            trace = _package_step(module, batch, *optimizers, 1, initial_layout)

    states = {
        "initial_parameters": initial_parameters,
        "parameters": {
            name: value.cpu()
            for name, value in {
                **_named_parameters(generator, "generator"),
                **_named_parameters(discriminator, "discriminator"),
            }.items()
        },
        "gradients": {
            name: value.cpu()
            for name, value in {
                **_named_gradients(generator, "generator"),
                **_named_gradients(discriminator, "discriminator"),
            }.items()
        },
        "optimizer_state": {
            name: value.cpu()
            for name, value in {
                **_named_optimizer_state(optimizers[0], generator, "generator"),
                **_named_optimizer_state(optimizers[1], discriminator, "discriminator"),
            }.items()
        },
        "optimizer_hyperparameters": {
            **_named_optimizer_hyperparameters(optimizers[0], generator, "generator"),
            **_named_optimizer_hyperparameters(
                optimizers[1], discriminator, "discriminator"
            ),
        },
    }

    state_path = json_path.with_suffix(".pt")
    torch.save(states, state_path)
    json_path.write_text(
        json.dumps(
            {
                **_metadata(),
                "stage": "S2-self-repeat",
                "system": system,
                "repeat": repeat,
                "seed": SEED,
                "process_id": os.getpid(),
                "state_artifact": str(state_path.relative_to(ROOT)),
                "device": str(device),
                "batch": batch_meta,
                "initial_layout_sha256": tensor_sha256(initial_layout),
                "operator": _vendor_cross_entropy_operator(),
                "trace_values": {
                    "loss_reconstruction": float(trace["loss_reconstruction"].item())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _run_s2_self_repeat_process(system: str, repeat: int) -> dict[str, Any]:
    artifact_root = EVIDENCE / "s2-optimizer-step-attempts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    json_path = artifact_root / f"{system}-self-repeat-{repeat}.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "s2-vendor-self-repeat",
            str(repeat),
            str(json_path),
            "--system",
            system,
        ],
        check=True,
        cwd=ROOT,
    )
    run = json.loads(json_path.read_text())
    run["_state"] = torch.load(
        json_path.with_suffix(".pt"), map_location="cpu", weights_only=True
    )
    return run


def _scalar_distribution(
    absolute: list[float], relative: list[float], field: str, pair_count: int
) -> dict[str, Any]:
    if not absolute or not relative:
        raise RuntimeError(f"empty S2 scalar distribution for {field}")

    return {
        "field": field,
        "pair_count": pair_count,
        "absolute_values": absolute,
        "relative_values": relative,
        "max_abs_difference": max(absolute),
        "median_abs_difference": float(median(absolute)),
        "max_relative_difference": max(relative),
        "median_relative_difference": float(median(relative)),
    }


def _scalar_self_distribution(runs: list[dict[str, Any]], field: str) -> dict[str, Any]:
    pairs = [
        (
            float(runs[first]["trace_values"][field]),
            float(runs[second]["trace_values"][field]),
        )
        for first in range(len(runs))
        for second in range(first + 1, len(runs))
    ]
    absolute = [abs(first - second) for first, second in pairs]
    relative = [
        difference / max(abs(first), abs(second))
        if max(abs(first), abs(second))
        else 0.0
        for (first, second), difference in zip(pairs, absolute, strict=True)
    ]
    return _scalar_distribution(absolute, relative, field, len(pairs))


def _scalar_cross_distribution(
    vendor_runs: list[dict[str, Any]], package_runs: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    pairs = [
        (
            float(vendor["trace_values"][field]),
            float(package["trace_values"][field]),
        )
        for vendor, package in zip(vendor_runs, package_runs, strict=True)
    ]
    absolute = [abs(vendor - package) for vendor, package in pairs]
    relative = [
        difference / max(abs(vendor), abs(package))
        if max(abs(vendor), abs(package))
        else 0.0
        for (vendor, package), difference in zip(pairs, absolute, strict=True)
    ]
    return _scalar_distribution(absolute, relative, field, len(pairs))


def _comparison_distribution(
    comparisons: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    if not comparisons:
        raise RuntimeError(f"empty S2 comparison distribution for {field}")

    absolute = [item["max_abs_difference"] for item in comparisons]
    relative = [item["max_relative_difference"] for item in comparisons]
    return {
        "field": field,
        "pair_count": len(comparisons),
        "max_abs_difference": max(absolute),
        "median_abs_difference": float(median(absolute)),
        "max_relative_difference": max(relative),
        "median_relative_difference": float(median(relative)),
        "comparisons": comparisons,
    }


def _state_distribution(
    states: list[dict[str, dict[str, torch.Tensor]]], field: str
) -> dict[str, Any]:
    comparisons = [
        _state_compare(states[first][field], states[second][field])
        for first in range(len(states))
        for second in range(first + 1, len(states))
    ]
    return _comparison_distribution(comparisons, field)


def _cross_state_distribution(
    vendor_states: list[dict[str, dict[str, torch.Tensor]]],
    package_states: list[dict[str, dict[str, torch.Tensor]]],
    field: str,
) -> dict[str, Any]:
    comparisons = [
        _state_compare(vendor[field], package[field])
        for vendor, package in zip(vendor_states, package_states, strict=True)
    ]
    return _comparison_distribution(comparisons, field)


def _inside_self_distributions(
    cross: dict[str, Any], vendor: dict[str, Any], package: dict[str, Any]
) -> bool:
    return cross["max_abs_difference"] <= max(
        vendor["max_abs_difference"], package["max_abs_difference"]
    )


def _elementwise_envelope(
    vendor_states: list[dict[str, dict[str, torch.Tensor]]],
    package_states: list[dict[str, dict[str, torch.Tensor]]],
    field: str,
) -> dict[str, Any]:
    outside: list[dict[str, Any]] = []
    maximum: dict[str, Any] | None = None
    maximum_cross_abs = -1.0
    outside_vendor = False
    outside_package = False
    for name in vendor_states[0][field]:
        vendor_values = torch.stack(
            [state[field][name].float() for state in vendor_states]
        )
        package_values = torch.stack(
            [state[field][name].float() for state in package_states]
        )
        vendor_envelope = vendor_values.amax(dim=0) - vendor_values.amin(dim=0)
        package_envelope = package_values.amax(dim=0) - package_values.amin(dim=0)
        cross = torch.stack(
            [
                (vendor[field][name].float() - package[field][name].float()).abs()
                for vendor, package in zip(vendor_states, package_states, strict=True)
            ]
        ).amax(dim=0)
        maximum_index = tuple(
            int(value) for value in torch.unravel_index(cross.argmax(), cross.shape)
        )
        maximum_candidate = {
            "name": name,
            "index": maximum_index,
            "cross_max_abs": float(cross[maximum_index].item()),
            "vendor_self_max_abs": float(vendor_envelope[maximum_index].item()),
            "package_self_max_abs": float(package_envelope[maximum_index].item()),
        }
        if field == "parameters":
            maximum_candidate["vendor_gradient_values"] = [
                float(state["gradients"][name][maximum_index].item())
                for state in vendor_states
            ]
            maximum_candidate["package_gradient_values"] = [
                float(state["gradients"][name][maximum_index].item())
                for state in package_states
            ]
            maximum_candidate["all_24_gradient_values"] = (
                maximum_candidate["vendor_gradient_values"]
                + maximum_candidate["package_gradient_values"]
            )
        candidate_cross_abs = float(maximum_candidate["cross_max_abs"])
        if candidate_cross_abs > maximum_cross_abs:
            maximum = maximum_candidate
            maximum_cross_abs = candidate_cross_abs
        envelope = torch.maximum(vendor_envelope, package_envelope)
        outside_vendor_mask = cross > vendor_envelope
        outside_package_mask = cross > package_envelope
        outside_mask = cross > envelope
        outside_vendor = outside_vendor or bool(outside_vendor_mask.any().item())
        outside_package = outside_package or bool(outside_package_mask.any().item())
        for index in zip(*torch.where(outside_mask)):
            index_tuple = tuple(int(value) for value in index)
            outside.append(
                {
                    "name": name,
                    "index": index_tuple,
                    "cross_max_abs": float(cross[index_tuple].item()),
                    "vendor_self_max_abs": float(vendor_envelope[index_tuple].item()),
                    "package_self_max_abs": float(package_envelope[index_tuple].item()),
                    "outside_vendor_self_envelope": bool(
                        outside_vendor_mask[index_tuple].item()
                    ),
                    "outside_package_self_envelope": bool(
                        outside_package_mask[index_tuple].item()
                    ),
                }
            )
            if field == "parameters":
                outside[-1]["vendor_gradient_values"] = [
                    float(state["gradients"][name][index_tuple].item())
                    for state in vendor_states
                ]
                outside[-1]["package_gradient_values"] = [
                    float(state["gradients"][name][index_tuple].item())
                    for state in package_states
                ]

    if maximum is None:
        raise RuntimeError(f"empty elementwise envelope for {field}")

    return {
        "field": field,
        "element_count_outside_self_envelope": len(outside),
        "outside_self_envelope": outside,
        "maximum_cross_element": maximum,
        "cross_inside_vendor_self_envelope": not outside_vendor,
        "cross_inside_package_self_envelope": not outside_package,
        "cross_inside_combined_self_envelope": not outside,
    }


def _sign_straddling_parameter_probe(
    vendor_states: list[dict[str, Any]],
    package_states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    maximum: dict[str, Any] | None = None
    maximum_cross_abs = -1.0
    for name in vendor_states[0]["parameters"]:
        vendor_gradients = torch.stack(
            [state["gradients"][name].float() for state in vendor_states]
        )
        package_gradients = torch.stack(
            [state["gradients"][name].float() for state in package_states]
        )
        vendor_parameters = torch.stack(
            [state["parameters"][name].float() for state in vendor_states]
        )
        package_parameters = torch.stack(
            [state["parameters"][name].float() for state in package_states]
        )
        all_gradients = torch.cat((vendor_gradients, package_gradients))
        sign_straddles = (all_gradients > 0).any(dim=0) & (all_gradients < 0).any(dim=0)
        for index in zip(*torch.where(sign_straddles)):
            index_tuple = tuple(int(value) for value in index)
            cross = torch.stack(
                [
                    (
                        vendor_parameters[repeat][index_tuple]
                        - package_parameters[repeat][index_tuple]
                    ).abs()
                    for repeat in range(len(vendor_states))
                ]
            ).max()
            cross_abs = float(cross.item())
            if cross_abs <= maximum_cross_abs:
                continue

            vendor_envelope = (
                vendor_parameters[:, index_tuple].amax()
                - vendor_parameters[:, index_tuple].amin()
            )
            package_envelope = (
                package_parameters[:, index_tuple].amax()
                - package_parameters[:, index_tuple].amin()
            )
            maximum = {
                "name": name,
                "index": index_tuple,
                "cross_max_abs": cross_abs,
                "vendor_self_max_abs": float(vendor_envelope.item()),
                "package_self_max_abs": float(package_envelope.item()),
            }
            maximum_cross_abs = cross_abs

    if maximum is None:
        return None

    return maximum


def _parameter_update_mechanism(
    vendor_states: list[dict[str, Any]],
    package_states: list[dict[str, Any]],
    parameter_envelope: dict[str, Any],
    *,
    include_maximum_probe: bool = True,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for item in parameter_envelope["outside_self_envelope"]:
        name = item["name"]
        index = tuple(item["index"])
        vendor_gradients = [
            float(state["gradients"][name][index].item()) for state in vendor_states
        ]
        package_gradients = [
            float(state["gradients"][name][index].item()) for state in package_states
        ]
        all_gradients = vendor_gradients + package_gradients
        gradient_self_max_abs = max(
            max(vendor_gradients) - min(vendor_gradients),
            max(package_gradients) - min(package_gradients),
        )
        gradient_cross_max_abs = max(
            abs(vendor - package)
            for vendor, package in zip(vendor_gradients, package_gradients, strict=True)
        )
        vendor_hyperparameters = vendor_states[0]["optimizer_hyperparameters"][name]
        package_hyperparameters = package_states[0]["optimizer_hyperparameters"][name]
        initial_values = [
            float(state["initial_parameters"][name][index].item())
            for state in vendor_states
        ] + [
            float(state["initial_parameters"][name][index].item())
            for state in package_states
        ]
        initial_cross_max_abs = max(
            abs(vendor - package)
            for vendor, package in zip(
                initial_values[: len(vendor_states)],
                initial_values[len(vendor_states) :],
                strict=True,
            )
        )
        predictions: list[dict[str, Any]] = []
        for repeat, (vendor_state, package_state) in enumerate(
            zip(vendor_states, package_states, strict=True), 1
        ):
            vendor_gradient = vendor_gradients[repeat - 1]
            package_gradient = package_gradients[repeat - 1]
            vendor_delta = _adam_first_step_delta(
                vendor_gradient, vendor_hyperparameters
            )
            package_delta = _adam_first_step_delta(
                package_gradient, package_hyperparameters
            )
            vendor_initial = vendor_state["initial_parameters"][name][index]
            package_initial = package_state["initial_parameters"][name][index]
            predicted_vendor = float(
                (vendor_initial + torch.tensor(vendor_delta)).item()
            )
            predicted_package = float(
                (package_initial + torch.tensor(package_delta)).item()
            )
            actual_vendor = float(vendor_state["parameters"][name][index].item())
            actual_package = float(package_state["parameters"][name][index].item())
            predicted_cross = predicted_vendor - predicted_package
            actual_cross = actual_vendor - actual_package
            predictions.append(
                {
                    "repeat": repeat,
                    "vendor_gradient": vendor_gradient,
                    "package_gradient": package_gradient,
                    "vendor_gradient_sign": (
                        "positive"
                        if vendor_gradient > 0
                        else "negative"
                        if vendor_gradient < 0
                        else "zero"
                    ),
                    "package_gradient_sign": (
                        "positive"
                        if package_gradient > 0
                        else "negative"
                        if package_gradient < 0
                        else "zero"
                    ),
                    "vendor_delta": vendor_delta,
                    "package_delta": package_delta,
                    "predicted_vendor_parameter": predicted_vendor,
                    "predicted_package_parameter": predicted_package,
                    "predicted_cross_parameter_difference": predicted_cross,
                    "actual_cross_parameter_difference": actual_cross,
                    "prediction_residual": actual_cross - predicted_cross,
                }
            )
        vendor_signs = {
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in vendor_gradients
        }
        package_signs = {
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in package_gradients
        }
        all_signs = {
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in all_gradients
        }
        candidates.append(
            {
                **item,
                "vendor_gradient_values": vendor_gradients,
                "package_gradient_values": package_gradients,
                "all_24_gradient_values": all_gradients,
                "vendor_gradient_signs": sorted(vendor_signs),
                "package_gradient_signs": sorted(package_signs),
                "all_gradient_signs": sorted(all_signs),
                "both_signs_occur": "positive" in all_signs and "negative" in all_signs,
                "gradient_cross_max_abs": gradient_cross_max_abs,
                "gradient_self_max_abs": gradient_self_max_abs,
                "gradient_inside_like_for_like_envelope": (
                    gradient_cross_max_abs <= gradient_self_max_abs
                ),
                "initial_parameter_cross_max_abs": initial_cross_max_abs,
                "vendor_optimizer_hyperparameters": vendor_hyperparameters,
                "package_optimizer_hyperparameters": package_hyperparameters,
                "adam_first_step_predictions": predictions,
                "adam_prediction_max_abs_residual": max(
                    abs(prediction["prediction_residual"]) for prediction in predictions
                ),
                "adam_prediction_bitwise": all(
                    prediction["prediction_residual"] == 0.0
                    for prediction in predictions
                ),
            }
        )
    explained = all(
        candidate["both_signs_occur"]
        and candidate["gradient_inside_like_for_like_envelope"]
        and candidate["adam_prediction_bitwise"]
        for candidate in candidates
    )
    maximum_probe = None
    if include_maximum_probe:
        maximum_probe = _parameter_update_mechanism(
            vendor_states,
            package_states,
            {
                **parameter_envelope,
                "outside_self_envelope": [parameter_envelope["maximum_cross_element"]],
            },
            include_maximum_probe=False,
        )["candidates"][0]
    return {
        "parameter_outside_count": len(candidates),
        "candidates": candidates,
        "maximum_cross_element_probe": maximum_probe,
        "all_outside_elements_explained_by_sign_straddling_adam": explained,
        "passed": explained,
    }


class _NaturalParityCallback(Callback):
    def __init__(
        self,
        vendor_generator: nn.Module,
        vendor_discriminator: nn.Module,
        vendor_loader: DataLoader[Any],
        vendor_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        vendor_schedulers: tuple[
            torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR
        ],
        device: torch.device,
    ) -> None:
        self.vendor_generator = vendor_generator
        self.vendor_discriminator = vendor_discriminator
        self.vendor_loader = vendor_loader
        self.vendor_iterator = iter(vendor_loader)
        self.vendor_optimizers = vendor_optimizers
        self.vendor_schedulers = vendor_schedulers
        self.device = device
        self.rows: list[dict[str, Any]] = []
        self._vendor_trace: dict[str, torch.Tensor] | None = None
        self._vendor_parameters: dict[str, torch.Tensor] = {}
        self._vendor_gradients: dict[str, torch.Tensor] = {}
        self._vendor_optimizer_state: dict[str, torch.Tensor] = {}
        self._batch_report: BatchStreamReport | None = None
        self.iterations = 0

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if trainer.current_epoch:
            self.vendor_iterator = iter(self.vendor_loader)

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del pl_module, batch_idx
        vendor_batch = _vendor_batch(next(self.vendor_iterator), self.device)
        package_batch = _package_batch(batch, self.device)
        self._batch_report, _ = _batch_stream_report(package_batch, vendor_batch)
        self._vendor_trace = _vendor_step(
            self.vendor_generator,
            self.vendor_discriminator,
            vendor_batch,
            *self.vendor_optimizers,
            _vendor_criterion(self.device),
            trainer.current_epoch + 1,
        )
        self._vendor_parameters = {
            **_named_parameters(self.vendor_generator, "generator"),
            **_named_parameters(self.vendor_discriminator, "discriminator"),
        }
        self._vendor_gradients = {
            **_named_gradients(self.vendor_generator, "generator"),
            **_named_gradients(self.vendor_discriminator, "discriminator"),
        }
        self._vendor_optimizer_state = {
            **_named_optimizer_state(
                self.vendor_optimizers[0], self.vendor_generator, "generator"
            ),
            **_named_optimizer_state(
                self.vendor_optimizers[1], self.vendor_discriminator, "discriminator"
            ),
        }

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx
        if self._vendor_trace is None or self._batch_report is None:
            raise RuntimeError("natural parity callback has no vendor step")

        package_schedulers = tuple(
            config.scheduler for config in trainer.lr_scheduler_configs
        )
        package_trace = dict(pl_module.latest_step_trace)
        package_parameters = {
            **_named_parameters(pl_module.generator, "generator"),
            **_named_parameters(pl_module.discriminator, "discriminator"),
        }
        package_gradients = {
            **_named_gradients(pl_module.generator, "generator"),
            **_named_gradients(pl_module.discriminator, "discriminator"),
        }
        package_optimizer_state = {
            **_named_optimizer_state(
                trainer.optimizers[0], pl_module.generator, "generator"
            ),
            **_named_optimizer_state(
                trainer.optimizers[1], pl_module.discriminator, "discriminator"
            ),
        }
        self.iterations += 1
        scheduler_values = {
            "vendor_last_epoch": [
                scheduler.last_epoch for scheduler in self.vendor_schedulers
            ],
            "package_last_epoch": [
                scheduler.last_epoch for scheduler in package_schedulers
            ],
            "vendor_learning_rates": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in self.vendor_optimizers
            ],
            "package_learning_rates": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in trainer.optimizers
            ],
            "stepped": self.iterations % TRAIN_BATCHES_PER_EPOCH == 0,
        }
        scheduler_values["iteration"] = self.iterations
        scheduler_equal = (
            scheduler_values["vendor_last_epoch"]
            == scheduler_values["package_last_epoch"]
            and scheduler_values["vendor_learning_rates"]
            == scheduler_values["package_learning_rates"]
        )
        self.rows.append(
            {
                "step": self.iterations,
                "epoch": trainer.current_epoch + 1,
                "batch_stream": {
                    "passed": self._batch_report.passed,
                    "checked_steps": self._batch_report.checked_steps,
                },
                "trace": _trace_compare(self._vendor_trace, package_trace),
                "gradients": _state_compare(self._vendor_gradients, package_gradients),
                "parameters": _state_compare(
                    self._vendor_parameters, package_parameters
                ),
                "optimizer_state": _state_compare(
                    self._vendor_optimizer_state, package_optimizer_state
                ),
                "scheduler": {**scheduler_values, "passed": scheduler_equal},
            }
        )
        trainer.should_stop = self.iterations >= LOCKSTEP_STEPS

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        for scheduler in self.vendor_schedulers:
            scheduler.step()


def _named_parameters(module: nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _named_gradients(module: nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _named_optimizer_state(
    optimizer: torch.optim.Optimizer, module: nn.Module, prefix: str
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                values[f"{prefix}.{name}.{key}"] = value.detach().clone()
    return values


def _named_optimizer_hyperparameters(
    optimizer: torch.optim.Optimizer, module: nn.Module, prefix: str
) -> dict[str, dict[str, Any]]:
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        f"{prefix}.{name}": {
            "lr": float(groups[id(parameter)]["lr"]),
            "betas": [float(value) for value in groups[id(parameter)]["betas"]],
            "eps": float(groups[id(parameter)]["eps"]),
            "maximize": bool(groups[id(parameter)].get("maximize", False)),
        }
        for name, parameter in module.named_parameters()
    }


def _adam_first_step_delta(gradient: float, hyperparameters: dict[str, Any]) -> float:
    value = torch.tensor(gradient, dtype=torch.float32)
    if hyperparameters["maximize"]:
        value = -value
    beta1, beta2 = hyperparameters["betas"]
    first_moment = (1.0 - beta1) * value
    second_moment = (1.0 - beta2) * value.square()
    step_size = hyperparameters["lr"] / (1.0 - beta1)
    denominator = second_moment.sqrt() / (1.0 - beta2) ** 0.5
    denominator = denominator + hyperparameters["eps"]
    return float((-step_size * first_moment / denominator).item())


def _fixed_batch(
    seed: int, device: torch.device
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, Any]
]:
    _set_determinism(seed)
    vendor_pre_loader_rng = _rng_digest(capture_rng_state())
    vendor_loader, _ = _vendor_loaders(seed)
    vendor_raw_batch = next(iter(vendor_loader))
    vendor_post_loader_rng = _rng_digest(capture_rng_state())
    _set_determinism(seed)
    package_pre_loader_rng = _rng_digest(capture_rng_state())
    package_loader, _ = _package_loaders(seed)
    package_raw_batch = next(iter(package_loader))
    package_post_loader_rng = _rng_digest(capture_rng_state())
    vendor_batch = _vendor_batch(vendor_raw_batch, device)
    package_batch = _package_batch(package_raw_batch, device)
    stream_report, stream = _batch_stream_report(package_batch, vendor_batch)
    if not stream_report.passed:
        raise RuntimeError(
            f"real train loader mismatch: {stream_report.first_mismatch}"
        )
    _set_determinism(seed)
    rng_before = capture_rng_state()
    initial_layout = (
        _vendor_main()
        .random_init(vendor_batch["pixel_values"].shape[0], MAX_ELEM)
        .to(device)
    )
    return (
        vendor_batch,
        package_batch,
        initial_layout,
        {
            "seed": seed,
            "rng_before_random_init": _rng_digest(rng_before),
            "initial_layout_sha256": tensor_sha256(initial_layout),
            "batch_stream": stream,
            "loader_rng": {
                "vendor": {
                    "pre_loader": vendor_pre_loader_rng,
                    "post_loader": vendor_post_loader_rng,
                    "first_sample_ids": _sample_ids(vendor_raw_batch[0]),
                    "generator": "torch global RNG used by vendor DataLoader",
                },
                "package": {
                    "pre_loader": package_pre_loader_rng,
                    "post_loader": package_post_loader_rng,
                    "first_sample_ids": _sample_ids(package_raw_batch["pixel_values"]),
                    "generator": (
                        "torch global RNG used by the parity overlay; production "
                        "DSGANDataModule uses a seeded torch.Generator"
                    ),
                    "production_seed": seed,
                },
            },
        },
    )


def run_s0() -> Path:
    _set_determinism(SEED)
    pre_model_rng = _rng_digest(capture_rng_state())
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(torch.device("cpu"))
    post_model_rng = _rng_digest(capture_rng_state())
    from ds_gan.training.lightning_module import DSGANTrainingModule

    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    )
    generator_keys_equal = set(vendor_generator.state_dict()) == set(
        package_generator.state_dict()
    )
    discriminator_keys_equal = set(vendor_discriminator.state_dict()) == set(
        package_discriminator.state_dict()
    )
    generator_count_vendor = sum(
        parameter.numel() for parameter in vendor_generator.parameters()
    )
    generator_count_package = sum(
        parameter.numel() for parameter in package_generator.parameters()
    )
    discriminator_count_vendor = sum(
        parameter.numel() for parameter in vendor_discriminator.parameters()
    )
    discriminator_count_package = sum(
        parameter.numel() for parameter in package_discriminator.parameters()
    )
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    vendor_schedulers = _schedulers(vendor_optimizers)
    package_optimizers, package_schedulers, package_scheduler_configs = (
        _package_optimizers_and_schedulers(package_module)
    )
    generator_initial_comparison = _state_compare(
        _named_parameters(vendor_generator, "generator"),
        _named_parameters(package_generator, "generator"),
    )
    discriminator_initial_comparison = _state_compare(
        _named_parameters(vendor_discriminator, "discriminator"),
        _named_parameters(package_discriminator, "discriminator"),
    )
    optimizer_static = {
        "vendor": [_optimizer_static(optimizer) for optimizer in vendor_optimizers],
        "package": [_optimizer_static(optimizer) for optimizer in package_optimizers],
    }
    optimizer_static_equal = optimizer_static["vendor"] == optimizer_static["package"]
    import hashlib

    def state_hash(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    generator_initial_hashes = {
        "vendor": state_hash(vendor_generator),
        "package": state_hash(package_generator),
    }
    discriminator_initial_hashes = {
        "vendor": state_hash(vendor_discriminator),
        "package": state_hash(package_discriminator),
    }
    vendor_batch, package_batch, _, batch_meta = _fixed_batch(SEED, torch.device("cpu"))
    batch_meta["model_rng"] = {
        "pre_model": pre_model_rng,
        "post_model": post_model_rng,
    }
    dataset_comparison = _trace_compare(
        {key: value for key, value in vendor_batch.items()},
        {key: value for key, value in package_batch.items()},
    )
    topology_pass = all(
        (
            generator_keys_equal,
            discriminator_keys_equal,
            generator_count_vendor == generator_count_package,
            discriminator_count_vendor == discriminator_count_package,
            generator_initial_comparison["passed"],
            discriminator_initial_comparison["passed"],
            optimizer_static_equal,
            dataset_comparison["passed"],
        )
    )
    return _write(
        "s0-static",
        {
            **_metadata(),
            "stage": "S0",
            "result": "PASS" if topology_pass else "FAIL",
            "topology": {
                "generator_state_keys_equal": generator_keys_equal,
                "discriminator_state_keys_equal": discriminator_keys_equal,
                "generator_parameter_count_equal": generator_count_vendor
                == generator_count_package,
                "discriminator_parameter_count_equal": discriminator_count_vendor
                == discriminator_count_package,
                "generator_parameter_count_vendor": generator_count_vendor,
                "generator_parameter_count_package": generator_count_package,
                "discriminator_parameter_count_vendor": discriminator_count_vendor,
                "discriminator_parameter_count_package": discriminator_count_package,
                "generator_initial_state": generator_initial_comparison,
                "discriminator_initial_state": discriminator_initial_comparison,
                "generator_initial_state_sha256": generator_initial_hashes,
                "discriminator_initial_state_sha256": discriminator_initial_hashes,
                "optimizer_static": optimizer_static,
                "optimizer_static_equal": optimizer_static_equal,
                "scheduler_static": {
                    "vendor": [
                        _scheduler_static(scheduler) for scheduler in vendor_schedulers
                    ],
                    "package": [
                        _scheduler_static(scheduler) for scheduler in package_schedulers
                    ],
                },
                "config": {
                    "generator": generator_config.to_dict(),
                    "discriminator": discriminator_config.to_dict(),
                },
            },
            "dataset_static": {"comparison": dataset_comparison, "batch": batch_meta},
        },
    )


def run_s1() -> Path:
    _set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(device)
    _copy_module_state(package_generator, vendor_generator)
    _copy_module_state(package_discriminator, vendor_discriminator)
    from ds_gan.training.lightning_module import DSGANTrainingModule

    vendor_batch, package_batch, initial_layout, batch_meta = _fixed_batch(SEED, device)
    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    with torch.no_grad():
        vendor_classes, vendor_boxes = vendor_generator(
            vendor_batch["pixel_values"], initial_layout
        )
        package_output = package_generator(
            package_batch["pixel_values"], initial_layout
        )
        vendor_score = vendor_discriminator(
            vendor_batch["pixel_values"],
            torch.stack((vendor_classes, vendor_boxes), dim=2),
        )
        package_score = package_discriminator(
            package_batch["pixel_values"],
            torch.stack((package_output.class_probs, package_output.bbox), dim=2),
        )
    with _capture_deterministic_warnings():
        vendor_losses = _vendor_criterion(device)(
            {"pred_logits": vendor_classes, "pred_boxes": vendor_boxes},
            _targets(vendor_batch),
        )
        package_losses = package_module.criterion(
            package_output.class_probs, package_output.bbox, _targets(package_batch)
        )
    vendor_trace = {
        "initial_layout": initial_layout,
        "class_probs": vendor_classes,
        "bbox": vendor_boxes,
        "discriminator_score": vendor_score,
        "loss_g_adv": nn.functional.hinge_embedding_loss(
            vendor_score.reshape(-1), torch.ones(vendor_score.shape[0], device=device)
        ),
        "loss_reconstruction": sum(vendor_losses.values()),
    }
    package_trace = {
        "initial_layout": initial_layout,
        "class_probs": package_output.class_probs,
        "bbox": package_output.bbox,
        "discriminator_score": package_score,
        "loss_g_adv": nn.functional.hinge_embedding_loss(
            package_score.reshape(-1), torch.ones(package_score.shape[0], device=device)
        ),
        "loss_reconstruction": sum(package_losses.values()),
    }
    comparison = _trace_compare(vendor_trace, package_trace)
    return _write(
        "s1-forward-loss",
        {
            **_metadata(),
            "stage": "S1",
            "result": "PASS" if comparison["passed"] else "FAIL",
            "batch": batch_meta,
            "comparison": comparison,
            "trace_fields": sorted(vendor_trace),
        },
    )


def run_s2() -> Path:
    _set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(device)
    _copy_module_state(package_generator, vendor_generator)
    _copy_module_state(package_discriminator, vendor_discriminator)
    from ds_gan.training.lightning_module import DSGANTrainingModule

    vendor_batch, package_batch, initial_layout, batch_meta = _fixed_batch(SEED, device)
    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    package_optimizers, package_schedulers, package_scheduler_configs = (
        _package_optimizers_and_schedulers(package_module)
    )
    vendor_schedulers = _schedulers(vendor_optimizers)
    with _capture_deterministic_warnings():
        vendor_trace = _vendor_step(
            vendor_generator,
            vendor_discriminator,
            vendor_batch,
            *vendor_optimizers,
            _vendor_criterion(device),
            1,
            initial_layout,
        )
        package_trace = _package_step(
            package_module, package_batch, *package_optimizers, 1, initial_layout
        )
    trace_comparison = _trace_compare(vendor_trace, package_trace)
    vendor_parameters = {
        **_named_parameters(vendor_generator, "generator"),
        **_named_parameters(vendor_discriminator, "discriminator"),
    }
    package_parameters = {
        **_named_parameters(package_generator, "generator"),
        **_named_parameters(package_discriminator, "discriminator"),
    }
    vendor_gradients = {
        **_named_gradients(vendor_generator, "generator"),
        **_named_gradients(vendor_discriminator, "discriminator"),
    }
    package_gradients = {
        **_named_gradients(package_generator, "generator"),
        **_named_gradients(package_discriminator, "discriminator"),
    }
    vendor_optimizer_state = {
        **_named_optimizer_state(vendor_optimizers[0], vendor_generator, "generator"),
        **_named_optimizer_state(
            vendor_optimizers[1], vendor_discriminator, "discriminator"
        ),
    }
    package_optimizer_state = {
        **_named_optimizer_state(package_optimizers[0], package_generator, "generator"),
        **_named_optimizer_state(
            package_optimizers[1], package_discriminator, "discriminator"
        ),
    }
    state_comparison = _state_compare(vendor_parameters, package_parameters)
    gradient_comparison = _state_compare(vendor_gradients, package_gradients)
    optimizer_comparison = _state_compare(
        vendor_optimizer_state, package_optimizer_state
    )
    vendor_scheduler_values = [
        _scheduler_static(scheduler) for scheduler in vendor_schedulers
    ]
    package_scheduler_values = [
        _scheduler_static(scheduler) for scheduler in package_schedulers
    ]
    scheduler_comparison = {
        "passed": vendor_scheduler_values == package_scheduler_values,
        "vendor": vendor_scheduler_values,
        "package": package_scheduler_values,
        "package_intervals": [
            config["interval"] for config in package_scheduler_configs
        ],
        "package_frequencies": [
            config["frequency"] for config in package_scheduler_configs
        ],
    }
    del (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        package_module,
        vendor_optimizers,
        package_optimizers,
        package_schedulers,
        vendor_schedulers,
        vendor_batch,
        package_batch,
        initial_layout,
        vendor_trace,
        package_trace,
        vendor_parameters,
        package_parameters,
        vendor_gradients,
        package_gradients,
        vendor_optimizer_state,
        package_optimizer_state,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    self_runs: dict[str, list[dict[str, Any]]] = {"vendor": [], "package": []}
    self_states: dict[str, list[dict[str, dict[str, torch.Tensor]]]] = {
        "vendor": [],
        "package": [],
    }
    for system in ("vendor", "package"):
        for repeat in range(1, S2_SELF_REPEATS + 1):
            run = _run_s2_self_repeat_process(system, repeat)
            self_states[system].append(
                cast(dict[str, dict[str, torch.Tensor]], run.pop("_state"))
            )
            self_runs[system].append(run)

    self_distributions = {
        system: {
            "loss_reconstruction": _scalar_self_distribution(
                runs, "loss_reconstruction"
            ),
            **{
                field: _state_distribution(self_states[system], field)
                for field in ("gradients", "parameters", "optimizer_state")
            },
        }
        for system, runs in self_runs.items()
    }
    cross_distributions = {
        "loss_reconstruction": _scalar_cross_distribution(
            self_runs["vendor"], self_runs["package"], "loss_reconstruction"
        ),
        **{
            field: _cross_state_distribution(
                self_states["vendor"], self_states["package"], field
            )
            for field in ("gradients", "parameters", "optimizer_state")
        },
    }
    inside_self_distributions = {
        field: _inside_self_distributions(
            cross_distributions[field],
            self_distributions["vendor"][field],
            self_distributions["package"][field],
        )
        for field in cross_distributions
    }
    elementwise_envelopes = {
        field: _elementwise_envelope(
            self_states["vendor"], self_states["package"], field
        )
        for field in ("gradients", "parameters")
    }
    parameter_update_mechanism = _parameter_update_mechanism(
        self_states["vendor"],
        self_states["package"],
        elementwise_envelopes["parameters"],
    )
    sign_straddling_probe = _sign_straddling_parameter_probe(
        self_states["vendor"], self_states["package"]
    )
    if sign_straddling_probe is not None:
        parameter_update_mechanism["sign_straddling_probe"] = (
            _parameter_update_mechanism(
                self_states["vendor"],
                self_states["package"],
                {"outside_self_envelope": [sign_straddling_probe]},
                include_maximum_probe=False,
            )["candidates"][0]
        )
    else:
        parameter_update_mechanism["sign_straddling_probe"] = None
    sign_straddling_probe = parameter_update_mechanism["sign_straddling_probe"]
    parameter_update_mechanism["sign_straddling_probe_passed"] = bool(
        sign_straddling_probe is not None
        and sign_straddling_probe["both_signs_occur"]
        and sign_straddling_probe["gradient_inside_like_for_like_envelope"]
        and sign_straddling_probe["adam_prediction_bitwise"]
    )
    vendor_operator = _vendor_cross_entropy_operator()
    package_differences = {
        "loss": trace_comparison,
        "gradients": gradient_comparison,
        "parameters": state_comparison,
        "optimizer_state": optimizer_comparison,
    }
    first_difference = next(
        (
            comparison["first_difference"]
            for comparison in package_differences.values()
            if comparison["first_difference"] is not None
        ),
        None,
    )
    self_processes_are_distinct = {
        system: len({run["process_id"] for run in runs}) == len(runs)
        for system, runs in self_runs.items()
    }
    self_source_is_current = {
        system: all(run["source_commit"] == _git("rev-parse", "HEAD") for run in runs)
        for system, runs in self_runs.items()
    }
    self_seed_is_paired = {
        system: all(run["seed"] == SEED for run in runs)
        for system, runs in self_runs.items()
    }
    self_layout_is_paired = {
        system: len({run["initial_layout_sha256"] for run in runs}) == 1
        for system, runs in self_runs.items()
    }
    self_warning_is_captured = {
        system: all(run["deterministic_warning"]["contains_nll_loss2d"] for run in runs)
        for system, runs in self_runs.items()
    }
    cross_layout_is_paired = all(
        vendor["initial_layout_sha256"] == package["initial_layout_sha256"]
        for vendor, package in zip(
            self_runs["vendor"], self_runs["package"], strict=True
        )
    )
    self_repeats_valid = (
        all(
            all(values.values())
            for values in (
                self_processes_are_distinct,
                self_source_is_current,
                self_seed_is_paired,
                self_layout_is_paired,
                self_warning_is_captured,
            )
        )
        and cross_layout_is_paired
    )
    cause = {
        "kind": "nondeterministic CUDA nll_loss2d cross_entropy reduction",
        "resolution_path": "B-envelope-fallback",
        "resolution_reason": (
            "strict deterministic S1 raised at the vendor cross_entropy operator; "
            "production uses the vendor operation and does not enable deterministic mode"
        ),
        "operator": vendor_operator,
        "package_operator": "torch.nn.functional.cross_entropy",
        "reduction": "mean (implicit default in both operators)",
        "first_difference": first_difference,
        "self_repeats": {
            "count": S2_SELF_REPEATS,
            "vendor": self_distributions["vendor"],
            "package": self_distributions["package"],
            "processes_are_distinct": self_processes_are_distinct,
            "source_is_current": self_source_is_current,
            "seed_is_paired": self_seed_is_paired,
            "initial_layout_is_paired": self_layout_is_paired,
            "warning_is_captured": self_warning_is_captured,
        },
        "cross_system_repeat_pairs": {
            "pair_count": S2_SELF_REPEATS,
            "distributions": cross_distributions,
            "inside_self_distributions": inside_self_distributions,
            "inside_self_distribution_criterion": (
                "cross max_abs_difference must be <= the larger corresponding "
                "vendor/package self envelope; relative differences and medians are diagnostic"
            ),
            "elementwise_envelopes": elementwise_envelopes,
            "parameter_update_mechanism": parameter_update_mechanism,
            "initial_layout_is_paired": cross_layout_is_paired,
        },
        "package_vendor_single_pair": {
            "comparisons": package_differences,
        },
        "self_repeats_valid": self_repeats_valid,
    }
    cause_passed = (
        all(inside_self_distributions.values())
        and elementwise_envelopes["gradients"]["cross_inside_combined_self_envelope"]
        and parameter_update_mechanism["passed"]
        and parameter_update_mechanism["sign_straddling_probe_passed"]
        and self_repeats_valid
    )
    passed = scheduler_comparison["passed"] and cause_passed
    return _write(
        "s2-optimizer-step",
        {
            **_metadata(),
            "stage": "S2",
            "result": "PASS" if passed else "FAIL",
            "batch": batch_meta,
            "trace": trace_comparison,
            "gradients": gradient_comparison,
            "post_step_parameters": state_comparison,
            "optimizer_state": optimizer_comparison,
            "scheduler": scheduler_comparison,
            "cause": cause,
        },
    )


def _run_natural(repeat: int, device: torch.device) -> dict[str, Any]:
    _set_determinism(SEED)
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(device)
    from ds_gan.training.lightning_module import DSGANTrainingModule

    _set_determinism(SEED)
    vendor_loader, _ = _vendor_loaders(SEED)
    _set_determinism(SEED)
    package_loader, _ = _package_loaders(SEED)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    vendor_schedulers = _schedulers(vendor_optimizers)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    callback = _NaturalParityCallback(
        vendor_generator,
        vendor_discriminator,
        vendor_loader,
        vendor_optimizers,
        vendor_schedulers,
        device,
    )
    trainer = Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision="32-true",
        max_epochs=LOCKSTEP_STEPS // TRAIN_BATCHES_PER_EPOCH + 2,
        limit_train_batches=TRAIN_BATCHES_PER_EPOCH,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        deterministic="warn",
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        callbacks=[callback],
    )
    with _capture_deterministic_warnings():
        trainer.fit(package_module, train_dataloaders=package_loader)
    if len(callback.rows) != LOCKSTEP_STEPS:
        raise RuntimeError(
            f"package Trainer produced {len(callback.rows)} iterations, expected {LOCKSTEP_STEPS}"
        )
    trace_rows = callback.rows
    first_divergence = next(
        (
            row
            for row in trace_rows
            if not all(
                comparison["passed"]
                for comparison in (
                    row["trace"],
                    row["gradients"],
                    row["parameters"],
                    row["optimizer_state"],
                    row["scheduler"],
                )
            )
        ),
        None,
    )
    trace_path = _write_jsonl(
        "s3-lockstep", f"natural-repeat-{repeat}.jsonl", trace_rows
    )
    max_abs = (
        max(
            comparison["first_difference"]["max_abs_diff"]
            for row in trace_rows
            for comparison in (
                row["trace"],
                row["gradients"],
                row["parameters"],
                row["optimizer_state"],
            )
            if comparison["first_difference"] is not None
        )
        if any(
            comparison["first_difference"] is not None
            for row in trace_rows
            for comparison in (
                row["trace"],
                row["gradients"],
                row["parameters"],
                row["optimizer_state"],
            )
        )
        else 0.0
    )
    return {
        "repeat": repeat,
        "steps": len(trace_rows),
        "first_divergence": first_divergence,
        "bitwise_300": first_divergence is None,
        "max_abs_difference": max_abs,
        "trace": str(trace_path.relative_to(ROOT)),
        "seed": SEED,
        "vendor_training_entry_point": (
            f"{Path(_vendor_main().__file__).resolve().relative_to(ROOT)}:"
            f"{_vendor_main().train.__qualname__}"
        ),
        "package_training_entry_point": (
            f"{type(package_module).__module__}.{type(package_module).__qualname__}.training_step"
        ),
        "package_trainer": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
        "package_optimizer_group_parameter_counts": [
            [len(group["params"]) for group in optimizer.param_groups]
            for optimizer in trainer.optimizers
        ],
        "deterministic_warning": _warning_record(),
    }


def _run_vendor_alone(repeat: int, device: torch.device) -> dict[str, Any]:
    _set_determinism(SEED)
    _vendor_main()
    vendor_generator, vendor_discriminator, _, _ = _independent_models("vendor", device)
    vendor_loader, _ = _vendor_loaders(SEED)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    vendor_schedulers = _schedulers(vendor_optimizers)
    vendor_iterator = iter(vendor_loader)
    for step in range(1, LOCKSTEP_STEPS + 1):
        epoch = (step - 1) // TRAIN_BATCHES_PER_EPOCH + 1
        if step > 1 and (step - 1) % TRAIN_BATCHES_PER_EPOCH == 0:
            vendor_iterator = iter(vendor_loader)
        vendor_batch = _vendor_batch(next(vendor_iterator), device)
        _vendor_step(
            vendor_generator,
            vendor_discriminator,
            vendor_batch,
            *vendor_optimizers,
            _vendor_criterion(device),
            epoch,
        )
        if step % TRAIN_BATCHES_PER_EPOCH == 0:
            for scheduler in vendor_schedulers:
                scheduler.step()
    return {
        "repeat": repeat,
        "steps": LOCKSTEP_STEPS,
        "seed": SEED,
        "vendor_training_entry_point": (
            f"{Path(_vendor_main().__file__).resolve().relative_to(ROOT)}:"
            f"{_vendor_main().train.__qualname__}"
        ),
        "final_parameters": {
            **_named_parameters(vendor_generator, "generator"),
            **_named_parameters(vendor_discriminator, "discriminator"),
        },
        "final_optimizer_state": {
            **_named_optimizer_state(
                vendor_optimizers[0], vendor_generator, "generator"
            ),
            **_named_optimizer_state(
                vendor_optimizers[1], vendor_discriminator, "discriminator"
            ),
        },
    }


class _StopAfterIterations(Callback):
    def __init__(self) -> None:
        self.iterations = 0

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        self.iterations += 1
        trainer.should_stop = self.iterations >= LOCKSTEP_STEPS


def _run_package_alone(repeat: int, device: torch.device) -> dict[str, Any]:
    from ds_gan.training.lightning_module import DSGANTrainingModule

    package_generator, package_discriminator, generator_config, discriminator_config = (
        _independent_models("package", device)
    )
    package_loader, _ = _package_loaders(SEED)
    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    stopper = _StopAfterIterations()
    trainer = Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision="32-true",
        max_epochs=LOCKSTEP_STEPS // TRAIN_BATCHES_PER_EPOCH + 2,
        limit_train_batches=TRAIN_BATCHES_PER_EPOCH,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        deterministic="warn",
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        callbacks=[stopper],
    )
    with _capture_deterministic_warnings():
        trainer.fit(package_module, train_dataloaders=package_loader)
    if stopper.iterations != LOCKSTEP_STEPS:
        raise RuntimeError(
            f"package Trainer produced {stopper.iterations} iterations, expected {LOCKSTEP_STEPS}"
        )
    return {
        "repeat": repeat,
        "steps": stopper.iterations,
        "seed": SEED,
        "package_training_entry_point": (
            f"{type(package_module).__module__}.{type(package_module).__qualname__}.training_step"
        ),
        "package_trainer": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
        "final_parameters": {
            **_named_parameters(package_module.generator, "generator"),
            **_named_parameters(package_module.discriminator, "discriminator"),
        },
        "final_optimizer_state": {
            **_named_optimizer_state(
                trainer.optimizers[0], package_module.generator, "generator"
            ),
            **_named_optimizer_state(
                trainer.optimizers[1], package_module.discriminator, "discriminator"
            ),
        },
        "deterministic_warning": _warning_record(),
    }


def _self_envelope(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    parameter_comparison = _state_compare(
        first["final_parameters"], second["final_parameters"]
    )
    optimizer_comparison = _state_compare(
        first["final_optimizer_state"], second["final_optimizer_state"]
    )
    return {
        "steps": [first["steps"], second["steps"]],
        "seed": [first["seed"], second["seed"]],
        "parameters": parameter_comparison,
        "optimizer_state": optimizer_comparison,
    }


def _envelope_run_metadata(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in run.items()
        if key not in {"final_parameters", "final_optimizer_state"}
    }


def _run_in_separate_process(stage: str, repeat: int) -> dict[str, Any]:
    artifact_root = EVIDENCE / "s3-lockstep"
    artifact_root.mkdir(parents=True, exist_ok=True)
    json_path = artifact_root / f"{stage}-{repeat}.json"
    state_path = artifact_root / f"{stage}-{repeat}.pt"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            stage,
            str(repeat),
            str(json_path),
            str(state_path),
        ],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(json_path.read_text())
    if stage == "s3-self-repeat":
        states = torch.load(state_path, map_location="cpu", weights_only=False)
        for system in ("vendor", "package"):
            payload[system].update(states[system])
    return payload


def run_s3_natural_repeat(repeat: int, json_path: Path, state_path: Path) -> None:
    del state_path
    result = _run_natural(
        repeat, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    result["process_id"] = os.getpid()
    result["source_commit"] = _git("rev-parse", "HEAD")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def run_s3_self_repeat(repeat: int, json_path: Path, state_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vendor = _run_vendor_alone(repeat, device)
    package = _run_package_alone(repeat, device)
    state = {
        "vendor": {
            "final_parameters": vendor.pop("final_parameters"),
            "final_optimizer_state": vendor.pop("final_optimizer_state"),
        },
        "package": {
            "final_parameters": package.pop("final_parameters"),
            "final_optimizer_state": package.pop("final_optimizer_state"),
        },
    }
    vendor["process_id"] = os.getpid()
    package["process_id"] = os.getpid()
    vendor["source_commit"] = _git("rev-parse", "HEAD")
    package["source_commit"] = _git("rev-parse", "HEAD")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, state_path)
    json_path.write_text(
        json.dumps({"vendor": vendor, "package": package}, indent=2, sort_keys=True)
        + "\n"
    )


def run_s3_production_wiring() -> Path:
    config_path = ROOT / "models/ds-gan/configs/training/ds_gan_pku_posterlayout.yaml"
    config_text = config_path.read_text()
    required_config_fragments = (
        "class_path: lightning.pytorch.loggers.CSVLogger",
        "class_path: lightning.pytorch.callbacks.ModelCheckpoint",
        "generator_backbone_weights: .cache/ds-gan/backbones/resnet50_a1_0-14fe96d1.pth",
        "discriminator_backbone_weights: .cache/ds-gan/backbones/resnet18-5c106cde.pth",
    )
    if any(fragment not in config_text for fragment in required_config_fragments):
        raise RuntimeError(
            "training config lacks the production logger, checkpoint, or backbone wiring"
        )
    executable = shutil.which("traingen")
    command = [
        executable or sys.executable,
        *([] if executable else ["-m", "traingen.lightning.cli"]),
        "fit",
        "--config",
        str(config_path),
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=1",
        "--trainer.devices=1",
        "--data.num_workers=0",
        "--trainer.enable_progress_bar=false",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_path = EVIDENCE / "s3-production-wiring" / "traingen-output.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout)
    checkpoint_root = CACHE / "training-runs" / "pku_posterlayout" / "checkpoints"
    checkpoints = sorted(checkpoint_root.glob("*.ckpt"))
    logger_root = CACHE / "training-runs" / "pku_posterlayout"
    checkpoint_scheduler_counts = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_scheduler_counts.append(
            {
                "path": str(checkpoint.relative_to(ROOT)),
                "count": len(payload.get("lr_schedulers", [])),
            }
        )
    passed = result.returncode == 0 and bool(checkpoints) and logger_root.is_dir()
    return _write(
        "s3-production-wiring",
        {
            **_metadata(),
            "stage": "S3",
            "evidence_layer": "production-wiring",
            "result": "PASS" if passed else "FAIL",
            "command": command,
            "returncode": result.returncode,
            "output": str(output_path.relative_to(ROOT)),
            "output_sha256": _sha256(output_path),
            "logger_root": str(logger_root.relative_to(ROOT)),
            "checkpoint_root": str(checkpoint_root.relative_to(ROOT)),
            "checkpoint_files": [str(path.relative_to(ROOT)) for path in checkpoints],
            "checkpoint_scheduler_counts": checkpoint_scheduler_counts,
            "config": str(config_path.relative_to(ROOT)),
            "traingen_entry_point": "traingen fit",
        },
    )


def run_s3() -> Path:
    repeats = [
        _run_in_separate_process("s3-natural-repeat", repeat) for repeat in (1, 2)
    ]
    self_runs = [
        _run_in_separate_process("s3-self-repeat", repeat) for repeat in (1, 2)
    ]
    natural_process_ids = [run["process_id"] for run in repeats]
    self_process_ids = [run["vendor"]["process_id"] for run in self_runs]
    if len(set(natural_process_ids)) != len(natural_process_ids):
        raise RuntimeError("natural S3 repeats did not use separate processes")
    if len(set(self_process_ids)) != len(self_process_ids):
        raise RuntimeError("S3 self repeats did not use separate processes")
    vendor_runs = [run["vendor"] for run in self_runs]
    package_runs = [run["package"] for run in self_runs]
    vendor_self = _self_envelope(vendor_runs[0], vendor_runs[1])
    package_self = _self_envelope(package_runs[0], package_runs[1])
    cross_system: list[dict[str, Any]] = [
        {
            "repeat": repeat,
            "parameters": _state_compare(
                vendor_run["final_parameters"], package_run["final_parameters"]
            ),
            "optimizer_state": _state_compare(
                vendor_run["final_optimizer_state"],
                package_run["final_optimizer_state"],
            ),
        }
        for repeat, vendor_run, package_run in zip(
            (1, 2), vendor_runs, package_runs, strict=True
        )
    ]
    cross_system_max = {
        "parameters": max(
            item["parameters"]["max_abs_difference"] for item in cross_system
        ),
        "optimizer_state": max(
            item["optimizer_state"]["max_abs_difference"] for item in cross_system
        ),
    }
    within_vendor_envelope = (
        cross_system_max["parameters"]
        <= vendor_self["parameters"]["max_abs_difference"]
        and cross_system_max["optimizer_state"]
        <= vendor_self["optimizer_state"]["max_abs_difference"]
    )
    within_package_envelope = (
        cross_system_max["parameters"]
        <= package_self["parameters"]["max_abs_difference"]
        and cross_system_max["optimizer_state"]
        <= package_self["optimizer_state"]["max_abs_difference"]
    )
    natural_pass = all(item["bitwise_300"] for item in repeats)
    return _write(
        "s3-lockstep",
        {
            **_metadata(),
            "stage": "S3",
            "evidence_layer": "natural",
            "result": "PASS" if natural_pass else "FAIL",
            "steps": LOCKSTEP_STEPS,
            "repeats": repeats,
            "repeat_process_ids": {
                "natural": natural_process_ids,
                "self": self_process_ids,
            },
            "repeat_run_envelope": {
                "cross_system_final_state": cross_system,
                "cross_system_final_state_max_abs": cross_system_max,
                "vendor_runs": [_envelope_run_metadata(run) for run in vendor_runs],
                "package_runs": [_envelope_run_metadata(run) for run in package_runs],
                "vendor_self": vendor_self,
                "package_self": package_self,
                "cross_system_within_vendor_self_envelope": within_vendor_envelope,
                "cross_system_within_package_self_envelope": within_package_envelope,
                "interpretation": (
                    "cross-system final-state drift is within both observed self envelopes"
                    if within_vendor_envelope and within_package_envelope
                    else "cross-system final-state drift exceeds at least one observed self envelope"
                ),
                "repeat_count": len(repeats),
            },
        },
    )


def _copy_optimizer_state(
    destination: torch.optim.Optimizer, source: torch.optim.Optimizer
) -> None:
    destination.load_state_dict(copy.deepcopy(source.state_dict()))
    source_ptrs = {
        value.data_ptr()
        for values in source.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    }
    destination_ptrs = {
        value.data_ptr()
        for values in destination.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    }
    if source_ptrs & destination_ptrs:
        raise RuntimeError("synchronized optimizer states share storage")


def _copy_scheduler_state(
    destination: torch.optim.lr_scheduler.MultiStepLR,
    source: torch.optim.lr_scheduler.MultiStepLR,
) -> None:
    destination.load_state_dict(copy.deepcopy(source.state_dict()))


def _optimizer_static(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "parameter_count": len(group["params"]),
            "lr": group["lr"],
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
            "amsgrad": group["amsgrad"],
        }
        for group in optimizer.param_groups
    ]


def _scheduler_static(
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
) -> dict[str, Any]:
    return {
        "milestones": sorted(scheduler.milestones.elements()),
        "gamma": scheduler.gamma,
        "last_epoch": scheduler.last_epoch,
        "learning_rates": [group["lr"] for group in scheduler.optimizer.param_groups],
    }


def _synchronized_step_once(
    vendor_generator: nn.Module,
    vendor_discriminator: nn.Module,
    package_module: Any,
    vendor_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    package_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    vendor_schedulers: tuple[
        torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR
    ],
    package_schedulers: tuple[
        torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR
    ],
    vendor_batch: dict[str, torch.Tensor],
    package_batch: dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    epoch: int,
) -> dict[str, Any]:
    _copy_module_state(package_module.generator, vendor_generator)
    _copy_module_state(package_module.discriminator, vendor_discriminator)
    _copy_optimizer_state(package_optimizers[0], vendor_optimizers[0])
    _copy_optimizer_state(package_optimizers[1], vendor_optimizers[1])
    _copy_scheduler_state(package_schedulers[0], vendor_schedulers[0])
    _copy_scheduler_state(package_schedulers[1], vendor_schedulers[1])
    vendor_rng = copy.deepcopy(capture_rng_state())
    vendor_trace = _vendor_step(
        vendor_generator,
        vendor_discriminator,
        vendor_batch,
        *vendor_optimizers,
        _vendor_criterion(device),
        epoch,
    )
    vendor_after_rng = copy.deepcopy(capture_rng_state())
    package_numpy_rng = np.random.RandomState()
    package_numpy_rng.set_state(vendor_rng.numpy)
    package_torch_generator = torch.Generator(device="cpu")
    package_torch_generator.set_state(vendor_rng.torch_cpu)
    package_before_rng = copy.deepcopy(capture_rng_state())
    package_trace = _package_step(
        package_module,
        package_batch,
        *package_optimizers,
        epoch,
        numpy_rng=package_numpy_rng,
        torch_generator=package_torch_generator,
    )
    package_after_rng = copy.deepcopy(capture_rng_state())
    vendor_rng_digest = _rng_digest(vendor_after_rng)
    package_rng_digest = _rng_digest(package_after_rng)
    restore_rng_state(vendor_after_rng)
    synchronized_rng = copy.deepcopy(capture_rng_state())
    package_numpy_state = cast(
        tuple[str, np.ndarray, int, int, float], package_numpy_rng.get_state()
    )
    trace_comparison = _trace_compare(vendor_trace, package_trace)
    gradient_comparison = _state_compare(
        {
            **_named_gradients(vendor_generator, "generator"),
            **_named_gradients(vendor_discriminator, "discriminator"),
        },
        {
            **_named_gradients(package_module.generator, "generator"),
            **_named_gradients(package_module.discriminator, "discriminator"),
        },
    )
    parameter_comparison = _state_compare(
        {
            **_named_parameters(vendor_generator, "generator"),
            **_named_parameters(vendor_discriminator, "discriminator"),
        },
        {
            **_named_parameters(package_module.generator, "generator"),
            **_named_parameters(package_module.discriminator, "discriminator"),
        },
    )
    optimizer_comparison = _state_compare(
        {
            **_named_optimizer_state(
                vendor_optimizers[0], vendor_generator, "generator"
            ),
            **_named_optimizer_state(
                vendor_optimizers[1], vendor_discriminator, "discriminator"
            ),
        },
        {
            **_named_optimizer_state(
                package_optimizers[0], package_module.generator, "generator"
            ),
            **_named_optimizer_state(
                package_optimizers[1], package_module.discriminator, "discriminator"
            ),
        },
    )
    per_step_deltas = {
        "trace_max_abs": trace_comparison["max_abs_difference"],
        "gradient_max_abs": gradient_comparison["max_abs_difference"],
        "parameter_max_abs": parameter_comparison["max_abs_difference"],
        "optimizer_state_max_abs": optimizer_comparison["max_abs_difference"],
    }
    return {
        "step": step,
        "epoch": epoch,
        "_vendor_trace": vendor_trace,
        "_package_trace": package_trace,
        "trace": trace_comparison,
        "gradients": gradient_comparison,
        "parameters": parameter_comparison,
        "optimizer_state": optimizer_comparison,
        "per_step_deltas": per_step_deltas,
        "rng_before": {
            "vendor": _rng_digest(vendor_rng),
            "package": _rng_digest(package_before_rng),
        },
        "rng_after": {"vendor": vendor_rng_digest, "package": package_rng_digest},
        "rng_equal_after_operation": vendor_rng_digest == package_rng_digest,
        "rng_after_resynchronization": _rng_digest(synchronized_rng),
        "rng_equal_after_resynchronization": vendor_rng_digest
        == _rng_digest(synchronized_rng),
        "explicit_generator_after": {
            "torch_cpu": tensor_sha256(package_torch_generator.get_state()),
            "numpy": tensor_sha256(
                torch.as_tensor(
                    package_numpy_state[1],
                    dtype=torch.uint32,
                )
            ),
        },
        "learning_rates": {
            "vendor": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in vendor_optimizers
            ],
            "package": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in package_optimizers
            ],
        },
        "scheduler": {
            "vendor": [_scheduler_static(scheduler) for scheduler in vendor_schedulers],
            "package": [
                _scheduler_static(scheduler) for scheduler in package_schedulers
            ],
        },
    }


def _synchronized_repeat_envelope(
    first: dict[str, Any],
    second: dict[str, Any],
    first_state: dict[str, dict[str, dict[str, torch.Tensor]]],
    second_state: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> dict[str, Any]:
    self_comparisons = {
        "trace": {
            "vendor": _trace_compare(first["_vendor_trace"], second["_vendor_trace"]),
            "package": _trace_compare(
                first["_package_trace"], second["_package_trace"]
            ),
        },
        "gradients": {
            "vendor": _state_max_summary(
                first_state["vendor"]["gradients"], second_state["vendor"]["gradients"]
            ),
            "package": _state_max_summary(
                first_state["package"]["gradients"],
                second_state["package"]["gradients"],
            ),
        },
        "optimizer_state": {
            "vendor": _state_max_summary(
                first_state["vendor"]["optimizer_state"],
                second_state["vendor"]["optimizer_state"],
            ),
            "package": _state_max_summary(
                first_state["package"]["optimizer_state"],
                second_state["package"]["optimizer_state"],
            ),
        },
        "parameters": {
            "vendor": _state_max_summary(
                first_state["vendor"]["parameters"],
                second_state["vendor"]["parameters"],
            ),
            "package": _state_max_summary(
                first_state["package"]["parameters"],
                second_state["package"]["parameters"],
            ),
        },
    }
    envelope: dict[str, Any] = {}
    for field in ("trace", "gradients", "optimizer_state", "parameters"):
        cross = _distribution_summary(
            [first[field]["max_abs_difference"], second[field]["max_abs_difference"]]
        )
        vendor_self = _distribution_summary(
            [self_comparisons[field]["vendor"]["max_abs_difference"]]
        )
        package_self = _distribution_summary(
            [self_comparisons[field]["package"]["max_abs_difference"]]
        )
        envelope[field] = {
            "cross_system": cross,
            "vendor_self": vendor_self,
            "package_self": package_self,
            "cross_inside_vendor_self_envelope": (
                cross["max_abs_difference"] <= vendor_self["max_abs_difference"]
                and cross["median_abs_difference"]
                <= vendor_self["median_abs_difference"]
            ),
            "cross_inside_package_self_envelope": (
                cross["max_abs_difference"] <= package_self["max_abs_difference"]
                and cross["median_abs_difference"]
                <= package_self["median_abs_difference"]
            ),
        }
    return envelope


def _synchronized_repeat_step(
    first_args: tuple[Any, ...],
    second_pair: dict[str, Any],
) -> dict[str, Any]:
    pre_rng = copy.deepcopy(capture_rng_state())
    first = _synchronized_step_once(*first_args)
    first_rng = copy.deepcopy(capture_rng_state())
    first_state = _synchronized_state_snapshot(
        first_args[0],
        first_args[1],
        first_args[2],
        first_args[3],
        first_args[4],
    )
    restore_rng_state(pre_rng)
    second = _synchronized_step_once(
        second_pair["vendor_generator"],
        second_pair["vendor_discriminator"],
        second_pair["package_module"],
        second_pair["vendor_optimizers"],
        second_pair["package_optimizers"],
        second_pair["vendor_schedulers"],
        second_pair["package_schedulers"],
        first_args[7],
        first_args[8],
        first_args[9],
        first_args[10],
        first_args[11],
    )
    second_state = _synchronized_state_snapshot(
        second_pair["vendor_generator"],
        second_pair["vendor_discriminator"],
        second_pair["package_module"],
        second_pair["vendor_optimizers"],
        second_pair["package_optimizers"],
    )
    restore_rng_state(first_rng)
    first_public = {
        key: value for key, value in first.items() if not key.startswith("_")
    }
    return {
        **first_public,
        "repeat_trace": second["trace"],
        "repeat_scheduler": second["scheduler"],
        "repeat_rng_equal_after_operation": second["rng_equal_after_operation"],
        "repeat_rng_equal_after_resynchronization": second[
            "rng_equal_after_resynchronization"
        ],
        "repeat_run_envelope": _synchronized_repeat_envelope(
            first, second, first_state, second_state
        ),
    }


def _synchronized_state_snapshot(
    vendor_generator: nn.Module,
    vendor_discriminator: nn.Module,
    package_module: Any,
    vendor_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    package_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    return {
        "vendor": {
            "gradients": {
                **_named_gradients(vendor_generator, "generator"),
                **_named_gradients(vendor_discriminator, "discriminator"),
            },
            "parameters": {
                **_named_parameters(vendor_generator, "generator"),
                **_named_parameters(vendor_discriminator, "discriminator"),
            },
            "optimizer_state": {
                **_named_optimizer_state(
                    vendor_optimizers[0], vendor_generator, "generator"
                ),
                **_named_optimizer_state(
                    vendor_optimizers[1], vendor_discriminator, "discriminator"
                ),
            },
        },
        "package": {
            "gradients": {
                **_named_gradients(package_module.generator, "generator"),
                **_named_gradients(package_module.discriminator, "discriminator"),
            },
            "parameters": {
                **_named_parameters(package_module.generator, "generator"),
                **_named_parameters(package_module.discriminator, "discriminator"),
            },
            "optimizer_state": {
                **_named_optimizer_state(
                    package_optimizers[0], package_module.generator, "generator"
                ),
                **_named_optimizer_state(
                    package_optimizers[1], package_module.discriminator, "discriminator"
                ),
            },
        },
    }


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "values": values,
        "max_abs_difference": max(values),
        "median_abs_difference": median(values),
    }


def _state_max_summary(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> dict[str, float]:
    missing = set(reference) - set(target)
    if missing:
        raise RuntimeError(f"state comparison is missing keys: {sorted(missing)}")
    return {
        "max_abs_difference": max(
            float((target[name].float() - reference[name].float()).abs().max().item())
            for name in reference
        )
    }


def _synchronized_step(
    vendor_generator: nn.Module,
    vendor_discriminator: nn.Module,
    package_module: Any,
    vendor_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    package_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    vendor_schedulers: tuple[
        torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR
    ],
    package_schedulers: tuple[
        torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.MultiStepLR
    ],
    vendor_batch: dict[str, torch.Tensor],
    package_batch: dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    epoch: int,
    repeat_pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if repeat_pair is not None:
        return _synchronized_repeat_step(
            (
                vendor_generator,
                vendor_discriminator,
                package_module,
                vendor_optimizers,
                package_optimizers,
                vendor_schedulers,
                package_schedulers,
                vendor_batch,
                package_batch,
                device,
                step,
                epoch,
            ),
            repeat_pair,
        )

    pre_model_state = {
        "generator": copy.deepcopy(vendor_generator.state_dict()),
        "discriminator": copy.deepcopy(vendor_discriminator.state_dict()),
    }
    pre_vendor_optimizer_state = tuple(
        copy.deepcopy(optimizer.state_dict()) for optimizer in vendor_optimizers
    )
    pre_package_optimizer_state = tuple(
        copy.deepcopy(optimizer.state_dict()) for optimizer in package_optimizers
    )
    pre_vendor_scheduler_state = tuple(
        copy.deepcopy(scheduler.state_dict()) for scheduler in vendor_schedulers
    )
    pre_package_scheduler_state = tuple(
        copy.deepcopy(scheduler.state_dict()) for scheduler in package_schedulers
    )
    pre_rng = copy.deepcopy(capture_rng_state())

    first = _synchronized_step_once(
        vendor_generator,
        vendor_discriminator,
        package_module,
        vendor_optimizers,
        package_optimizers,
        vendor_schedulers,
        package_schedulers,
        vendor_batch,
        package_batch,
        device,
        step,
        epoch,
    )
    first_rng = copy.deepcopy(capture_rng_state())
    first_state = _synchronized_state_snapshot(
        vendor_generator,
        vendor_discriminator,
        package_module,
        vendor_optimizers,
        package_optimizers,
    )
    first_model_state = {
        "vendor_generator": copy.deepcopy(vendor_generator.state_dict()),
        "vendor_discriminator": copy.deepcopy(vendor_discriminator.state_dict()),
        "package_generator": copy.deepcopy(package_module.generator.state_dict()),
        "package_discriminator": copy.deepcopy(
            package_module.discriminator.state_dict()
        ),
    }
    first_vendor_optimizer_state = tuple(
        copy.deepcopy(optimizer.state_dict()) for optimizer in vendor_optimizers
    )
    first_package_optimizer_state = tuple(
        copy.deepcopy(optimizer.state_dict()) for optimizer in package_optimizers
    )
    first_vendor_scheduler_state = tuple(
        copy.deepcopy(scheduler.state_dict()) for scheduler in vendor_schedulers
    )
    first_package_scheduler_state = tuple(
        copy.deepcopy(scheduler.state_dict()) for scheduler in package_schedulers
    )

    def restore_pre_step() -> None:
        vendor_generator.load_state_dict(copy.deepcopy(pre_model_state["generator"]))
        vendor_discriminator.load_state_dict(
            copy.deepcopy(pre_model_state["discriminator"])
        )
        package_module.generator.load_state_dict(
            copy.deepcopy(pre_model_state["generator"])
        )
        package_module.discriminator.load_state_dict(
            copy.deepcopy(pre_model_state["discriminator"])
        )
        for optimizer, state in zip(
            vendor_optimizers, pre_vendor_optimizer_state, strict=True
        ):
            optimizer.load_state_dict(copy.deepcopy(state))
        for optimizer, state in zip(
            package_optimizers, pre_package_optimizer_state, strict=True
        ):
            optimizer.load_state_dict(copy.deepcopy(state))
        for scheduler, state in zip(
            vendor_schedulers, pre_vendor_scheduler_state, strict=True
        ):
            scheduler.load_state_dict(copy.deepcopy(state))
        for scheduler, state in zip(
            package_schedulers, pre_package_scheduler_state, strict=True
        ):
            scheduler.load_state_dict(copy.deepcopy(state))

    restore_pre_step()
    restore_rng_state(pre_rng)
    second = _synchronized_step_once(
        vendor_generator,
        vendor_discriminator,
        package_module,
        vendor_optimizers,
        package_optimizers,
        vendor_schedulers,
        package_schedulers,
        vendor_batch,
        package_batch,
        device,
        step,
        epoch,
    )
    second_state = _synchronized_state_snapshot(
        vendor_generator,
        vendor_discriminator,
        package_module,
        vendor_optimizers,
        package_optimizers,
    )

    vendor_gradient_self = _state_compare(
        first_state["vendor"]["gradients"], second_state["vendor"]["gradients"]
    )
    package_gradient_self = _state_compare(
        first_state["package"]["gradients"], second_state["package"]["gradients"]
    )
    vendor_optimizer_self = _state_compare(
        first_state["vendor"]["optimizer_state"],
        second_state["vendor"]["optimizer_state"],
    )
    package_optimizer_self = _state_compare(
        first_state["package"]["optimizer_state"],
        second_state["package"]["optimizer_state"],
    )
    vendor_parameter_self = _state_compare(
        first_state["vendor"]["parameters"], second_state["vendor"]["parameters"]
    )
    package_parameter_self = _state_compare(
        first_state["package"]["parameters"],
        second_state["package"]["parameters"],
    )
    self_comparisons = {
        "gradients": {
            "vendor": vendor_gradient_self,
            "package": package_gradient_self,
        },
        "optimizer_state": {
            "vendor": vendor_optimizer_self,
            "package": package_optimizer_self,
        },
        "parameters": {
            "vendor": vendor_parameter_self,
            "package": package_parameter_self,
        },
    }
    repeat_envelope: dict[str, Any] = {}
    for field in ("trace", "gradients", "optimizer_state", "parameters"):
        cross_values = [
            first[field]["max_abs_difference"],
            second[field]["max_abs_difference"],
        ]
        vendor_self = self_comparisons[field]["vendor"]
        package_self = self_comparisons[field]["package"]
        cross = _distribution_summary(cross_values)
        vendor_distribution = _distribution_summary([vendor_self["max_abs_difference"]])
        package_distribution = _distribution_summary(
            [package_self["max_abs_difference"]]
        )
        repeat_envelope[field] = {
            "cross_system": cross,
            "vendor_self": vendor_distribution,
            "package_self": package_distribution,
            "cross_inside_vendor_self_envelope": (
                cross["max_abs_difference"] <= vendor_distribution["max_abs_difference"]
                and cross["median_abs_difference"]
                <= vendor_distribution["median_abs_difference"]
            ),
            "cross_inside_package_self_envelope": (
                cross["max_abs_difference"]
                <= package_distribution["max_abs_difference"]
                and cross["median_abs_difference"]
                <= package_distribution["median_abs_difference"]
            ),
        }

    vendor_generator.load_state_dict(first_model_state["vendor_generator"])
    vendor_discriminator.load_state_dict(first_model_state["vendor_discriminator"])
    package_module.generator.load_state_dict(first_model_state["package_generator"])
    package_module.discriminator.load_state_dict(
        first_model_state["package_discriminator"]
    )
    for optimizer, state in zip(
        vendor_optimizers, first_vendor_optimizer_state, strict=True
    ):
        optimizer.load_state_dict(state)
    for optimizer, state in zip(
        package_optimizers, first_package_optimizer_state, strict=True
    ):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(
        vendor_schedulers, first_vendor_scheduler_state, strict=True
    ):
        scheduler.load_state_dict(state)
    for scheduler, state in zip(
        package_schedulers, first_package_scheduler_state, strict=True
    ):
        scheduler.load_state_dict(state)
    restore_rng_state(first_rng)

    return {
        **first,
        "repeat_trace": second["trace"],
        "repeat_scheduler": second["scheduler"],
        "repeat_rng_equal_after_operation": second["rng_equal_after_operation"],
        "repeat_rng_equal_after_resynchronization": second[
            "rng_equal_after_resynchronization"
        ],
        "repeat_run_envelope": repeat_envelope,
    }


def run_s3_synchronized() -> Path:
    _set_determinism(SEED)
    _vendor_main()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(device)
    from ds_gan.training.lightning_module import DSGANTrainingModule

    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    (
        repeat_vendor_generator,
        repeat_vendor_discriminator,
        repeat_package_generator,
        repeat_package_discriminator,
        repeat_generator_config,
        repeat_discriminator_config,
    ) = _models(device)
    repeat_package_module = DSGANTrainingModule(
        config=repeat_generator_config,
        discriminator_config=repeat_discriminator_config,
        generator=repeat_package_generator,
        discriminator=repeat_package_discriminator,
    ).to(device)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    package_optimizers, package_schedulers, _ = _package_optimizers_and_schedulers(
        package_module
    )
    vendor_schedulers = _schedulers(vendor_optimizers)
    repeat_vendor_optimizers = _optimizers(
        repeat_vendor_generator, repeat_vendor_discriminator
    )
    repeat_package_optimizers, repeat_package_schedulers, _ = (
        _package_optimizers_and_schedulers(repeat_package_module)
    )
    repeat_vendor_schedulers = _schedulers(repeat_vendor_optimizers)
    repeat_pair = {
        "vendor_generator": repeat_vendor_generator,
        "vendor_discriminator": repeat_vendor_discriminator,
        "package_module": repeat_package_module,
        "vendor_optimizers": repeat_vendor_optimizers,
        "package_optimizers": repeat_package_optimizers,
        "vendor_schedulers": repeat_vendor_schedulers,
        "package_schedulers": repeat_package_schedulers,
    }
    _set_determinism(SEED)
    vendor_loader, _ = _vendor_loaders(SEED)
    _set_determinism(SEED)
    package_loader, _ = _package_loaders(SEED)
    vendor_iterator = iter(vendor_loader)
    package_iterator = iter(package_loader)
    rows: list[dict[str, Any]] = []
    pre_synchronization_mismatches = 0
    for step in range(1, LOCKSTEP_STEPS + 1):
        epoch = (step - 1) // TRAIN_BATCHES_PER_EPOCH + 1
        if step > 1 and (step - 1) % TRAIN_BATCHES_PER_EPOCH == 0:
            vendor_iterator = iter(vendor_loader)
            package_iterator = iter(package_loader)
        vendor_batch = _vendor_batch(next(vendor_iterator), device)
        package_batch = _package_batch(next(package_iterator), device)
        batch_report, _ = _batch_stream_report(package_batch, vendor_batch)
        if not batch_report.passed:
            pre_synchronization_mismatches += 1
            package_batch = {
                key: value.detach().clone() for key, value in vendor_batch.items()
            }
        row = _synchronized_step(
            vendor_generator,
            vendor_discriminator,
            package_module,
            vendor_optimizers,
            package_optimizers,
            vendor_schedulers,
            package_schedulers,
            vendor_batch,
            package_batch,
            device,
            step,
            epoch,
            repeat_pair,
        )
        row["batch_stream"] = {
            "passed_before_synchronization": batch_report.passed,
            "synchronized_batch_source": (
                "independent_package_loader"
                if batch_report.passed
                else "vendor_batch_copy"
            ),
        }
        rows.append(row)
        if step % TRAIN_BATCHES_PER_EPOCH == 0:
            for scheduler in (*vendor_schedulers, *package_schedulers):
                scheduler.step()
            for scheduler in (*repeat_vendor_schedulers, *repeat_package_schedulers):
                scheduler.step()
    trace_path = _write_jsonl("s3-lockstep-synchronized", "trace.jsonl", rows)
    synchronized_envelope: dict[str, Any] = {}
    for field in ("gradients", "optimizer_state", "parameters"):
        cross_values = [
            value
            for row in rows
            for value in row["repeat_run_envelope"][field]["cross_system"]["values"]
        ]
        vendor_values = [
            row["repeat_run_envelope"][field]["vendor_self"]["max_abs_difference"]
            for row in rows
        ]
        package_values = [
            row["repeat_run_envelope"][field]["package_self"]["max_abs_difference"]
            for row in rows
        ]
        cross = _distribution_summary(cross_values)
        vendor_self = _distribution_summary(vendor_values)
        package_self = _distribution_summary(package_values)
        synchronized_envelope[field] = {
            "cross_system": cross,
            "vendor_self": vendor_self,
            "package_self": package_self,
            "cross_inside_vendor_self_envelope": (
                cross["max_abs_difference"] <= vendor_self["max_abs_difference"]
                and cross["median_abs_difference"]
                <= vendor_self["median_abs_difference"]
            ),
            "cross_inside_package_self_envelope": (
                cross["max_abs_difference"] <= package_self["max_abs_difference"]
                and cross["median_abs_difference"]
                <= package_self["median_abs_difference"]
            ),
        }
    trace_bitwise = all(
        row["trace"]["passed"] and row["repeat_trace"]["passed"] for row in rows
    )
    rng_pass = all(
        row["rng_equal_after_resynchronization"]
        and row["repeat_rng_equal_after_resynchronization"]
        for row in rows
    )
    scheduler_pass = all(
        row["scheduler"]["vendor"] == row["scheduler"]["package"]
        and row["repeat_scheduler"]["vendor"] == row["repeat_scheduler"]["package"]
        for row in rows
    )
    numeric_envelope_pass = all(
        field_summary["cross_inside_vendor_self_envelope"]
        and field_summary["cross_inside_package_self_envelope"]
        for field, field_summary in synchronized_envelope.items()
        if field != "trace"
    )
    trace_envelope_pass = (
        synchronized_envelope["trace"]["cross_inside_vendor_self_envelope"]
        and synchronized_envelope["trace"]["cross_inside_package_self_envelope"]
    )
    passed = (
        trace_envelope_pass and rng_pass and scheduler_pass and numeric_envelope_pass
    )
    first_divergence = next(
        (
            row
            for row in rows
            if not (
                row["rng_equal_after_resynchronization"]
                and row["repeat_rng_equal_after_resynchronization"]
                and row["scheduler"]["vendor"] == row["scheduler"]["package"]
                and row["repeat_scheduler"]["vendor"]
                == row["repeat_scheduler"]["package"]
            )
        ),
        None,
    )
    return _write(
        "s3-lockstep-synchronized",
        {
            **_metadata(),
            "stage": "S3",
            "evidence_layer": "synchronized",
            "result": "PASS" if passed else "FAIL",
            "steps": len(rows),
            "first_divergence": first_divergence,
            "trace": str(trace_path.relative_to(ROOT)),
            "per_step_equivalence": {
                "trace_bitwise": trace_bitwise,
                "trace_envelope_pass": trace_envelope_pass,
                "rng_bitwise": rng_pass,
                "scheduler_bitwise": scheduler_pass,
                "numeric_state_gate": "cross distributions inside both self envelopes",
                "numeric_state_pass": numeric_envelope_pass,
            },
            "repeat_run_envelope": synchronized_envelope,
            "cause": {
                "operation": "CUDA nll_loss2d",
                "systems": "vendor and package",
                "mode": "envelope",
                "strict_determinism_probe": "raises because nll_loss2d has no deterministic CUDA implementation",
            },
            "natural_record": ".cache/ds-gan/stage-evidence/s3-lockstep/run.json",
            "pre_synchronization_loader_mismatch_steps": pre_synchronization_mismatches,
            "synchronized_batch_policy": (
                "copy vendor batch into the package step when independent loader streams differ"
            ),
        },
    )


def _metric_summary(classes: np.ndarray, boxes: np.ndarray) -> dict[str, float | int]:
    valid = classes > 0
    out_of_bounds = (
        (boxes[..., 0] < 0)
        | (boxes[..., 1] < 0)
        | (boxes[..., 2] > 513)
        | (boxes[..., 3] > 750)
    )
    return {
        "prediction_count": int(valid.sum()),
        "sample_count": int(classes.shape[0]),
        "out_of_bounds_count": int((out_of_bounds & valid).sum()),
        "mean_pixel_box": float(boxes.mean()),
    }


def _metrics_from_eval_output(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in re.findall(
        r"^(metrics_[a-z_]+):\s*([-+0-9.eE]+)$", output, re.MULTILINE
    ):
        metrics[name] = float(value)
    expected = {
        "metrics_val",
        "metrics_ove",
        "metrics_ali",
        "metrics_und_l",
        "metrics_und_s",
        "metrics_uti",
        "metrics_occ",
        "metrics_rea",
    }
    if set(metrics) != expected:
        raise RuntimeError(
            f"eval.py:main output did not contain exactly the expected metrics: {sorted(metrics)}"
        )
    return metrics


def _prediction_file(
    path: Path, classes: np.ndarray, boxes: np.ndarray
) -> dict[str, Any]:
    np.savez(path, classes=classes, boxes_xyxy_pixels=boxes)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "rows": int(classes.shape[0]),
        "class_shape": list(classes.shape),
        "box_shape": list(boxes.shape),
    }


def _module_state_hash(module: nn.Module) -> str:
    digest = __import__("hashlib").sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_s4_bridge() -> Path:
    manifest = _materialize_bridge()
    paths = _bridge_paths()
    stale_vendor_write_paths = [
        str(path.relative_to(paths["root"]))
        for path in (paths["root"] / "output", paths["root"] / "test_order.pt")
        if path.exists()
    ]
    _make_bridge_read_only()
    return _write(
        "s4-bridge",
        {
            **_metadata(),
            "stage": "S4",
            "result": "PASS" if not manifest["missing_source_fields"] else "BLOCKED",
            "manifest": manifest,
            "manifest_file_sha256": _sha256(paths["manifest"]),
            "source_artifact_hashes": manifest["hashes"],
            "bridge_read_only": True,
            "vendor_write_overlay": str(
                (EVIDENCE / "s4-evaluation" / "vendor-overlay").relative_to(ROOT)
            ),
            "stale_vendor_write_paths": stale_vendor_write_paths,
        },
    )


def run_s4() -> Path:
    _set_determinism(SEED)
    paths = _bridge_paths()
    manifest = json.loads(paths["manifest"].read_text())
    manifest_file_hash = _sha256(paths["manifest"])
    source_hashes = _source_artifact_hashes()
    source_hashes_verified = source_hashes == manifest["hashes"]
    bridge_hashes = {
        "train.inpainted_poster": _aggregate_files(
            list(paths["train_images"].glob("*.png")), paths["root"]
        ),
        "train.saliency_pfpnet": _aggregate_files(
            list(paths["train_pfpn"].glob("*.png")), paths["root"]
        ),
        "train.saliency_basnet": _aggregate_files(
            list(paths["train_basnet"].glob("*.png")), paths["root"]
        ),
        "train.annotations": manifest["bridge"]["train.annotations"],
        "test.canvas": _aggregate_files(
            list(paths["test_images"].glob("*.png")), paths["root"]
        ),
        "test.saliency_pfpnet": _aggregate_files(
            list(paths["test_pfpn"].glob("*.png")), paths["root"]
        ),
        "test.saliency_basnet": _aggregate_files(
            list(paths["test_basnet"].glob("*.png")), paths["root"]
        ),
    }
    bridge_hashes_verified = bridge_hashes == manifest["bridge"]
    if not source_hashes_verified:
        raise RuntimeError("approved-source Arrow or metadata hash verification failed")
    if not bridge_hashes_verified:
        raise RuntimeError("bridge PNG or annotation hash verification failed")
    if manifest["missing_source_fields"]:
        raise RuntimeError(
            f"approved source lacks vendor fields: {manifest['missing_source_fields']}"
        )
    checkpoint_path = Path(
        os.environ.get("DSGAN_CHECKPOINT", ".cache/ds-gan/original/DS-GAN-Epoch300.pth")
    )
    converted_path = Path(
        os.environ.get(
            "DSGAN_CONVERTED_DIR", ".cache/ds-gan/converted/ds-gan-pku-posterlayout"
        )
    )
    if not checkpoint_path.exists() or not converted_path.exists():
        raise FileNotFoundError(
            "released checkpoint and converted package checkpoint are required"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ds_gan import DSGANPipeline

    pipeline = DSGANPipeline.from_pretrained(converted_path, local_files_only=True).to(
        device
    )
    vendor_generator_class, _, _, _ = _vendor_classes()
    _, _, generator_args, _ = _configs()
    with _vendor_backbone_loader():
        vendor_generator = vendor_generator_class(generator_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vendor_generator.load_state_dict(
        {key.removeprefix("module."): value for key, value in checkpoint.items()},
        strict=True,
    )
    vendor_generator.eval()
    initial_layout_rng_before = _rng_digest(capture_rng_state())
    initial_layout = _vendor_main().random_init(TEST_BATCH_SIZE, MAX_ELEM).to(device)
    initial_layout_rng_after = _rng_digest(capture_rng_state())
    vendor_root = EVIDENCE / "s4-evaluation" / "vendor-overlay"
    vendor_root.mkdir(parents=True, exist_ok=True)
    (vendor_root / "output").mkdir(parents=True, exist_ok=True)
    _, vendor_eval_loader = _vendor_loaders(SEED, overlay_root=vendor_root)
    names = list(torch.load(vendor_root / "test_order.pt", weights_only=False))
    _, package_loader = _package_loaders(SEED)
    _, vendor_stream_loader = _vendor_loaders(SEED)
    with redirect_stdout(io.StringIO()) as output:
        sys.path.insert(0, str(VENDOR))
        import infer as vendor_infer
        import eval as vendor_eval

        vendor_infer.device = device
        vendor_infer.fix_noise = initial_layout
        os.chdir(vendor_root)
        vendor_infer.test(vendor_generator, vendor_eval_loader, 1)
        vendor_classes_full = np.asarray(
            torch.load("output/clses-Epoch300.pt", weights_only=False)
        )
        vendor_boxes_full = np.asarray(
            torch.load("output/boxes-Epoch300.pt", weights_only=False)
        )
        vendor_classes = np.squeeze(vendor_classes_full[: len(names)], axis=-1)
        vendor_boxes = vendor_boxes_full[: len(names)] * np.asarray(
            (513, 750, 513, 750)
        )
        with TemporaryDirectory(dir=EVIDENCE / "s4-evaluation") as evaluation_dir:
            evaluation_root = Path(evaluation_dir)
            (evaluation_root / "Dataset").symlink_to(
                paths["root"], target_is_directory=True
            )
            (evaluation_root / "output").symlink_to(
                vendor_root / "output", target_is_directory=True
            )
            (evaluation_root / "test_order.pt").symlink_to(
                vendor_root / "test_order.pt"
            )
            os.chdir(evaluation_root)
            vendor_eval.main()
    os.chdir(ROOT)
    (EVIDENCE / "s4-evaluation").mkdir(parents=True, exist_ok=True)
    vendor_eval_output = output.getvalue()
    (EVIDENCE / "s4-evaluation" / "vendor-eval.txt").write_text(vendor_eval_output)
    package_classes: list[np.ndarray] = []
    package_boxes: list[np.ndarray] = []
    package_classes_raw: list[np.ndarray] = []
    package_boxes_raw: list[np.ndarray] = []
    stream_rows: list[dict[str, Any]] = []
    vendor_stream_iterator = iter(vendor_stream_loader)
    for index, batch in enumerate(package_loader):
        start = index * TEST_BATCH_SIZE
        valid = min(TEST_BATCH_SIZE, len(names) - start)
        package_inputs = batch["pixel_values"]
        if valid < TEST_BATCH_SIZE:
            package_inputs = torch.cat(
                (
                    package_inputs,
                    package_inputs[:1].expand(TEST_BATCH_SIZE - valid, -1, -1, -1),
                )
            )
        vendor_stream_batch = next(vendor_stream_iterator)
        _, stream_summary = _batch_stream_report(
            {"pixel_values": package_inputs},
            {"pixel_values": vendor_stream_batch},
        )
        stream_rows.append(stream_summary)
        with torch.no_grad():
            result = pipeline(
                pixel_values=package_inputs.to(device),
                batch_size=TEST_BATCH_SIZE,
                initial_layout=initial_layout,
                return_intermediates=True,
            )
        from laygen.modeling_outputs import LayoutGenerationOutput

        output = cast(LayoutGenerationOutput, result)
        if output.intermediates is None:
            raise RuntimeError("package evaluation did not return class probabilities")
        intermediates = cast(dict[str, object], output.intermediates)
        class_probs = cast(torch.Tensor, intermediates["class_probs"])
        boxes_tensor = cast(torch.Tensor, output.bbox)
        classes_full = class_probs.argmax(dim=-1).cpu().numpy()
        boxes_full = boxes_tensor.cpu().numpy()
        boxes_full = np.stack(
            (
                boxes_full[..., 0] - boxes_full[..., 2] / 2,
                boxes_full[..., 1] - boxes_full[..., 3] / 2,
                boxes_full[..., 0] + boxes_full[..., 2] / 2,
                boxes_full[..., 1] + boxes_full[..., 3] / 2,
            ),
            axis=-1,
        )
        package_classes_raw.append(classes_full)
        package_boxes_raw.append(boxes_full)
        package_classes.append(classes_full[:valid])
        package_boxes.append(boxes_full[:valid] * np.asarray((513, 750, 513, 750)))
    package_classes_array = np.concatenate(package_classes)
    package_boxes_array = np.concatenate(package_boxes)
    package_classes_raw_array = np.concatenate(package_classes_raw)
    package_boxes_raw_array = np.concatenate(package_boxes_raw)
    package_eval_root = EVIDENCE / "s4-evaluation" / "package-overlay"
    (package_eval_root / "output").mkdir(parents=True, exist_ok=True)
    package_boxes_for_eval = np.asarray(
        package_boxes_raw_array,
        dtype=vendor_boxes_full.dtype,
    )
    torch.save(
        package_classes_raw_array[..., np.newaxis],
        package_eval_root / "output/clses-Epoch300.pt",
    )
    torch.save(
        package_boxes_for_eval,
        package_eval_root / "output/boxes-Epoch300.pt",
    )
    with TemporaryDirectory(dir=EVIDENCE / "s4-evaluation") as package_metrics_dir:
        package_metrics_root = Path(package_metrics_dir)
        (package_metrics_root / "Dataset").symlink_to(
            paths["root"], target_is_directory=True
        )
        (package_metrics_root / "output").symlink_to(
            package_eval_root / "output", target_is_directory=True
        )
        (package_metrics_root / "test_order.pt").symlink_to(
            vendor_root / "test_order.pt"
        )
        old_cwd = Path.cwd()
        try:
            os.chdir(package_metrics_root)
            with redirect_stdout(io.StringIO()) as package_output:
                vendor_eval.main()
            package_eval_output = package_output.getvalue()
        finally:
            os.chdir(old_cwd)
    (EVIDENCE / "s4-evaluation" / "package-eval.txt").write_text(package_eval_output)
    checkpoint_hash = _sha256(checkpoint_path)
    vendor_prediction_file = _prediction_file(
        EVIDENCE / "s4-evaluation" / "vendor-predictions.npz",
        vendor_classes,
        vendor_boxes,
    )
    package_prediction_file = _prediction_file(
        EVIDENCE / "s4-evaluation" / "package-predictions.npz",
        package_classes_array,
        package_boxes_array,
    )
    vendor_metrics = _metrics_from_eval_output(vendor_eval_output)
    package_metrics = _metrics_from_eval_output(package_eval_output)
    vendor_summary = _metric_summary(vendor_classes, vendor_boxes)
    package_summary = _metric_summary(package_classes_array, package_boxes_array)
    vendor_weight_hash = _module_state_hash(vendor_generator)
    package_weight_hash = _module_state_hash(pipeline.model)
    same_weights = vendor_weight_hash == package_weight_hash
    prediction_equal = np.array_equal(
        vendor_classes, package_classes_array
    ) and np.array_equal(vendor_boxes, package_boxes_array)
    count_equal_nonzero = (
        vendor_summary["prediction_count"] == package_summary["prediction_count"]
        and vendor_summary["prediction_count"] > 0
    )
    test_stream_passed = all(row["passed"] for row in stream_rows)
    vendor_entry_point = (
        f"{Path(vendor_infer.__file__).resolve().relative_to(ROOT)}:"
        f"{vendor_infer.test.__qualname__};"
        f"{Path(vendor_eval.__file__).resolve().relative_to(ROOT)}:"
        f"{vendor_eval.main.__qualname__}"
    )
    package_prediction_entry_point = (
        f"{type(pipeline).__module__}.{type(pipeline).__qualname__}.__call__"
    )
    evaluation = {
        "vendor_evaluation_entry_point": vendor_entry_point,
        "package_prediction_entry_point": package_prediction_entry_point,
        "package_evaluation_entry_point": (
            f"{Path(vendor_eval.__file__).resolve().relative_to(ROOT)}:"
            f"{vendor_eval.main.__qualname__}"
        ),
        "same_weights": same_weights,
        "checkpoint_sha256": checkpoint_hash,
        "weight_state_sha256": {
            "vendor": vendor_weight_hash,
            "package": package_weight_hash,
        },
        "same_inputs": test_stream_passed,
        "input_bridge_manifest_sha256": _sha256(paths["manifest"]),
        "evaluator_settings": {
            "test_split": "TEST",
            "rows": len(names),
            "batch_size": TEST_BATCH_SIZE,
            "shuffle": False,
            "sampling_seed": SEED,
            "initial_layout_rng": {
                "source": "global RNG immediately after vendor model construction",
                "before": initial_layout_rng_before,
                "after": initial_layout_rng_after,
            },
            "padded_final_batch_rows": (TEST_BATCH_SIZE - len(names) % TEST_BATCH_SIZE)
            % TEST_BATCH_SIZE,
        },
        "coordinate_frame": "pixel xyxy on 513x750 canvas",
        "vendor_native_coordinate_frame": "normalized xyxy from infer.py converted by multiplying x coordinates by 513 and y coordinates by 750",
        "package_native_coordinate_frame": "normalized center xywh decoded by DSGANPipeline and converted to xyxy then multiplied by 513 and 750",
        "vendor_output_handling": {
            "raw_output_untouched_for_eval_main": True,
            "canonical_comparison_only": "in-memory squeeze of class singleton axis and pixel-scale conversion after eval.main input was preserved",
            "write_overlay": str(vendor_root.relative_to(ROOT)),
        },
        "prediction_files": {
            "vendor": vendor_prediction_file,
            "package": package_prediction_file,
        },
        "prediction_max_abs": float(np.max(np.abs(vendor_boxes - package_boxes_array))),
        "prediction_classes_equal": bool(
            np.array_equal(vendor_classes, package_classes_array)
        ),
        "prediction_boxes_bitwise_equal": bool(
            np.array_equal(vendor_boxes, package_boxes_array)
        ),
        "vendor": {
            **vendor_summary,
            "metrics": vendor_metrics,
            "evaluator_source_commit": _git("-C", str(VENDOR), "rev-parse", "HEAD"),
        },
        "package": {
            **package_summary,
            "metrics": package_metrics,
            "evaluator_source_commit": _git("rev-parse", "HEAD"),
        },
        "metrics_source": {
            "vendor": "captured vendor eval.py:main stdout",
            "package": "captured vendor eval.py:main stdout on package raw predictions",
        },
        "evaluation_stdout_sha256": {
            "vendor": _sha256(EVIDENCE / "s4-evaluation" / "vendor-eval.txt"),
            "package": _sha256(EVIDENCE / "s4-evaluation" / "package-eval.txt"),
        },
        "counts_equal_and_nonzero": count_equal_nonzero,
    }
    passed = prediction_equal and count_equal_nonzero and test_stream_passed
    return _write(
        "s4-evaluation",
        {
            **_metadata(),
            "stage": "S4",
            "result": "PASS" if passed else "FAIL",
            "bridge_manifest": ".cache/ds-gan/bridge/pku_posterlayout_manifest.json",
            "bridge_manifest_file_sha256": manifest_file_hash,
            "source_artifact_hashes": source_hashes,
            "source_artifact_hashes_verified": source_hashes_verified,
            "bridge_artifact_hashes": bridge_hashes,
            "bridge_artifact_hashes_verified": bridge_hashes_verified,
            "test_stream": {
                "split": "TEST",
                "rows": len(names),
                "loader": "vendor canvas/DataLoader semantics with package DSGANDataModule comparison",
                "comparison": {
                    "passed": test_stream_passed,
                    "batches": len(stream_rows),
                    "first_mismatch": next(
                        (
                            row["first_mismatch"]
                            for row in stream_rows
                            if not row["passed"]
                        ),
                        None,
                    ),
                },
            },
            "evaluation_path": evaluation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "s0-static",
            "s1-forward-loss",
            "s2-optimizer-step",
            "s2-vendor-self-repeat",
            "s3-lockstep",
            "s3-lockstep-synchronized",
            "s3-natural-repeat",
            "s3-self-repeat",
            "s3-production-wiring",
            "s4-bridge",
            "s4-evaluation",
        ),
    )
    parser.add_argument("repeat", nargs="?", type=int)
    parser.add_argument("json_path", nargs="?", type=Path)
    parser.add_argument("state_path", nargs="?", type=Path)
    parser.add_argument("--system", choices=("vendor", "package"))
    args = parser.parse_args()
    functions = {
        "s0-static": run_s0,
        "s1-forward-loss": run_s1,
        "s2-optimizer-step": run_s2,
        "s3-lockstep": run_s3,
        "s3-lockstep-synchronized": run_s3_synchronized,
        "s3-production-wiring": run_s3_production_wiring,
        "s4-bridge": run_s4_bridge,
        "s4-evaluation": run_s4,
    }
    if args.stage == "s2-vendor-self-repeat":
        if args.repeat is None or args.json_path is None or args.system is None:
            raise ValueError("S2 self repeat requires system, repeat, and JSON path")
        _run_s2_self_repeat(args.system, args.repeat, args.json_path)
        return
    if args.stage == "s3-natural-repeat":
        if args.repeat is None or args.json_path is None or args.state_path is None:
            raise ValueError(
                "S3 natural repeat requires repeat, JSON path, and state path"
            )
        run_s3_natural_repeat(args.repeat, args.json_path, args.state_path)
        return
    if args.stage == "s3-self-repeat":
        if args.repeat is None or args.json_path is None or args.state_path is None:
            raise ValueError(
                "S3 self repeat requires repeat, JSON path, and state path"
            )
        run_s3_self_repeat(args.repeat, args.json_path, args.state_path)
        return
    print(functions[args.stage]())


if __name__ == "__main__":
    main()
