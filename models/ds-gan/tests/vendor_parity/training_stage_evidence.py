"""Run DS-GAN training and evaluation agreement evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
from contextlib import contextmanager, redirect_stdout
import csv
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, cast  # noqa: TID251 - vendor adapter boundary is heterogeneous

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_wheel(name: str, override_path: str) -> dict[str, str]:
    metadata = distribution(name)
    wheel_path = Path(os.environ[override_path])
    if not wheel_path.exists():
        raise RuntimeError(f"wheel path for {name} is unavailable: {wheel_path}")
    wheel_name = wheel_path.name
    cuda_tag = next(
        (part for part in wheel_name.split("+")[1].split("-") if part.startswith("cu")),
        "unknown",
    )
    return {
        "name": name,
        "version": metadata.version,
        "wheel_name": wheel_name,
        "cuda_tag": cuda_tag,
        "sha256": _sha256(wheel_path),
    }


def _runtime() -> dict[str, Any]:
    freeze_hash = os.environ.get("DSGAN_AUDIT_FREEZE_SHA256")
    if not freeze_hash:
        raise RuntimeError("DSGAN_AUDIT_FREEZE_SHA256 is required for evidence")
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_wheel": _runtime_wheel("torch", "DSGAN_TORCH_WHEEL_PATH"),
        "torchvision_wheel": _runtime_wheel(
            "torchvision", "DSGAN_TORCHVISION_WHEEL_PATH"
        ),
        "audit_venv_env": "DSGAN_AUDIT_VENV",
        "venv_creation_command": 'python -m venv "$DSGAN_AUDIT_VENV"',
        "environment_basis": "lockfile environment for all CPU-only checks and tests; audited runtime only for CUDA evidence",
        "pip_freeze_sha256": freeze_hash,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
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
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            allow_tf32=False,
            cublas_workspace_config=":4096:8",
        )
    )
    torch.backends.cudnn.deterministic = True


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
            for label, box in zip(
                annotations["cls_elem"], annotations["box_elem"], strict=True
            ):
                vendor_label = SOURCE_TO_VENDOR_LABEL[int(label)]
                if vendor_label == 0:
                    continue
                writer.writerow(
                    [poster_path, len(annotations["cls_elem"]), vendor_label, repr(box)]
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
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["bridge_manifest_sha256"] = (
        __import__("hashlib").sha256(manifest_payload.encode()).hexdigest()
    )
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _vendor_loaders(seed: int) -> tuple[Any, Any]:
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
    old_cwd = Path.cwd()
    try:
        os.chdir(paths["root"])
        test_dataset = canvas(
            str(paths["test_images"]),
            str(paths["test_pfpn"]),
            str(paths["test_basnet"]),
            train=False,
        )
    finally:
        os.chdir(old_cwd)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        generator=generator,
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
    return module.train_dataloader(), module.test_dataloader()


def _batch_stream_report(
    package_batch: dict[str, torch.Tensor], vendor_batch: dict[str, torch.Tensor]
) -> tuple[BatchStreamReport, dict[str, Any]]:
    report = compare_batch_stream([vendor_batch], [package_batch], steps=1)
    return report, {
        "passed": report.passed,
        "checked_steps": report.checked_steps,
        "first_mismatch": report.first_mismatch,
    }


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
    batch_size = batch["pixel_values"].shape[0]
    if initial_layout is None:
        initial_layout = vendor_main.random_init(batch_size, MAX_ELEM).to(
            batch["pixel_values"].device
        )
    real = torch.ones(batch_size, device=batch["pixel_values"].device)
    fake = torch.full((batch_size,), -1.0, device=batch["pixel_values"].device)
    targets = _targets(batch)
    weight = min(1.0, max(0, epoch - 1) / 100)
    generator.train()
    discriminator.train()
    optimizer_g.zero_grad(set_to_none=True)
    classes, boxes = generator(batch["pixel_values"], initial_layout)
    generated_layout = torch.stack((classes, boxes), dim=2)
    generated_score = discriminator(batch["pixel_values"], generated_layout)
    loss_g_adv = nn.functional.hinge_embedding_loss(generated_score.reshape(-1), real)
    losses = criterion({"pred_logits": classes, "pred_boxes": boxes}, targets)
    loss_reconstruction = sum(losses.values())
    loss_g = weight * loss_g_adv + loss_reconstruction
    loss_g.backward()
    optimizer_g.step()
    optimizer_d.zero_grad(set_to_none=True)
    discriminator_fake = discriminator(batch["pixel_values"], generated_layout.detach())
    discriminator_real = discriminator(batch["pixel_values"], batch["layout"].clone())
    loss_d_fake = nn.functional.hinge_embedding_loss(
        discriminator_fake.reshape(-1), fake
    )
    loss_d_real = nn.functional.hinge_embedding_loss(
        discriminator_real.reshape(-1), real
    )
    loss_d = weight * (loss_d_real + loss_d_fake)
    loss_d.backward()
    optimizer_d.step()
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


def _fixed_batch(
    seed: int, device: torch.device
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, Any]
]:
    vendor_loader, _ = _vendor_loaders(seed)
    package_loader, _ = _package_loaders(seed)
    vendor_batch = _vendor_batch(next(iter(vendor_loader)), device)
    package_batch = _package_batch(next(iter(package_loader)), device)
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
        },
    )


def run_s0() -> Path:
    _set_determinism(SEED)
    (
        vendor_generator,
        vendor_discriminator,
        package_generator,
        package_discriminator,
        generator_config,
        discriminator_config,
    ) = _models(torch.device("cpu"))
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
    package_optimizers = _optimizers(package_generator, package_discriminator)
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
    package_optimizers = _optimizers(package_generator, package_discriminator)
    vendor_schedulers = _schedulers(vendor_optimizers)
    package_schedulers = _schedulers(package_optimizers)
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
        "cadence": "MultiStepLR advances only after each complete 78-batch epoch",
        "vendor": vendor_scheduler_values,
        "package": package_scheduler_values,
    }
    passed = all(
        comparison["passed"]
        for comparison in (
            trace_comparison,
            state_comparison,
            gradient_comparison,
            optimizer_comparison,
            scheduler_comparison,
        )
    )
    return _write(
        "s2-optimizer-step",
        {
            **_metadata(),
            "stage": "S2",
            "result": "PASS" if passed else "FAIL",
            "batch": batch_meta,
            "optimizer_order": "generator_then_discriminator",
            "trace": trace_comparison,
            "gradients": gradient_comparison,
            "post_step_parameters": state_comparison,
            "optimizer_state": optimizer_comparison,
            "scheduler": scheduler_comparison,
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

    vendor_loader, _ = _vendor_loaders(SEED)
    package_loader, _ = _package_loaders(SEED)
    vendor_iterator = iter(vendor_loader)
    package_iterator = iter(package_loader)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    package_optimizers = _optimizers(package_generator, package_discriminator)
    vendor_schedulers = _schedulers(vendor_optimizers)
    package_schedulers = _schedulers(package_optimizers)
    package_numpy_rng = np.random.RandomState(SEED)
    package_torch_generator = torch.Generator(device="cpu")
    package_torch_generator.manual_seed(SEED)
    trace_rows: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    initial_rng = _rng_digest(capture_rng_state())
    for step in range(1, LOCKSTEP_STEPS + 1):
        epoch = (step - 1) // TRAIN_BATCHES_PER_EPOCH + 1
        if step > 1 and (step - 1) % TRAIN_BATCHES_PER_EPOCH == 0:
            vendor_iterator = iter(vendor_loader)
            package_iterator = iter(package_loader)
        vendor_batch = _vendor_batch(next(vendor_iterator), device)
        package_batch = _package_batch(next(package_iterator), device)
        batch_report, _ = _batch_stream_report(package_batch, vendor_batch)
        if not batch_report.passed:
            raise RuntimeError(
                f"natural loader mismatch at step {step}: {batch_report.first_mismatch}"
            )
        vendor_trace = _vendor_step(
            vendor_generator,
            vendor_discriminator,
            vendor_batch,
            *vendor_optimizers,
            _vendor_criterion(device),
            epoch,
        )
        package_trace = _package_step(
            package_module,
            package_batch,
            *package_optimizers,
            epoch,
            numpy_rng=package_numpy_rng,
            torch_generator=package_torch_generator,
        )
        trace_comparison = _trace_compare(vendor_trace, package_trace)
        parameter_comparison = _state_compare(
            {
                **_named_parameters(vendor_generator, "generator"),
                **_named_parameters(vendor_discriminator, "discriminator"),
            },
            {
                **_named_parameters(package_generator, "generator"),
                **_named_parameters(package_discriminator, "discriminator"),
            },
        )
        gradient_comparison = _state_compare(
            {
                **_named_gradients(vendor_generator, "generator"),
                **_named_gradients(vendor_discriminator, "discriminator"),
            },
            {
                **_named_gradients(package_generator, "generator"),
                **_named_gradients(package_discriminator, "discriminator"),
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
                    package_optimizers[0], package_generator, "generator"
                ),
                **_named_optimizer_state(
                    package_optimizers[1], package_discriminator, "discriminator"
                ),
            },
        )
        scheduler_event = step % TRAIN_BATCHES_PER_EPOCH == 0
        if scheduler_event:
            for scheduler in (*vendor_schedulers, *package_schedulers):
                scheduler.step()
        scheduler_values = {
            "vendor_last_epoch": [
                scheduler.last_epoch for scheduler in vendor_schedulers
            ],
            "package_last_epoch": [
                scheduler.last_epoch for scheduler in package_schedulers
            ],
            "vendor_learning_rates": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in vendor_optimizers
            ],
            "package_learning_rates": [
                [group["lr"] for group in optimizer.param_groups]
                for optimizer in package_optimizers
            ],
            "stepped": scheduler_event,
        }
        scheduler_equal = (
            scheduler_values["vendor_last_epoch"]
            == scheduler_values["package_last_epoch"]
            and scheduler_values["vendor_learning_rates"]
            == scheduler_values["package_learning_rates"]
        )
        row = {
            "step": step,
            "epoch": epoch,
            "batch_stream": {
                "passed": batch_report.passed,
                "checked_steps": batch_report.checked_steps,
            },
            "trace": trace_comparison,
            "gradients": gradient_comparison,
            "parameters": parameter_comparison,
            "optimizer_state": optimizer_comparison,
            "scheduler": {**scheduler_values, "passed": scheduler_equal},
        }
        trace_rows.append(row)
        if first_divergence is None and not all(
            comparison["passed"]
            for comparison in (
                trace_comparison,
                gradient_comparison,
                parameter_comparison,
                optimizer_comparison,
                {"passed": scheduler_equal},
            )
        ):
            first_divergence = row
    trace_path = _write_jsonl(
        "s3-lockstep", f"natural-repeat-{repeat}.jsonl", trace_rows
    )
    max_abs = (
        max(
            comparison["first_difference"]["max_abs"]
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
        "initial_rng_digest": initial_rng,
        "randomness": "each system starts from seed 0 and consumes its own setup_seed/random_init code path; no per-step RNG restore or injected shared layout",
    }


def run_s3() -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repeats = [_run_natural(repeat, device) for repeat in (1, 2)]
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
            "repeat_run_envelope": {
                "max_abs_difference": max(
                    item["max_abs_difference"] for item in repeats
                ),
                "first_divergence_steps": [
                    item["first_divergence"]["step"]
                    if item["first_divergence"] is not None
                    else None
                    for item in repeats
                ],
                "repeat_count": len(repeats),
            },
            "randomness": "each system starts from seed 0 and consumes its own setup_seed/random_init code path; no per-step RNG restore or injected shared layout",
            "scheduler_cadence": "MultiStepLR steps after each complete 78-batch training epoch; generator milestones every 50 epochs and discriminator every 25 epochs",
            "bound": {
                "declared_before_run": False,
                "status": "no approximate bound claimed",
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
) -> dict[str, Any]:
    _copy_module_state(package_module.generator, vendor_generator)
    _copy_module_state(package_module.discriminator, vendor_discriminator)
    _copy_optimizer_state(package_optimizers[0], vendor_optimizers[0])
    _copy_optimizer_state(package_optimizers[1], vendor_optimizers[1])
    _copy_scheduler_state(package_schedulers[0], vendor_schedulers[0])
    _copy_scheduler_state(package_schedulers[1], vendor_schedulers[1])
    vendor_rng = capture_rng_state()
    vendor_trace = _vendor_step(
        vendor_generator,
        vendor_discriminator,
        vendor_batch,
        *vendor_optimizers,
        _vendor_criterion(device),
        epoch,
    )
    vendor_after_rng = capture_rng_state()
    restore_rng_state(vendor_rng)
    package_trace = _package_step(
        package_module, package_batch, *package_optimizers, epoch
    )
    package_after_rng = capture_rng_state()
    return {
        "step": step,
        "epoch": epoch,
        "trace": _trace_compare(vendor_trace, package_trace),
        "gradients": _state_compare(
            {
                **_named_gradients(vendor_generator, "generator"),
                **_named_gradients(vendor_discriminator, "discriminator"),
            },
            {
                **_named_gradients(package_module.generator, "generator"),
                **_named_gradients(package_module.discriminator, "discriminator"),
            },
        ),
        "parameters": _state_compare(
            {
                **_named_parameters(vendor_generator, "generator"),
                **_named_parameters(vendor_discriminator, "discriminator"),
            },
            {
                **_named_parameters(package_module.generator, "generator"),
                **_named_parameters(package_module.discriminator, "discriminator"),
            },
        ),
        "optimizer_state": _state_compare(
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
        ),
        "rng_equal_after_operation": _rng_digest(vendor_after_rng)
        == _rng_digest(package_after_rng),
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


def run_s3_synchronized() -> Path:
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
    from ds_gan.training.lightning_module import DSGANTrainingModule

    package_module = DSGANTrainingModule(
        config=generator_config,
        discriminator_config=discriminator_config,
        generator=package_generator,
        discriminator=package_discriminator,
    ).to(device)
    vendor_optimizers = _optimizers(vendor_generator, vendor_discriminator)
    package_optimizers = _optimizers(package_generator, package_discriminator)
    vendor_schedulers = _schedulers(vendor_optimizers)
    package_schedulers = _schedulers(package_optimizers)
    vendor_loader, _ = _vendor_loaders(SEED)
    package_loader, _ = _package_loaders(SEED)
    vendor_iterator = iter(vendor_loader)
    package_iterator = iter(package_loader)
    rows: list[dict[str, Any]] = []
    for step in range(1, LOCKSTEP_STEPS + 1):
        epoch = (step - 1) // TRAIN_BATCHES_PER_EPOCH + 1
        if step > 1 and (step - 1) % TRAIN_BATCHES_PER_EPOCH == 0:
            vendor_iterator = iter(vendor_loader)
            package_iterator = iter(package_loader)
        vendor_batch = _vendor_batch(next(vendor_iterator), device)
        package_batch = _package_batch(next(package_iterator), device)
        batch_report, _ = _batch_stream_report(package_batch, vendor_batch)
        if not batch_report.passed:
            raise RuntimeError(f"synchronized loader mismatch at step {step}")
        rows.append(
            _synchronized_step(
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
        )
        if step % TRAIN_BATCHES_PER_EPOCH == 0:
            for scheduler in (*vendor_schedulers, *package_schedulers):
                scheduler.step()
    trace_path = _write_jsonl("s3-lockstep-synchronized", "trace.jsonl", rows)
    passed = all(
        row["trace"]["passed"]
        and row["gradients"]["passed"]
        and row["parameters"]["passed"]
        and row["optimizer_state"]["passed"]
        and row["rng_equal_after_operation"]
        for row in rows
    )
    first_divergence = next(
        (
            row
            for row in rows
            if not (
                row["trace"]["passed"]
                and row["gradients"]["passed"]
                and row["parameters"]["passed"]
                and row["optimizer_state"]["passed"]
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
            "natural_record": ".cache/ds-gan/stage-evidence/s3-lockstep/run.json",
            "synchronized_criterion": "parameters, optimizer state, and RNG state synchronized at every optimizer boundary; no shared input tensor injected",
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


def _vendor_metrics(
    vendor_eval: Any, names: list[str], classes: np.ndarray, boxes: np.ndarray
) -> dict[str, float]:
    return {
        "metrics_val": float(vendor_eval.metrics_val((513, 750), classes, boxes)),
        "metrics_ove": float(vendor_eval.metrics_ove(classes, boxes)),
        "metrics_ali": float(vendor_eval.metrics_ali(classes, boxes)),
        "metrics_und_l": float(vendor_eval.metrics_und_l(classes, boxes)),
        "metrics_und_s": float(vendor_eval.metrics_und_s(classes, boxes)),
        "metrics_uti": float(vendor_eval.metrics_uti(names, classes, boxes)),
        "metrics_occ": float(vendor_eval.metrics_occ(names, classes, boxes)),
        "metrics_rea": float(vendor_eval.metrics_rea(names, classes, boxes)),
    }


def run_s4_bridge() -> Path:
    manifest = _materialize_bridge()
    return _write(
        "s4-bridge",
        {
            **_metadata(),
            "stage": "S4",
            "result": "PASS" if not manifest["missing_source_fields"] else "BLOCKED",
            "manifest": manifest,
        },
    )


def run_s4() -> Path:
    _set_determinism(SEED)
    paths = _bridge_paths()
    manifest = json.loads(paths["manifest"].read_text())
    recorded_manifest_hash = manifest.pop("bridge_manifest_sha256", None)
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_hash_verified = (
        recorded_manifest_hash
        == __import__("hashlib").sha256(manifest_payload.encode()).hexdigest()
    )
    if not manifest_hash_verified:
        raise RuntimeError("bridge manifest hash verification failed")
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
    vendor_root = paths["root"]
    (vendor_root / "output").mkdir(parents=True, exist_ok=True)
    _, vendor_eval_loader = _vendor_loaders(SEED)
    names = list(torch.load(vendor_root / "test_order.pt", weights_only=False))
    _, package_loader = _package_loaders(SEED)
    _, vendor_stream_loader = _vendor_loaders(SEED)
    initial_layout = _vendor_main().random_init(TEST_BATCH_SIZE, MAX_ELEM).to(device)
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
        torch.save(torch.as_tensor(vendor_classes), "output/clses-Epoch300.pt")
        torch.save(
            torch.as_tensor(vendor_boxes / np.asarray((513, 750, 513, 750))),
            "output/boxes-Epoch300.pt",
        )
        vendor_eval.main()
    os.chdir(ROOT)
    (EVIDENCE / "s4-evaluation").mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "s4-evaluation" / "vendor-eval.txt").write_text(output.getvalue())
    package_classes: list[np.ndarray] = []
    package_boxes: list[np.ndarray] = []
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
        classes = class_probs.argmax(dim=-1).cpu().numpy()[:valid]
        boxes = boxes_tensor.cpu().numpy()[:valid]
        boxes = np.stack(
            (
                boxes[..., 0] - boxes[..., 2] / 2,
                boxes[..., 1] - boxes[..., 3] / 2,
                boxes[..., 0] + boxes[..., 2] / 2,
                boxes[..., 1] + boxes[..., 3] / 2,
            ),
            axis=-1,
        )
        package_classes.append(classes)
        package_boxes.append(boxes * np.asarray((513, 750, 513, 750)))
    package_classes_array = np.concatenate(package_classes)
    package_boxes_array = np.concatenate(package_boxes)
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
    vendor_metrics = _vendor_metrics(vendor_eval, names, vendor_classes, vendor_boxes)
    package_metrics = _vendor_metrics(
        vendor_eval, names, package_classes_array, package_boxes_array
    )
    vendor_summary = _metric_summary(vendor_classes, vendor_boxes)
    package_summary = _metric_summary(package_classes_array, package_boxes_array)
    vendor_weight_hash = _module_state_hash(vendor_generator)
    package_weight_hash = _module_state_hash(pipeline.model)
    same_weights = vendor_weight_hash == package_weight_hash
    prediction_equal = np.array_equal(
        vendor_classes, package_classes_array
    ) and np.array_equal(vendor_boxes, package_boxes_array)
    metrics_equal = all(
        (left == right) or (np.isnan(left) and np.isnan(right))
        for left, right in zip(
            vendor_metrics.values(), package_metrics.values(), strict=True
        )
    )
    count_equal_nonzero = (
        vendor_summary["prediction_count"] == package_summary["prediction_count"]
        and vendor_summary["prediction_count"] > 0
    )
    test_stream_passed = all(row["passed"] for row in stream_rows)
    evaluation = {
        "vendor_evaluation_entry_point": "vendor/posterlayout-cvpr2023/infer.py:test then eval.py:main",
        "package_evaluation_entry_point": "ds_gan.DSGANPipeline.__call__",
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
            "initial_layout_seed": SEED,
            "padded_final_batch_rows": (TEST_BATCH_SIZE - len(names) % TEST_BATCH_SIZE)
            % TEST_BATCH_SIZE,
        },
        "coordinate_frame": "pixel xyxy on 513x750 canvas",
        "vendor_native_coordinate_frame": "normalized xyxy from infer.py converted by multiplying x coordinates by 513 and y coordinates by 750",
        "package_native_coordinate_frame": "normalized center xywh decoded by DSGANPipeline and converted to xyxy then multiplied by 513 and 750",
        "prediction_files": {
            "vendor": vendor_prediction_file,
            "package": package_prediction_file,
        },
        "prediction_max_abs": float(np.max(np.abs(vendor_boxes - package_boxes))),
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
        "metrics_identical": metrics_equal,
        "counts_equal_and_nonzero": count_equal_nonzero,
        "evaluation_stdout_sha256": _sha256(
            EVIDENCE / "s4-evaluation" / "vendor-eval.txt"
        ),
    }
    passed = (
        prediction_equal
        and metrics_equal
        and count_equal_nonzero
        and test_stream_passed
    )
    return _write(
        "s4-evaluation",
        {
            **_metadata(),
            "stage": "S4",
            "result": "PASS" if passed else "FAIL",
            "bridge_manifest": ".cache/ds-gan/bridge/pku_posterlayout_manifest.json",
            "bridge_manifest_hash_verified": manifest_hash_verified,
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
            "s3-lockstep",
            "s3-lockstep-synchronized",
            "s4-bridge",
            "s4-evaluation",
        ),
    )
    args = parser.parse_args()
    functions = {
        "s0-static": run_s0,
        "s1-forward-loss": run_s1,
        "s2-optimizer-step": run_s2,
        "s3-lockstep": run_s3,
        "s3-lockstep-synchronized": run_s3_synchronized,
        "s4-bridge": run_s4_bridge,
        "s4-evaluation": run_s4,
    }
    print(functions[args.stage]())


if __name__ == "__main__":
    main()
