"""Generate LayoutFlow S3/S4 training reproduction evidence."""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final, cast  # noqa: TID251 - vendor batches and JSON are dynamic.

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT: Final[Path] = (
    REPO_ROOT / ".cache" / "layout-flow" / "stage-evidence"
)
DEFAULT_VENDOR_ROOT: Final[Path] = REPO_ROOT / "vendor" / "layout-flow"


def _write_split(path: Path, *, sample_count: int) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", track_order=True) as h5:
        for index in range(sample_count):
            length = 1 + (index % 5)
            group = h5.create_group(f"sample-{index:03d}")
            left = np.linspace(0.01 * (index + 1), 0.03 * (index + 1), length)
            top = np.linspace(0.02 * (index + 1), 0.04 * (index + 1), length)
            width = np.full(length, 0.08 + 0.005 * (index % 3))
            height = np.full(length, 0.10 + 0.004 * (index % 4))
            group.create_dataset(
                "bbox",
                data=np.stack([left, top, width, height], axis=1).astype("float32"),
            )
            group.create_dataset(
                "categories",
                data=((np.arange(length) + index) % 5 + 1).astype("int64"),
            )
            group.create_dataset("length", data=np.array(length, dtype="int64"))


def write_fixture(output_root: Path) -> Path:
    data_root = output_root / "data" / "publaynet"
    for name, count in {
        "publaynet_train.h5": 12,
        "publaynet_val.h5": 6,
        "publaynet_test.h5": 6,
    }.items():
        _write_split(data_root / name, sample_count=count)
    return data_root


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().double().sum().item())


def _batch_summary(batch: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "bbox_shape": list(cast(torch.Tensor, batch["bbox"]).shape),
        "bbox_checksum": round(_tensor_checksum(cast(torch.Tensor, batch["bbox"])), 8),
        "length": cast(torch.Tensor, batch["length"]).cpu().tolist(),
        "mask_true": int(cast(torch.Tensor, batch["mask"]).sum().item()),
        "type": cast(torch.Tensor, batch["type"]).cpu().tolist(),
        "type_checksum": int(cast(torch.Tensor, batch["type"]).sum().item()),
    }
    if "id" in batch:
        summary["id"] = list(batch["id"])
    return summary


def _assert_batch_equal(
    package_batch: dict[str, Any],
    vendor_batch: dict[str, Any],
    *,
    compare_ids: bool = False,
) -> None:
    del compare_ids
    for key in ("bbox", "type", "mask", "length"):
        package_value = cast(torch.Tensor, package_batch[key])
        vendor_value = cast(torch.Tensor, vendor_batch[key])
        if not torch.equal(package_value.cpu(), vendor_value.cpu()):
            raise AssertionError(f"Loader stream mismatch for {key}")


def _vendor_loader_symbols(
    vendor_root: Path,
) -> tuple[type[object], Callable[..., dict[str, Any]]]:
    marker = vendor_root / "src" / "datamodule" / "PubLayNet.py"
    if not marker.exists():
        raise FileNotFoundError(
            f"{marker} is missing; run `git submodule update --init vendor/layout-flow`"
        )
    sys.path.insert(0, str(vendor_root))
    from src.datamodule.PubLayNet import PubLayNet, collate_fn

    return PubLayNet, cast(Callable[..., dict[str, Any]], collate_fn)


def _stream_batches(
    loader: Iterable[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == limit:
            break
    return batches


def run_s4_loader_stream(output_root: Path, vendor_root: Path) -> Path:
    from layout_flow.training.dataset import (
        LayoutFlowH5Dataset,
        collate_layout_flow_batch,
    )

    data_root = write_fixture(output_root)
    vendor_dataset_class, vendor_collate_fn = _vendor_loader_symbols(vendor_root)
    package_train = LayoutFlowH5Dataset(
        data_path=data_root, dataset_name="publaynet", split="train"
    )
    vendor_train = vendor_dataset_class(split="train", data_path=str(data_root))
    package_val = LayoutFlowH5Dataset(
        data_path=data_root, dataset_name="publaynet", split="validation"
    )
    vendor_val = vendor_dataset_class(split="validation", data_path=str(data_root))

    seed = 314159
    package_train_loader = DataLoader(
        package_train,
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=lambda batch: collate_layout_flow_batch(
            batch, max_length=5, box_format="xywh"
        ),
    )
    vendor_train_loader = DataLoader(
        vendor_train,
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=functools.partial(vendor_collate_fn, max_len=5, format="xywh"),
    )
    package_val_loader = DataLoader(
        package_val,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_layout_flow_batch(
            batch, max_length=5, box_format="xywh"
        ),
    )
    vendor_val_loader = DataLoader(
        vendor_val,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=functools.partial(vendor_collate_fn, max_len=5, format="xywh"),
    )

    train_package_batches = _stream_batches(package_train_loader, limit=4)
    train_vendor_batches = _stream_batches(vendor_train_loader, limit=4)
    val_package_batches = _stream_batches(package_val_loader, limit=2)
    val_vendor_batches = _stream_batches(vendor_val_loader, limit=2)
    for package_batch, vendor_batch in zip(
        train_package_batches, train_vendor_batches, strict=True
    ):
        _assert_batch_equal(package_batch, vendor_batch)
    for package_batch, vendor_batch in zip(
        val_package_batches, val_vendor_batches, strict=True
    ):
        _assert_batch_equal(package_batch, vendor_batch)

    artifact = output_root / "s4-loader-stream" / "summary.json"
    _write_json(
        artifact,
        {
            "stage": "S4",
            "dataset": "publaynet",
            "fixture": data_root.relative_to(REPO_ROOT),
            "vendor_root": vendor_root.relative_to(REPO_ROOT),
            "seed": seed,
            "max_length": 5,
            "batch_size": 3,
            "train_batches_checked": len(train_package_batches),
            "validation_batches_checked": len(val_package_batches),
            "train_package": [_batch_summary(batch) for batch in train_package_batches],
            "train_vendor": [_batch_summary(batch) for batch in train_vendor_batches],
            "validation_package": [
                _batch_summary(batch) for batch in val_package_batches
            ],
            "validation_vendor": [
                _batch_summary(batch) for batch in val_vendor_batches
            ],
            "result": (
                "PASS: package and vendor loader streams match for sample order, "
                "xywh transform, mask, padding, and class ids."
            ),
        },
    )
    return artifact


def run_s3_short_run(output_root: Path) -> Path:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from layout_flow import LayoutFlowConfig
    from layout_flow.training.datamodule import LayoutFlowDataModule
    from layout_flow.training.lightning_module import LayoutFlowTrainingModule
    from layout_flow.training.seed import apply_layout_flow_seed_mode

    data_root = write_fixture(output_root)
    run_root = output_root / "s3-short-run"
    shutil.rmtree(run_root, ignore_errors=True)
    apply_layout_flow_seed_mode("deterministic", seed=42975)
    pl.seed_everything(42975, workers=True)
    module = LayoutFlowTrainingModule(
        config=LayoutFlowConfig(
            dataset_name="publaynet",
            max_length=5,
            latent_dim=8,
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            num_layers=1,
            dropout=0.0,
        ),
        scheduler=None,
        seed_mode="deterministic",
        fid_calc_every_n=0,
    )
    scheduler_probe = LayoutFlowTrainingModule(
        config=module.layout_flow_config,
        scheduler="reduce_on_plateau",
        seed_mode="deterministic",
        fid_calc_every_n=20,
    ).configure_optimizers()
    datamodule = LayoutFlowDataModule(
        data_path=data_root,
        dataset_name="publaynet",
        batch_size=2,
        max_length=5,
        num_workers=0,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_root / "checkpoints",
        filename="step-{step:04d}",
        every_n_train_steps=2,
        save_top_k=-1,
        save_last=True,
    )
    logger = CSVLogger(save_dir=run_root, name="csv")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        deterministic=True,
        benchmark=False,
        max_steps=4,
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=run_root,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    checkpoint_files = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (run_root / "checkpoints").glob("*.ckpt")
    )
    metrics_path = Path(logger.log_dir) / "metrics.csv"
    if trainer.global_step != 4:
        raise AssertionError(f"Expected 4 global steps, got {trainer.global_step}")
    if not metrics_path.exists():
        raise AssertionError(f"Missing CSV metrics at {metrics_path}")
    if not checkpoint_files:
        raise AssertionError("No checkpoints were written")

    scheduler_metadata: dict[str, Any] = {}
    if isinstance(scheduler_probe, dict):
        scheduler_config = cast(dict[str, Any], scheduler_probe["lr_scheduler"])
        scheduler_metadata = {
            "class": type(scheduler_config["scheduler"]).__name__,
            "monitor": scheduler_config["monitor"],
            "frequency": scheduler_config["frequency"],
        }
    artifact = run_root / "summary.json"
    _write_json(
        artifact,
        {
            "stage": "S3",
            "dataset": "publaynet",
            "fixture": data_root.relative_to(REPO_ROOT),
            "seed": 42975,
            "accelerator": accelerator,
            "max_steps": trainer.max_steps,
            "global_step": trainer.global_step,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "gradient_clip_val": trainer.gradient_clip_val,
            "scheduler_probe": scheduler_metadata,
            "runtime_scheduler": (
                "disabled to match current full-run commands until package-local FID "
                "validation is implemented"
            ),
            "csv_metrics": metrics_path.relative_to(REPO_ROOT),
            "checkpoints": checkpoint_files,
            "latest_trace_keys": sorted(module.latest_step_trace),
            "latest_train_loss": float(
                module.latest_step_trace["train_loss"].cpu().item()
            ),
            "result": (
                "PASS: deterministic 4-step training run exercised repeated loader, "
                "accumulation, clipping, logging, and checkpoint wiring."
            ),
        },
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("write-fixture", "s3-short-run", "s4-loader-stream", "all"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    vendor_root = args.vendor_root.resolve()
    artifacts: list[Path] = []
    if args.stage in {"write-fixture", "all"}:
        artifacts.append(write_fixture(output_root))
    if args.stage in {"s3-short-run", "all"}:
        artifacts.append(run_s3_short_run(output_root))
    if args.stage in {"s4-loader-stream", "all"}:
        artifacts.append(run_s4_loader_stream(output_root, vendor_root))
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    main()
