# ruff: noqa: E402 - audited-runtime gate must run before optional imports.
from __future__ import annotations

import ast
import hashlib
import json
import gc
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251 - vendor APIs are dynamic.
    Final,
    Protocol,
    cast,
)  # noqa: TID251 - vendor APIs are dynamic.
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


def _require_audited_runtime_when_gated() -> None:
    if os.environ.get("PARITY_REQUIRE") != "1":
        return

    required = (
        "LAYOUT_CORRECTOR_AUDIT_VENV",
        "LAYOUT_CORRECTOR_TORCH_WHEEL",
        "LAYOUT_CORRECTOR_TORCHVISION_WHEEL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "PARITY_REQUIRE=1 requires audited runtime metadata: " + ", ".join(missing)
        )

    runtime = Path(os.environ["LAYOUT_CORRECTOR_AUDIT_VENV"])
    expected_python = (runtime / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(
            "PARITY_REQUIRE=1 must run under LAYOUT_CORRECTOR_AUDIT_VENV: "
            f"{sys.executable} != {expected_python}"
        )


pytest.importorskip("lightning", _require_audited_runtime_when_gated())
pytest.importorskip("traingen_parity")
pytest.importorskip("omegaconf")
pytest.importorskip("torch_geometric")

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from traingen_parity.determinism import (
    DeterminismConfig,
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
)
from traingen_parity.compare import (
    compare_batch_stream,
    compare_optimizer_step,
    compare_tensors,
)
from traingen_parity.trace import tensor_sha256

from layout_corrector import LayoutCorrectorConfig
from layout_corrector.training import (
    FrozenLayoutDMReference,
    LayoutCorrectorDataModule,
    LayoutCorrectorTrainingModule,
)
from layout_corrector.training.parity import (
    _optimizer_state_digest,
    _scheduler_state_digest,
    build_layout_corrector_step_trace,
    compare_layout_corrector_step,
)

if TYPE_CHECKING:
    from layout_corrector.training.parity import DigestValue
from layout_corrector.training.config import (
    LayoutCorrectorTrainingDatasetName,
    LayoutCorrectorTrainingSplit,
)
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.training.config import LayoutDMTrainingDatasetName

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT: Final = Path(__file__).resolve().parents[4]
DATASETS: Final[tuple[LayoutCorrectorTrainingDatasetName, ...]] = (
    "rico25",
    "publaynet",
)
CORRECTOR_HIDDEN_SIZE: Final = 432
CORRECTOR_INTERMEDIATE_SIZE: Final = 1728
EVALUATION_BATCH_SIZE: Final = 512

EVALUATION_CONDITIONS: Final[tuple[str, ...]] = ("unconditional", "c", "cwh")
EVALUATION_CORRECTOR_T_LIST: Final[tuple[int, ...]] = (10, 20, 30)
EVALUATION_SEED: Final[int] = 0
NATURAL_STREAM_SEED: Final[int] = 42975
S3_STEP_BOUND: Final[int] = 300
VALIDATION_LOSS_TOLERANCE: Final[float] = 2.0e-8
CHECKPOINT_SHA256: Final = {
    "rico25": "7759bdf9e05cccef7a6a7e4260adc50f8c1ef6e6faa10351b79fb63f6b51c853",
    "publaynet": "9f7aee8ca600cc7cc96182affc85f96ebafc2b41a9ae72b05dfacfd64e89791d",
}


class VendorCorrectorModel(Protocol):
    """Dynamic model surface exposed by the original corrector."""

    module: torch.nn.Module

    def __call__(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Run the original corrector backbone."""

    def named_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        """Return the original parameter registration order."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the original corrector state."""


class VendorCorrector(Protocol):
    """Reference-adapter surface used by the gated parity harness."""

    model: VendorCorrectorModel

    def __call__(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run the original corrector forward and loss path."""

    def eval(self) -> VendorCorrector:
        """Switch the original corrector to evaluation mode."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return original corrector parameters."""

    def to(self, device: torch.device) -> VendorCorrector:
        """Move the original corrector to a device."""

    def preprocess(
        self,
        batch: dict[str, torch.Tensor],
        diffusion: torch.nn.Module,
        _sampler_config: DictConfig,
    ) -> dict[str, torch.Tensor]:
        """Prepare a batch using the original corruption path."""

    def optim_groups(self, *, weight_decay: float) -> list[dict[str, Any]]:
        """Return original optimizer parameter groups."""


def _layout_dm_cache() -> Path:
    value = os.environ.get("LAYOUT_DM_CACHE")
    if value is None:
        skip_or_fail_vendor_parity(
            "LayoutDM training cache is local-only",
            missing_paths=[ROOT / ".cache" / "layout-dm"],
            regeneration_hint="set LAYOUT_DM_CACHE to the LayoutDM download cache",
        )

    assert value is not None
    return Path(value)


def _evidence_root() -> Path:
    path = ROOT / ".cache" / "layout-corrector" / "stage-evidence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(stage: str, dataset: str, payload: dict[str, Any]) -> Path:
    path = _evidence_root() / stage / dataset / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_jsonl(stage: str, dataset: str, rows: list[dict[str, Any]]) -> Path:
    path = _evidence_root() / stage / dataset / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    return tensor_sha256(value)


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()
    torch_cpu = getattr(value, "torch_cpu", None)
    if isinstance(torch_cpu, torch.Tensor):
        digest.update(
            torch_cpu.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
        for item in getattr(value, "torch_cuda", ()):
            digest.update(
                item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        digest.update(repr(getattr(value, "python", None)).encode())
        numpy_state = getattr(value, "numpy", None)
        if numpy_state is not None:
            digest.update(str(numpy_state[0]).encode())
            digest.update(numpy_state[1].tobytes())
        return digest.hexdigest()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            item = value[key]
            if isinstance(item, torch.Tensor):
                digest.update(
                    item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
            else:
                digest.update(repr(item).encode())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def _canonical_device(device: str | torch.device) -> str:
    value = torch.device(device)
    if value.type == "cuda" and value.index is None:
        value = torch.device("cuda", torch.cuda.current_device())
    return str(value)


def _source_commit(path: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(f"refusing evidence from dirty source tree: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_record() -> dict[str, Any]:
    """Return the audited runtime identity used to produce a stage record."""
    runtime = os.environ.get("LAYOUT_CORRECTOR_AUDIT_VENV")
    if runtime is None:
        if os.environ.get("PARITY_REQUIRE") == "1":
            raise RuntimeError(
                "PARITY_REQUIRE=1 requires LAYOUT_CORRECTOR_AUDIT_VENV for evidence"
            )
        return {"environment": "lockfile"}

    python = Path(runtime) / "bin" / "python"
    if Path(sys.executable).resolve() != python.resolve():
        raise RuntimeError(
            "evidence must run from LAYOUT_CORRECTOR_AUDIT_VENV: "
            f"{sys.executable} != {python}"
        )

    result = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = result.stdout
    records: dict[str, Any] = {
        "environment": "audited-runtime",
        "python": subprocess.run(
            [str(python), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "freeze_includes_file_urls": "file://" in freeze,
    }
    for package in ("torch", "torchvision"):
        package_result = subprocess.run(
            [str(python), "-c", f"import {package}; print({package}.__version__)"],
            check=True,
            capture_output=True,
            text=True,
        )
        version = package_result.stdout.strip()
        records[package] = {"version": version}

    wheel_paths = {
        "torch": os.environ.get("LAYOUT_CORRECTOR_TORCH_WHEEL"),
        "torchvision": os.environ.get("LAYOUT_CORRECTOR_TORCHVISION_WHEEL"),
    }
    for package, wheel_path in wheel_paths.items():
        if wheel_path is None:
            raise RuntimeError(f"{package} wheel path is required for evidence")
        path = Path(wheel_path)
        if not path.is_file():
            raise RuntimeError(f"{package} wheel is missing: {path}")
        records[package]["wheel"] = path.name
        records[package]["wheel_sha256"] = _sha256(path)

    return records


def _vendor_imports() -> tuple[type[torch.nn.Module], type[torch.nn.Module], type[Any]]:
    vendor_src = ROOT / "vendor" / "layout-corrector" / "src" / "trainer"
    if not vendor_src.exists():
        skip_or_fail_vendor_parity(
            "Layout-Corrector vendor checkout is missing",
            missing_paths=[vendor_src],
            regeneration_hint="run `git submodule update --init vendor/layout-corrector`",
        )
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))
    from trainer.corrector.layout_corrector import LayoutCorrector
    from trainer.helpers import bbox_tokenizer
    from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from trainer.models.layoutdm import LayoutDM

    bbox_tokenizer.KMEANS_WEIGHT_ROOT = str(_layout_dm_cache() / "clustering_weights")
    return (
        cast(type[torch.nn.Module], LayoutCorrector),
        LayoutDM,
        LayoutSequenceTokenizer,
    )


def _vendor_config(dataset: str) -> tuple[DictConfig, DictConfig, DictConfig]:
    config_dir = (
        ROOT / "vendor" / "layout-corrector" / "src" / "trainer" / "trainer" / "config"
    )
    data = OmegaConf.create(
        {
            "pad_until_max": True,
            "shared_bbox_vocab": "x-y-w-h",
            "bbox_quantization": "kmeans",
            "num_bin_bboxes": 32,
            "special_tokens": ["pad", "mask"],
            "transforms": ["RandomOrder"],
            "var_order": "c-x-y-w-h",
        }
    )
    dataset_cfg = cast(
        DictConfig, OmegaConf.load(config_dir / "dataset" / f"{dataset}.yaml")
    )
    backbone = cast(DictConfig, OmegaConf.load(config_dir / "backbone" / "medium.yaml"))
    backbone.encoder_layer.timestep_type = "adalayernorm"
    backbone.encoder_layer.diffusion_step = 100
    backbone.encoder_layer.dropout = 0.0
    return data, dataset_cfg, backbone


def _checkpoint_path(dataset: str) -> Path:
    name = "rico" if dataset == "rico25" else "publaynet"
    return (
        _layout_dm_cache()
        / "pretrained_weights"
        / f"layoutdm_{name}"
        / "0"
        / "best_model.pt"
    )


def _cluster_path(dataset: str) -> Path:
    name = "rico25-max25" if dataset == "rico25" else "publaynet-max25"
    name = name.replace("-", "_")
    return (
        _layout_dm_cache() / "clustering_weights" / f"{name}_kmeans_train_clusters.pkl"
    )


def _package_layout_dm_config(dataset: str) -> LayoutDMConfig:
    return LayoutDMConfig(
        dataset_name=cast(LayoutDMTrainingDatasetName, dataset),
        max_seq_length=25,
        num_bin_bboxes=32,
        bbox_quantization="kmeans",
        cluster_centers_path=str(_cluster_path(dataset)),
        hidden_size=464,
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=1856,
        dropout=0.0,
        timestep_type="adalayernorm",
        num_timesteps=100,
        q_type="constrained",
    )


def _vendor_reference(
    dataset: str,
    tokenizer_cls: type[Any],
    layout_dm_cls: type[Any],
    backbone: DictConfig,
) -> tuple[torch.nn.Module, Any]:
    data, dataset_cfg, _ = _vendor_config(dataset)
    tokenizer = tokenizer_cls(data, dataset_cfg)
    model = layout_dm_cls(
        backbone_cfg=backbone,
        tokenizer=tokenizer,
        transformer_type="flattened",
        pos_emb="elem_attr",
        num_timesteps=100,
        q_type="constrained",
        seq_type="poset",
        use_padding_as_vocab=True,
        backbone_shrink_ratio=29 / 32,
        auxiliary_loss_weight=0.1,
    )
    raw = torch.load(_checkpoint_path(dataset), map_location="cpu", weights_only=False)
    model.load_state_dict(raw, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _corrector_config(dataset: str, vocab_size: int) -> LayoutCorrectorConfig:
    return LayoutCorrectorConfig(
        dataset_name=dataset,
        vocab_size=vocab_size,
        max_seq_length=25,
        hidden_size=CORRECTOR_HIDDEN_SIZE,
        intermediate_size=CORRECTOR_INTERMEDIATE_SIZE,
        num_attention_heads=8,
        num_hidden_layers=4,
        dropout=0.0,
        timestep_type="adalayernorm",
        num_timesteps=100,
        recon_type="x_t-1",
        target="recon_acc",
        attr_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        use_padding_as_vocab=True,
        pos_emb="none",
        transformer_type="aggregated",
    )


@dataclass
class Fixture:
    dataset: LayoutCorrectorTrainingDatasetName
    package: LayoutCorrectorTrainingModule
    vendor: VendorCorrector
    vendor_diffusion: torch.nn.Module
    package_reference: FrozenLayoutDMReference
    vendor_tokenizer: Any
    vendor_model_construction_device: str
    vendor_initialization_device: str
    package_construction_device: str
    package_initialization_device: str
    vendor_pre_model_rng_digest: str
    package_pre_model_rng_digest: str


def _fixture(
    dataset: LayoutCorrectorTrainingDatasetName,
    device: torch.device,
    seed: int = 123,
    *,
    align_corrector_weights: bool = True,
) -> Fixture:
    corrector_cls, layout_dm_cls, tokenizer_cls = _vendor_imports()
    _, _, backbone = _vendor_config(dataset)
    package_config = _package_layout_dm_config(dataset)
    torch.manual_seed(seed)
    vendor_pre_model_rng_digest = _state_digest(capture_rng_state())
    vendor_diffusion, vendor_tokenizer = _vendor_reference(
        dataset, tokenizer_cls, layout_dm_cls, backbone
    )
    construction_devices: list[str] = []
    data_parallel_init = torch.nn.DataParallel.__init__

    def capture_data_parallel_init(
        data_parallel: torch.nn.DataParallel[torch.nn.Module],
        module: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        construction_devices.append(_canonical_device(next(module.parameters()).device))
        data_parallel_init(data_parallel, module, *args, **kwargs)

    with patch.object(torch.nn.DataParallel, "__init__", capture_data_parallel_init):
        vendor_corrector = cast(
            VendorCorrector,
            corrector_cls(
                backbone_cfg=backbone,
                tokenizer=vendor_tokenizer,
                shrink_ratio=27 / 32,
                pos_emb="none",
                use_padding_as_vocab=True,
                num_timesteps=100,
                target="recon_acc",
                recon_type="x_t-1",
                transformer_type="aggregated",
            ),
        )
    torch.manual_seed(seed)
    package_pre_model_rng_digest = _state_digest(capture_rng_state())
    package = LayoutCorrectorTrainingModule(
        config=_corrector_config(dataset, package_config.vocab_size),
        layout_dm_checkpoint_path=_checkpoint_path(dataset),
        cluster_centers_path=_cluster_path(dataset),
        learning_rate=5.0e-4,
        weight_decay=0.1,
        betas=(0.9, 0.98),
        gradient_clip_norm=1.0,
    )
    package_reference = package._reference_value()
    assert package.pre_model_rng_digest == package_pre_model_rng_digest
    if len(construction_devices) != 1:
        raise AssertionError(
            f"expected one vendor DataParallel construction, got {construction_devices}"
        )
    vendor_model_construction_device = construction_devices[0]
    vendor_initialization_device = _canonical_device(
        next(vendor_corrector.parameters()).device
    )
    package_construction_device = package.model_construction_device
    package_initialization_device = _canonical_device(package.initialization_device)
    if align_corrector_weights:
        package.model.model.load_state_dict(
            vendor_corrector.model.module.state_dict(), strict=True
        )
    package.to(device)
    vendor_corrector.to(device)
    vendor_diffusion.to(device)
    vendor_corrector.eval()
    package.eval()
    vendor_diffusion.eval()
    return Fixture(
        dataset=dataset,
        package=package,
        vendor=vendor_corrector,
        vendor_diffusion=vendor_diffusion,
        package_reference=package_reference,
        vendor_tokenizer=vendor_tokenizer,
        vendor_model_construction_device=vendor_model_construction_device,
        vendor_initialization_device=vendor_initialization_device,
        package_construction_device=package_construction_device,
        package_initialization_device=package_initialization_device,
        vendor_pre_model_rng_digest=vendor_pre_model_rng_digest,
        package_pre_model_rng_digest=package_pre_model_rng_digest,
    )


def _shipped_training_config(dataset: str) -> DictConfig:
    config_path = (
        ROOT
        / "models"
        / "layout-corrector"
        / "configs"
        / "training"
        / f"layoutcorrector_{dataset}.yaml"
    )
    return cast(DictConfig, OmegaConf.load(config_path))


def _loader_worker_count(dataset: str) -> int:
    config = _shipped_training_config(dataset)
    return int(config.data.init_args.num_workers)


def _paired_real_batch(
    dataset: LayoutCorrectorTrainingDatasetName,
    split: LayoutCorrectorTrainingSplit,
    tokenizer: Any,
    *,
    batch_size: int = 4,
    random_order: bool = True,
    num_workers: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load one matching batch through both production dataset adapters."""
    from trainer.data.util import compose_transform, sparse_to_dense

    workers = _loader_worker_count(dataset) if num_workers is None else num_workers
    vendor_split = "val" if split == "validation" else split
    vendor_dataset = _vendor_dataset(
        dataset,
        vendor_split,
        transform=compose_transform(["RandomOrder"]) if random_order else None,
    )
    package_module = LayoutCorrectorDataModule(
        dataset_name=dataset,
        config=_package_layout_dm_config(dataset),
        processed_data_dir=_layout_dm_cache() / "datasets",
        batch_size=batch_size,
        num_workers=workers,
        random_order=random_order,
        pin_memory=False,
    )
    package_module.setup("fit" if split in {"train", "validation"} else "test")
    package_loader = (
        package_module.train_dataloader()
        if split == "train"
        else package_module.val_dataloader()
        if split == "validation"
        else package_module.test_dataloader()
    )
    vendor_loader = GeometricDataLoader(
        vendor_dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
    )
    seed = 42975
    torch.manual_seed(seed)
    vendor_batch = next(iter(vendor_loader))
    torch.manual_seed(seed)
    package_batch = next(iter(package_loader))
    bbox, labels, _, mask = sparse_to_dense(vendor_batch)
    vendor_encoded = tokenizer.encode({"bbox": bbox, "label": labels, "mask": mask})
    package_input_ids = cast(torch.Tensor, package_batch["input_ids"])
    package_mask = cast(torch.Tensor, package_batch["attention_mask"])
    stream_report = compare_batch_stream(
        [
            {
                "input_ids": vendor_encoded["seq"],
                "attention_mask": vendor_encoded["mask"],
            }
        ],
        [{"input_ids": package_input_ids, "attention_mask": package_mask}],
        steps=1,
    )
    assert stream_report.passed, stream_report
    return (
        {
            "input_ids": vendor_encoded["seq"],
            "attention_mask": vendor_encoded["mask"],
        },
        package_batch,
    )


def _vendor_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "seq": cast(torch.Tensor, batch["input_ids"]).to(device),
        "mask": cast(torch.Tensor, batch["attention_mask"]).to(device),
    }


def _vendor_trace(fixture: Fixture, batch: dict[str, Any]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    captured_probability: list[torch.Tensor] = []
    diffusion_model = cast(Any, fixture.vendor_diffusion).model
    original_sample_time = diffusion_model.sample_time

    def sample_time_with_probability(
        batch_size: int, device: torch.device, method: str = "uniform"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps, probabilities = original_sample_time(batch_size, device, method)
        captured_probability.append(probabilities.detach())
        return timesteps, probabilities

    vendor_batch = _vendor_batch(batch, next(fixture.vendor.parameters()).device)
    diffusion_model.sample_time = sample_time_with_probability
    try:
        prepared = fixture.vendor.preprocess(
            vendor_batch,
            fixture.vendor_diffusion,
            OmegaConf.create({"name": "random", "temperature": 1.0}),
        )
    finally:
        diffusion_model.sample_time = original_sample_time
    if len(captured_probability) != 1:
        raise AssertionError("vendor importance-sampler probability was not captured")
    prepared["pt"] = captured_probability[0]
    outputs, losses = fixture.vendor(prepared)
    logits = outputs["logits"].squeeze(-1).detach()
    target = prepared["recon_acc"].float()
    attribute_count = len(cast(Any, fixture.vendor).tokenizer.var_names)
    element_count = logits.shape[-1] // attribute_count
    weights = (
        torch.as_tensor(
            cast(Any, fixture.vendor).attr_loss_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        .repeat(element_count, 1)
        .reshape(1, -1)
    )
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce_loss = bce_loss * weights
    loss = losses["bce_loss"]
    return {
        **prepared,
        "logits": logits,
        "bce_loss": bce_loss.detach(),
        "weighted_bce_loss": weighted_bce_loss.detach(),
        "train_loss": loss,
        "input_ids_digest": _tensor_digest(vendor_batch["seq"]),
        "attention_mask_digest": _tensor_digest(vendor_batch["mask"]),
    }


def _package_trace(fixture: Fixture, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    prepared = fixture.package.preprocess(
        {
            key: value.to(next(fixture.package.parameters()).device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
    )
    loss, trace = fixture.package._corrector_loss(prepared)
    trace["mask"] = (prepared["xt"] == prepared["x0"]).long()
    trace["train_loss"] = loss
    return trace


def _tensor_diffs(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, float], str | None]:
    report = compare_layout_corrector_step(
        build_layout_corrector_step_trace("reference", reference),
        build_layout_corrector_step_trace("target", target),
    )
    diffs = {item.name: item.max_abs_diff for item in report.comparisons}
    diffs.update({name: float("inf") for name in report.missing})
    first = next(
        (item.name for item in report.comparisons if not item.passed),
        report.missing[0] if report.missing else None,
    )
    return diffs, first


def _state_diffs(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, float], str | None]:
    report = compare_optimizer_step(
        {name: value.detach().cpu() for name, value in reference.items()},
        {name: value.detach().cpu() for name, value in target.items()},
    )
    diffs = {item.name: item.max_abs_diff for item in report.comparisons}
    diffs.update({name: float("inf") for name in report.missing})
    first = next(
        (item.name for item in report.comparisons if not item.passed),
        report.missing[0] if report.missing else None,
    )
    return diffs, first


def _mapping_diffs(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, float], str | None]:
    diffs: dict[str, float] = {}
    first: str | None = None
    for name, value in reference.items():
        target_value = target.get(name)
        if not isinstance(target_value, torch.Tensor):
            diffs[name] = float("inf")
            first = first or name
            continue
        comparison = compare_tensors(name, target_value, value)
        diffs[name] = comparison.max_abs_diff
        if not comparison.passed and first is None:
            first = name
    return diffs, first


def _paired_trace(
    fixture: Fixture,
    vendor_batch: dict[str, torch.Tensor],
    package_batch: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = capture_rng_state()
    vendor_trace = _vendor_trace(fixture, vendor_batch)
    restore_rng_state(state)
    package_trace = _package_trace(fixture, package_batch)
    return vendor_trace, package_trace


def _gradient_diffs(
    vendor: torch.nn.Module, package: torch.nn.Module
) -> tuple[float, str | None]:
    package_parameters = dict(package.named_parameters())
    maximum = 0.0
    first: str | None = None
    for name, vendor_parameter in vendor.named_parameters():
        package_parameter = package_parameters[name]
        if vendor_parameter.grad is None or package_parameter.grad is None:
            continue
        difference = (
            vendor_parameter.grad.detach().float()
            - package_parameter.grad.detach().float()
        ).abs()
        value = float(difference.max().item()) if difference.numel() else 0.0
        maximum = max(maximum, value)
        if first is None and value > 0.0:
            first = name
    return maximum, first


def _adamw(
    parameters: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
) -> torch.optim.AdamW:
    mode = os.environ.get("LAYOUT_CORRECTOR_ADAMW_MODE", "default")
    options: dict[str, bool] = {}
    if mode == "scalar":
        options = {"foreach": False, "fused": False}
    elif mode == "foreach":
        options = {"foreach": True, "fused": False}
    return torch.optim.AdamW(parameters, lr=5e-4, betas=(0.9, 0.98), **options)


def _clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter], *, foreach: bool | None = None
) -> torch.Tensor:
    max_norm = (
        float("inf") if os.environ.get("LAYOUT_CORRECTOR_DISABLE_CLIP") == "1" else 1.0
    )
    return torch.nn.utils.clip_grad_norm_(
        list(parameters),
        max_norm,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=foreach,
    )


def _s3_steps() -> int:
    """Return the fixed protocol bound for the shipped training configs."""
    return S3_STEP_BOUND


def _s3_stage(name: str) -> str:
    tag = os.environ.get("LAYOUT_CORRECTOR_DIAGNOSTIC_TAG")
    return name if not tag else f"{name}/{tag}"


def _parameter_order_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode())
        digest.update(
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _optimizer_defaults(optimizer: torch.optim.Optimizer) -> dict[str, str]:
    return {
        key: repr(value)
        for key, value in sorted(optimizer.defaults.items())
        if key != "weight_decay"
    }


def _optimizer_group_weight_decays(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["weight_decay"]) for group in optimizer.param_groups]


def _optimizer_state_tensors(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    state = optimizer.state_dict()["state"]
    flattened: dict[str, torch.Tensor] = {}
    for state_id, values in state.items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                flattened[f"{state_id}:{name}"] = value.detach()
            elif isinstance(value, (int, float)):
                flattened[f"{state_id}:{name}"] = torch.tensor(value)
            else:
                raise TypeError(f"unsupported optimizer state value: {type(value)!r}")
    return flattened


def _optimizer_state_diffs(
    reference: torch.optim.Optimizer, target: torch.optim.Optimizer
) -> tuple[dict[str, float], str | None]:
    return _state_diffs(
        _optimizer_state_tensors(reference), _optimizer_state_tensors(target)
    )


def _vendor_dataset(dataset: str, split: str, *, transform: Any = None) -> Any:
    _vendor_imports()
    if dataset == "rico25":
        from trainer.datasets.rico import Rico25Dataset

        dataset_cls = Rico25Dataset
    else:
        from trainer.datasets.publaynet import PubLayNetDataset

        dataset_cls = PubLayNetDataset
    return dataset_cls(
        str(_layout_dm_cache() / "datasets"),
        split,
        25,
        transform=transform,
    )


def _vendor_encoded_sample(sample: Any, tokenizer: Any) -> dict[str, torch.Tensor]:
    return tokenizer.encode(
        {
            "bbox": sample.x.unsqueeze(0),
            "label": sample.y.unsqueeze(0),
            "mask": torch.ones((1, sample.y.shape[0]), dtype=torch.bool),
        }
    )


def _apply_s3_determinism() -> None:
    deterministic = os.environ.get("LAYOUT_CORRECTOR_DETERMINISTIC_ALGORITHMS") == "1"
    apply_determinism(
        DeterminismConfig(seed=42975, deterministic_algorithms=deterministic)
    )
    torch.backends.cudnn.deterministic = deterministic


def _optimizer_group_names(
    groups: list[dict[str, Any]],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[dict[str, Any]]:
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    return [
        {
            "names": [names_by_id[id(parameter)] for parameter in group["params"]],
            "weight_decay": float(group["weight_decay"]),
        }
        for group in groups
    ]


@pytest.mark.parametrize("dataset", DATASETS)
def test_s0_training_static_state_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=123, deterministic_algorithms=False))
    fixture = _fixture(dataset, torch.device("cpu"), align_corrector_weights=False)
    checkpoint = _checkpoint_path(dataset)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    package_state = fixture.package_reference.model.state_dict()
    frozen_diffs: dict[str, float] = {}
    for key, value in package_state.items():
        raw_key = f"model.module.{key}"
        frozen_diffs[key] = float((value - raw[raw_key]).abs().max().item())
    vendor_state = fixture.vendor.model.module.state_dict()
    package_state_corrector = fixture.package.model.model.state_dict()
    corrector_diffs, corrector_first = _state_diffs(
        vendor_state, package_state_corrector
    )
    vendor_optimizer_groups = _optimizer_group_names(
        fixture.vendor.optim_groups(weight_decay=0.1),
        fixture.vendor.model.module.named_parameters(),
    )
    package_optimizer_groups = _optimizer_group_names(
        fixture.package.optim_groups(), fixture.package.model.model.named_parameters()
    )
    vendor_optimizer = torch.optim.AdamW(
        fixture.vendor.optim_groups(weight_decay=0.1), lr=5.0e-4, betas=(0.9, 0.98)
    )
    package_optimizer = torch.optim.AdamW(
        fixture.package.optim_groups(), lr=5.0e-4, betas=(0.9, 0.98)
    )
    vendor_input, package_batch = _paired_real_batch(
        dataset,
        "train",
        fixture.vendor_tokenizer,
        random_order=False,
        num_workers=0,
    )
    dataset_input_diffs, dataset_first = _mapping_diffs(vendor_input, package_batch)
    vendor_frozen_model = cast(Any, fixture.vendor_diffusion).model.module
    vendor_frozen_state = vendor_frozen_model.state_dict()
    package_frozen_state = fixture.package_reference.model.state_dict()
    common_frozen_keys = set(vendor_frozen_state) & set(package_frozen_state)
    frozen_vendor_diffs, frozen_vendor_first = _state_diffs(
        {key: vendor_frozen_state[key] for key in common_frozen_keys},
        {key: package_frozen_state[key] for key in common_frozen_keys},
    )
    vendor_diffusion = vendor_frozen_model
    history_diff = float(
        (vendor_diffusion.Lt_history - fixture.package_reference.lt_history)
        .abs()
        .max()
        .item()
    )
    count_diff = float(
        (vendor_diffusion.Lt_count - fixture.package_reference.lt_count)
        .abs()
        .max()
        .item()
    )
    assert _sha256(checkpoint) == CHECKPOINT_SHA256[dataset]
    assert max(frozen_diffs.values()) == 0.0
    assert not frozen_vendor_first, frozen_vendor_diffs
    assert history_diff == 0.0
    assert count_diff == 0.0
    assert not dataset_first, dataset_input_diffs
    assert not corrector_first, corrector_diffs
    assert (
        fixture.vendor_model_construction_device == fixture.package_construction_device
    )
    assert fixture.vendor_initialization_device == fixture.package_initialization_device
    assert not set(vendor_state) - set(package_state_corrector)
    assert not set(package_state_corrector) - set(vendor_state)
    assert sum(value.numel() for value in vendor_state.values()) == sum(
        value.numel() for value in package_state_corrector.values()
    )
    assert vendor_optimizer_groups == package_optimizer_groups
    assert vendor_optimizer.defaults == package_optimizer.defaults
    vendor_optimizer_state_entries = len(vendor_optimizer.state)
    package_optimizer_state_entries = len(package_optimizer.state)
    assert vendor_optimizer_state_entries == package_optimizer_state_entries
    path = _write_json(
        "s0-static",
        dataset,
        {
            "dataset": dataset,
            "frozen_layoutdm_checkpoint": str(
                checkpoint.relative_to(_layout_dm_cache())
            ),
            "frozen_layoutdm_sha256": _sha256(checkpoint),
            "frozen_layoutdm_max_abs_diff": max(frozen_diffs.values()),
            "frozen_layoutdm_system_max_abs_diff": max(frozen_vendor_diffs.values()),
            "frozen_layoutdm_tensors_identical": not frozen_vendor_first,
            "frozen_layoutdm_sampler_history_max_abs_diff": history_diff,
            "frozen_layoutdm_sampler_count_max_abs_diff": count_diff,
            "frozen_layoutdm_vendor_extra_state_keys": sorted(
                set(vendor_frozen_state) - set(package_frozen_state)
            ),
            "frozen_layoutdm_package_extra_state_keys": sorted(
                set(package_frozen_state) - set(vendor_frozen_state)
            ),
            "vendor_corrector_parameter_count": sum(
                value.numel() for value in vendor_state.values()
            ),
            "package_corrector_parameter_count": sum(
                value.numel() for value in package_state_corrector.values()
            ),
            "vendor_corrector_parameter_tensor_count": len(vendor_state),
            "package_corrector_parameter_tensor_count": len(package_state_corrector),
            "corrector_state_key_missing_from_package": sorted(
                set(vendor_state) - set(package_state_corrector)
            ),
            "corrector_state_key_extra_in_package": sorted(
                set(package_state_corrector) - set(vendor_state)
            ),
            "native_initialization_max_abs_diff": max(corrector_diffs.values()),
            "native_initialization_first_difference": corrector_first,
            "vendor_model_construction_device": fixture.vendor_model_construction_device,
            "vendor_initialization_device": fixture.vendor_initialization_device,
            "package_construction_device": fixture.package_construction_device,
            "package_initialization_device": fixture.package_initialization_device,
            "construction_device_equal": fixture.vendor_model_construction_device
            == fixture.package_construction_device,
            "initialization_device_equal": fixture.vendor_initialization_device
            == fixture.package_initialization_device,
            "parameter_registration_order_equal": [
                name for name, _ in fixture.vendor.model.module.named_parameters()
            ]
            == [name for name, _ in fixture.package.model.model.named_parameters()],
            "optimizer_groups_equal": vendor_optimizer_groups
            == package_optimizer_groups,
            "optimizer_defaults": {
                "vendor": _optimizer_defaults(vendor_optimizer),
                "package": _optimizer_defaults(package_optimizer),
                "equal": vendor_optimizer.defaults == package_optimizer.defaults,
            },
            "optimizer_group_weight_decays": {
                "vendor": _optimizer_group_weight_decays(vendor_optimizer),
                "package": _optimizer_group_weight_decays(package_optimizer),
                "equal": _optimizer_group_weight_decays(vendor_optimizer)
                == _optimizer_group_weight_decays(package_optimizer),
            },
            "optimizer_initial_state_entries": {
                "vendor": vendor_optimizer_state_entries,
                "package": package_optimizer_state_entries,
            },
            "dataset_static_values_equal": not dataset_first,
            "dataset_static_max_abs_diff": max(dataset_input_diffs.values()),
            "first_divergence": {
                "frozen_layoutdm": frozen_vendor_first,
                "dataset": dataset_first,
                "native_initialization": corrector_first,
            },
            "runtime": _runtime_record(),
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()


@pytest.mark.parametrize("dataset", DATASETS)
def test_s1_fixed_batch_pre_optimizer_trace_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=42975, deterministic_algorithms=False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixture = _fixture(dataset, device)
    vendor_batch, package_batch = _paired_real_batch(
        dataset, "train", fixture.vendor_tokenizer
    )
    vendor_trace, package_trace = _paired_trace(fixture, vendor_batch, package_batch)
    diffs, first = _tensor_diffs(vendor_trace, package_trace)
    assert not first, diffs
    path = _write_json(
        "s1-fixed-batch",
        dataset,
        {
            "dataset": dataset,
            "max_abs_diffs": diffs,
            "first_divergence": first,
            "runtime": _runtime_record(),
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()


@pytest.mark.parametrize("dataset", DATASETS)
def test_s2_one_optimizer_step_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=42975, deterministic_algorithms=False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixture = _fixture(dataset, device)
    vendor_batch, package_batch = _paired_real_batch(
        dataset, "train", fixture.vendor_tokenizer
    )
    vendor_trace, package_trace = _paired_trace(fixture, vendor_batch, package_batch)
    vendor_optimizer = torch.optim.AdamW(
        fixture.vendor.optim_groups(weight_decay=0.1), lr=5e-4, betas=(0.9, 0.98)
    )
    package_optimizer = torch.optim.AdamW(
        fixture.package.optim_groups(), lr=5e-4, betas=(0.9, 0.98)
    )
    vendor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        vendor_optimizer, mode="min", factor=0.5, patience=2, threshold=1.0e-2
    )
    package_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        package_optimizer, mode="min", factor=0.5, patience=2, threshold=1.0e-2
    )
    vendor_optimizer.zero_grad()
    package_optimizer.zero_grad()
    vendor_loss = vendor_trace["train_loss"]
    package_loss = package_trace["train_loss"]
    vendor_loss.backward()
    package_loss.backward()
    gradient_max_abs_diff, gradient_first = _gradient_diffs(
        fixture.vendor.model.module, fixture.package.model.model
    )
    assert gradient_first is None, {
        "first_gradient_difference": gradient_first,
        "max_abs_gradient_diff": gradient_max_abs_diff,
    }
    assert gradient_max_abs_diff == 0.0
    vendor_grad_norm = torch.nn.utils.clip_grad_norm_(fixture.vendor.parameters(), 1.0)
    package_grad_norm = torch.nn.utils.clip_grad_norm_(
        fixture.package.model.parameters(), 1.0
    )
    vendor_optimizer.step()
    package_optimizer.step()
    optimizer_state_diffs, optimizer_state_first = _optimizer_state_diffs(
        vendor_optimizer, package_optimizer
    )
    assert optimizer_state_first is None, {
        "first_optimizer_state_difference": optimizer_state_first,
        "optimizer_state_diffs": optimizer_state_diffs,
    }
    assert max(optimizer_state_diffs.values(), default=0.0) == 0.0
    diffs, first = _state_diffs(
        fixture.vendor.model.module.state_dict(),
        fixture.package.model.model.state_dict(),
    )
    assert not first, diffs
    assert torch.allclose(vendor_grad_norm, package_grad_norm, atol=1e-5, rtol=1e-5)
    vendor_scheduler_digest = _state_digest(vendor_scheduler.state_dict())
    package_scheduler_digest = _state_digest(package_scheduler.state_dict())
    assert vendor_scheduler_digest == package_scheduler_digest
    path = _write_json(
        "s2-optimizer-step",
        dataset,
        {
            "dataset": dataset,
            "max_abs_parameter_diffs": diffs,
            "first_divergence": first,
            "vendor_gradient_norm": float(vendor_grad_norm.item()),
            "package_gradient_norm": float(package_grad_norm.item()),
            "max_abs_gradient_tensor_diff": gradient_max_abs_diff,
            "first_gradient_tensor_difference": gradient_first,
            "max_abs_optimizer_state_diff": max(
                optimizer_state_diffs.values(), default=0.0
            ),
            "first_optimizer_state_difference": optimizer_state_first,
            "scheduler_state": {
                "vendor_digest": vendor_scheduler_digest,
                "package_digest": package_scheduler_digest,
                "equal": vendor_scheduler_digest == package_scheduler_digest,
            },
            "runtime": _runtime_record(),
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()


def _natural_training_batches(
    dataset: LayoutCorrectorTrainingDatasetName,
    tokenizer: Any,
    steps: int,
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Materialize paired production train and validation loader streams."""
    from trainer.data.util import compose_transform, sparse_to_dense

    workers = _loader_worker_count(dataset)
    vendor_dataset = _vendor_dataset(
        dataset, "train", transform=compose_transform(["RandomOrder"])
    )
    package_module = LayoutCorrectorDataModule(
        dataset_name=dataset,
        config=_package_layout_dm_config(dataset),
        processed_data_dir=_layout_dm_cache() / "datasets",
        batch_size=64,
        num_workers=workers,
        random_order=True,
        pin_memory=True,
    )
    package_module.setup("fit")
    vendor_loader = GeometricDataLoader(
        vendor_dataset, batch_size=64, shuffle=True, num_workers=workers
    )
    package_loader = package_module.train_dataloader()
    seed = 42975
    torch.manual_seed(seed)
    vendor_pre_loader_rng_digest = _state_digest(capture_rng_state())
    vendor_iterator = iter(vendor_loader)
    vendor_batches: list[dict[str, Any]] = []
    vendor_ids: list[list[str]] = []
    for batch_index, batch in enumerate(vendor_iterator):
        if batch_index >= steps:
            break
        bbox, labels, _, mask = sparse_to_dense(batch)
        encoded = tokenizer.encode({"bbox": bbox, "label": labels, "mask": mask})
        sample_ids = [str(value) for value in batch.attr["name"]]
        vendor_batches.append(
            {
                "input_ids": encoded["seq"],
                "attention_mask": encoded["mask"],
                "id": sample_ids,
            }
        )
        vendor_ids.append(sample_ids)
    torch.manual_seed(seed)
    package_pre_loader_rng_digest = _state_digest(capture_rng_state())
    package_iterator = iter(package_loader)
    package_batches: list[dict[str, Any]] = []
    package_ids: list[list[str]] = []
    for batch_index, batch in enumerate(package_iterator):
        if batch_index >= steps:
            break
        package_batches.append(batch)
        values = batch.get("id", [])
        package_ids.append(
            [str(value) for value in values]
            if isinstance(values, (list, tuple))
            else [str(values)]
        )
    stream_report = compare_batch_stream(
        vendor_batches,
        package_batches,
        steps=min(len(vendor_batches), len(package_batches)),
    )
    assert stream_report.passed, stream_report
    assert len(vendor_batches) == len(package_batches) == steps
    assert vendor_ids == package_ids
    del vendor_iterator, package_iterator
    gc.collect()
    loader_record = {
        "vendor_pre_loader_rng_digest": vendor_pre_loader_rng_digest,
        "package_pre_loader_rng_digest": package_pre_loader_rng_digest,
        "configured_num_workers": workers,
        "stage_step_bound": _s3_steps(),
        "vendor_train_loader_num_workers": vendor_loader.num_workers,
        "package_train_loader_num_workers": package_loader.num_workers,
    }
    assert vendor_pre_loader_rng_digest == package_pre_loader_rng_digest

    vendor_validation_dataset = _vendor_dataset(
        dataset, "val", transform=compose_transform(["RandomOrder"])
    )
    package_validation_module = LayoutCorrectorDataModule(
        dataset_name=dataset,
        config=_package_layout_dm_config(dataset),
        processed_data_dir=_layout_dm_cache() / "datasets",
        batch_size=64,
        num_workers=workers,
        random_order=True,
        pin_memory=True,
    )
    package_validation_module.setup("fit")
    vendor_validation_loader = GeometricDataLoader(
        vendor_validation_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=workers,
    )
    package_validation_loader = package_validation_module.val_dataloader()
    torch.manual_seed(seed)
    vendor_validation_batches: list[dict[str, Any]] = []
    for batch in vendor_validation_loader:
        bbox, labels, _, mask = sparse_to_dense(batch)
        encoded = tokenizer.encode({"bbox": bbox, "label": labels, "mask": mask})
        vendor_validation_batches.append(
            {
                "input_ids": encoded["seq"],
                "attention_mask": encoded["mask"],
                "id": [str(value) for value in batch.attr["name"]],
            }
        )
    torch.manual_seed(seed)
    package_validation_batches: list[dict[str, Any]] = [
        {
            "input_ids": cast(torch.Tensor, batch["input_ids"]),
            "attention_mask": cast(torch.Tensor, batch["attention_mask"]),
            "id": [str(value) for value in batch.get("id", [])],
        }
        for batch in package_validation_loader
    ]
    del vendor_validation_loader, package_validation_loader
    gc.collect()
    validation_stream_report = compare_batch_stream(
        vendor_validation_batches,
        package_validation_batches,
        steps=len(vendor_validation_batches),
    )
    assert validation_stream_report.passed, validation_stream_report
    assert len(vendor_validation_batches) == len(package_validation_batches)
    assert [batch["id"] for batch in vendor_validation_batches] == [
        batch["id"] for batch in package_validation_batches
    ]
    loader_record["validation_batch_stream_digest"] = _state_digest(
        {
            "vendor": [
                _tensor_digest(batch["input_ids"])
                for batch in vendor_validation_batches
            ],
            "package": [
                _tensor_digest(batch["input_ids"])
                for batch in package_validation_batches
            ],
        }
    )
    return (
        vendor_batches,
        package_batches,
        vendor_validation_batches,
        package_validation_batches,
        loader_record,
    )


def _run_natural_side(
    fixture: Fixture,
    batches: list[dict[str, Any]],
    validation_batches: list[dict[str, Any]],
    *,
    side: str,
    initial_state: dict[str, torch.Tensor],
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    """Run one system without restoring RNG inside the multi-step trajectory."""
    if side == "vendor":
        model = fixture.vendor.model.module
        optimizer = _adamw(fixture.vendor.optim_groups(weight_decay=0.1))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=1.0e-2
        )
    else:
        return _run_package_natural_side(
            fixture,
            batches,
            validation_batches,
            initial_state=initial_state,
            steps=steps,
        )

    model.load_state_dict(initial_state, strict=True)
    model.train()
    _apply_s3_determinism()
    rows: list[dict[str, Any]] = []
    for step, batch in enumerate(batches[:steps]):
        optimizer.zero_grad()
        rng_digest = _state_digest(capture_rng_state())
        trace = _vendor_trace(fixture, batch)
        trace["train_loss"].backward()
        grad_norm = _clip_grad_norm(fixture.vendor.parameters(), foreach=False)
        optimizer.step()
        loss = float(trace["train_loss"].detach().cpu().item())
        importance_probability = trace.get("pt")
        rows.append(
            {
                "step": step,
                "loss": loss,
                "gradient_norm": float(grad_norm.detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_state_digest": _optimizer_state_digest(optimizer),
                "parameter_state_digest": _state_dict_digest(model.state_dict()),
                "rng_digest": rng_digest,
                "sample_ids": [str(value) for value in batch.get("id", [])],
                "timesteps_digest": _tensor_digest(trace["t"]),
                "importance_probability_digest": (
                    None
                    if importance_probability is None
                    else _tensor_digest(importance_probability)
                ),
                "input_ids_digest": trace["input_ids_digest"],
                "attention_mask_digest": trace["attention_mask_digest"],
                "corrupted_tokens_digest": _tensor_digest(trace["xt"]),
                "reconstructed_tokens_digest": _tensor_digest(trace["x0_recon"]),
                "model_training": model.training,
                "bce_loss_digest": _tensor_digest(trace["bce_loss"]),
                "weighted_bce_loss_digest": _tensor_digest(trace["weighted_bce_loss"]),
            }
        )
    model.eval()
    with torch.no_grad():
        validation_loss_sum = torch.zeros(
            (),
            device=next(fixture.vendor.parameters()).device,
            dtype=torch.float32,
        )
        validation_sample_count = 0
        for batch in validation_batches:
            loss = _vendor_trace(fixture, batch)["train_loss"]
            batch_size = int(cast(torch.Tensor, batch["input_ids"]).shape[0])
            validation_loss_sum += loss * batch_size
            validation_sample_count += batch_size
    validation_loss = float(
        (validation_loss_sum / validation_sample_count).cpu().item()
    )
    scheduler.step(validation_loss)
    return (
        rows,
        deepcopy(model.state_dict()),
        {
            "scheduler_class": type(scheduler).__name__,
            "scheduler_state": scheduler.state_dict(),
            "scheduler_state_digest": _scheduler_state_digest(
                cast("DigestValue", scheduler.state_dict())
            ),
            "validation_batches": len(validation_batches),
            "validation_executed": True,
            "validation_loss": validation_loss,
            "pre_model_rng_digest": fixture.vendor_pre_model_rng_digest,
        },
    )


def _run_package_natural_side(
    fixture: Fixture,
    batches: list[dict[str, Any]],
    validation_batches: list[dict[str, Any]],
    *,
    initial_state: dict[str, torch.Tensor],
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    del batches, validation_batches
    with tempfile.TemporaryDirectory(prefix="layout-corrector-natural-") as root_text:
        root = Path(root_text)
        initial_state_path = root / "initial-state.pt"
        final_state_path = root / "final-state.pt"
        trace_path = root / "trace.json"
        torch.save(initial_state, initial_state_path)
        config_path = (
            ROOT
            / "models"
            / "layout-corrector"
            / "configs"
            / "training"
            / f"layoutcorrector_{fixture.dataset}.yaml"
        )
        cluster_path = _cluster_path(fixture.dataset)
        checkpoint_path = _checkpoint_path(fixture.dataset)
        executable = Path(sys.executable).with_name("traingen")
        command = [
            str(executable),
            "fit",
            "--config",
            str(config_path),
            f"--seed_everything={NATURAL_STREAM_SEED}",
            f"--trainer.max_steps={steps}",
            f"--trainer.limit_train_batches={steps}",
            f"--trainer.val_check_interval={steps}",
            f"--trainer.default_root_dir={root / 'trainer'}",
            f"--model.init_args.layout_dm_checkpoint_path={checkpoint_path}",
            f"--model.init_args.cluster_centers_path={cluster_path}",
            f"--data.init_args.config.cluster_centers_path={cluster_path}",
            f"--data.init_args.processed_data_dir={_layout_dm_cache() / 'datasets'}",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        environment["LAYOUT_CORRECTOR_TRACE_PATH"] = str(trace_path)
        environment["LAYOUT_CORRECTOR_INITIAL_STATE_PATH"] = str(initial_state_path)
        environment["LAYOUT_CORRECTOR_FINAL_STATE_PATH"] = str(final_state_path)
        environment["LAYOUT_CORRECTOR_TRACE_SEED"] = str(NATURAL_STREAM_SEED)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        record = json.loads(trace_path.read_text())
        if command[1:3] != ["fit", "--config"]:
            raise AssertionError(f"unexpected production command: {command}")
        record["production_command"] = command
        record["production_path"] = command[0:2]
        record["production_config"] = str(config_path.relative_to(ROOT))
        shipped_config = _shipped_training_config(fixture.dataset)
        record["configured_num_workers"] = int(
            shipped_config.data.init_args.num_workers
        )
        record["configured_max_epochs"] = int(shipped_config.trainer.max_epochs)
        record["stage_step_bound"] = _s3_steps()
        rows = cast(list[dict[str, Any]], record["rows"])
        if len(rows) != steps:
            raise AssertionError(
                f"production path recorded {len(rows)} of {steps} steps"
            )
        final_state = torch.load(
            final_state_path, map_location="cpu", weights_only=True
        )
        if not isinstance(final_state, dict):
            raise TypeError("production path final state is not a state dictionary")
        return rows, cast(dict[str, torch.Tensor], final_state), record


def _natural_comparison(
    dataset: LayoutCorrectorTrainingDatasetName,
    vendor_rows: list[dict[str, Any]],
    package_rows: list[dict[str, Any]],
    vendor_record: dict[str, Any],
    package_record: dict[str, Any],
    loader_record: dict[str, Any],
) -> dict[str, Any]:
    assert len(vendor_rows) == len(package_rows)
    assert Path(package_record["production_path"][0]).name == "traingen"
    assert package_record["production_path"][1] == "fit"
    production_command = package_record["production_command"]
    allowed_overrides = (
        "--seed_everything=",
        "--trainer.max_steps=",
        "--trainer.limit_train_batches=",
        "--trainer.val_check_interval=",
        "--trainer.default_root_dir=",
        "--model.init_args.layout_dm_checkpoint_path=",
        "--model.init_args.cluster_centers_path=",
        "--data.init_args.config.cluster_centers_path=",
        "--data.init_args.processed_data_dir=",
    )
    overrides = [
        argument for argument in production_command[3:] if argument.startswith("--")
    ]
    assert all(argument.startswith(allowed_overrides) for argument in overrides), (
        production_command,
        overrides,
    )
    assert package_record["scheduler_class"] == vendor_record["scheduler_class"]
    assert isinstance(package_record["scheduler_state_digest"], str)
    assert package_record["scheduler_state_digest"]
    assert isinstance(vendor_record["scheduler_state_digest"], str)
    assert vendor_record["scheduler_state_digest"]
    assert package_record["validation_batches"] == vendor_record["validation_batches"]
    assert package_record["validation_executed"] is True
    assert vendor_record["validation_executed"] is True
    validation_loss_abs_diff = abs(
        package_record["validation_loss"] - vendor_record["validation_loss"]
    )
    assert validation_loss_abs_diff <= VALIDATION_LOSS_TOLERANCE, {
        "package_validation_loss": package_record["validation_loss"],
        "vendor_validation_loss": vendor_record["validation_loss"],
        "abs_diff": validation_loss_abs_diff,
        "tolerance": VALIDATION_LOSS_TOLERANCE,
    }
    package_scheduler_state = package_record["scheduler_state"]
    vendor_scheduler_state = vendor_record["scheduler_state"]
    assert package_scheduler_state.keys() == vendor_scheduler_state.keys()
    scheduler_state_mismatches: dict[str, tuple[Any, Any]] = {}
    scheduler_best_abs_diff: float | None = None
    for key in package_scheduler_state:
        package_value = package_scheduler_state[key]
        vendor_value = vendor_scheduler_state[key]
        if key == "best":
            scheduler_best_abs_diff = abs(package_value - vendor_value)
            if scheduler_best_abs_diff > VALIDATION_LOSS_TOLERANCE:
                scheduler_state_mismatches[key] = (package_value, vendor_value)
        elif package_value != vendor_value:
            scheduler_state_mismatches[key] = (package_value, vendor_value)
    scheduler_state_equal = not scheduler_state_mismatches
    assert scheduler_state_equal, {
        "package_scheduler_state": package_scheduler_state,
        "vendor_scheduler_state": vendor_scheduler_state,
        "mismatches": scheduler_state_mismatches,
        "best_abs_diff": scheduler_best_abs_diff,
        "tolerance": VALIDATION_LOSS_TOLERANCE,
    }
    shipped_config = _shipped_training_config(dataset)
    configured_workers = int(shipped_config.data.init_args.num_workers)
    configured_max_epochs = int(shipped_config.trainer.max_epochs)
    expected_config = str(
        (
            ROOT
            / "models"
            / "layout-corrector"
            / "configs"
            / "training"
            / f"layoutcorrector_{dataset}.yaml"
        ).relative_to(ROOT)
    )
    assert package_record["trace_seed"] == NATURAL_STREAM_SEED
    assert package_record["production_config"] == expected_config
    assert package_record["num_workers"] == configured_workers
    assert package_record["configured_num_workers"] == configured_workers
    assert package_record["max_epochs"] == configured_max_epochs
    assert package_record["configured_max_epochs"] == configured_max_epochs
    assert loader_record["configured_num_workers"] == configured_workers
    assert loader_record["vendor_train_loader_num_workers"] == configured_workers
    assert loader_record["package_train_loader_num_workers"] == configured_workers
    assert package_record["model_training"] is True
    assert package_record["validation_batches"]
    assert package_record["observed_train_batches"] == len(package_rows)
    assert package_record["max_steps"] == _s3_steps() == len(package_rows)
    assert package_record["stage_step_bound"] == _s3_steps()
    assert loader_record["stage_step_bound"] == _s3_steps()
    assert package_record["train_batches"] >= len(package_rows)
    assert (
        package_record["pre_model_rng_digest"] == vendor_record["pre_model_rng_digest"]
    )
    assert (
        package_record["pre_loader_rng_digest"]
        == loader_record["package_pre_loader_rng_digest"]
    )
    assert (
        loader_record["vendor_pre_loader_rng_digest"]
        == loader_record["package_pre_loader_rng_digest"]
    )
    comparison_fields = (
        "input_ids_digest",
        "attention_mask_digest",
        "timesteps_digest",
        "corrupted_tokens_digest",
        "reconstructed_tokens_digest",
        "bce_loss_digest",
        "weighted_bce_loss_digest",
        "rng_digest",
        "optimizer_state_digest",
        "parameter_state_digest",
        "model_training",
    )
    digest_mismatches: dict[str, list[int]] = {}
    missing_probability_steps: list[int] = []
    probability_mismatches: list[int] = []
    gradient_mismatches: list[int] = []
    learning_rate_mismatches: list[int] = []
    for index, (vendor, package) in enumerate(
        zip(vendor_rows, package_rows, strict=True)
    ):
        assert vendor["step"] == package["step"] == index
        assert vendor["sample_ids"] == package["sample_ids"]
        for field in comparison_fields:
            if vendor[field] != package[field]:
                digest_mismatches.setdefault(field, []).append(index)
        vendor_probability = vendor["importance_probability_digest"]
        package_probability = package["importance_probability_digest"]
        if not vendor_probability or not package_probability:
            missing_probability_steps.append(index)
        elif vendor_probability != package_probability:
            probability_mismatches.append(index)
        if vendor["gradient_norm"] != package["gradient_norm"]:
            gradient_mismatches.append(index)
        if vendor["learning_rate"] != package["learning_rate"]:
            learning_rate_mismatches.append(index)
    losses = [
        abs(vendor["loss"] - package["loss"])
        for vendor, package in zip(vendor_rows, package_rows, strict=True)
    ]
    relative_losses = [
        absolute / max(abs(vendor["loss"]), 1.0e-12)
        for absolute, vendor in zip(losses, vendor_rows, strict=True)
    ]
    first = next(
        (
            {
                "step": index,
                "tensor": "train_loss",
                "max_abs": absolute,
                "relative": relative,
            }
            for index, (absolute, relative) in enumerate(
                zip(losses, relative_losses, strict=True)
            )
            if relative != 0.0
        ),
        None,
    )
    parameter_mismatches = [
        index
        for index, (vendor, package) in enumerate(
            zip(vendor_rows, package_rows, strict=True)
        )
        if vendor["parameter_state_digest"] != package["parameter_state_digest"]
    ]
    comparison = {
        "max_abs_loss_diff": max(losses),
        "max_relative_loss_diff": max(relative_losses),
        "max_abs_gradient_norm_diff": max(
            abs(vendor["gradient_norm"] - package["gradient_norm"])
            for vendor, package in zip(vendor_rows, package_rows, strict=True)
        ),
        "max_abs_learning_rate_diff": max(
            abs(vendor["learning_rate"] - package["learning_rate"])
            for vendor, package in zip(vendor_rows, package_rows, strict=True)
        ),
        "parameter_state_hash_mismatch_steps": parameter_mismatches,
        "digest_mismatch_steps": digest_mismatches,
        "importance_probability_missing_steps": missing_probability_steps,
        "importance_probability_hash_mismatch_steps": probability_mismatches,
        "scheduler_state_equal": scheduler_state_equal,
        "scheduler_state_best_abs_diff": scheduler_best_abs_diff,
        "scheduler_state_tolerance": VALIDATION_LOSS_TOLERANCE,
        "validation_executed": package_record["validation_executed"],
        "package_validation_loss": package_record["validation_loss"],
        "vendor_validation_loss": vendor_record["validation_loss"],
        "validation_loss_abs_diff": validation_loss_abs_diff,
        "validation_loss_tolerance": VALIDATION_LOSS_TOLERANCE,
        "gradient_norm_mismatch_steps": gradient_mismatches,
        "learning_rate_mismatch_steps": learning_rate_mismatches,
        "first_divergence": first,
        "relative_loss_criterion_passed": first is None,
    }
    assert not digest_mismatches, digest_mismatches
    assert not missing_probability_steps, missing_probability_steps
    assert not probability_mismatches, probability_mismatches
    assert not gradient_mismatches, gradient_mismatches
    assert not learning_rate_mismatches, learning_rate_mismatches
    assert not parameter_mismatches, parameter_mismatches
    assert comparison["relative_loss_criterion_passed"], comparison
    return comparison


def _repeat_envelope(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    first_record: dict[str, Any] | None = None,
    second_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loss_diffs = [
        abs(left["loss"] - right["loss"])
        for left, right in zip(first, second, strict=True)
    ]
    envelope = {
        "max_abs_loss_diff": max(loss_diffs),
        "gradient_norm_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["gradient_norm"] != right["gradient_norm"]
        ],
        "learning_rate_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["learning_rate"] != right["learning_rate"]
        ],
        "input_digest_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["input_ids_digest"] != right["input_ids_digest"]
            or left["attention_mask_digest"] != right["attention_mask_digest"]
        ],
        "trace_digest_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if any(
                left[field] != right[field]
                for field in (
                    "timesteps_digest",
                    "corrupted_tokens_digest",
                    "reconstructed_tokens_digest",
                    "rng_digest",
                )
            )
        ],
        "parameter_state_hash_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["parameter_state_digest"] != right["parameter_state_digest"]
        ],
        "optimizer_state_hash_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["optimizer_state_digest"] != right["optimizer_state_digest"]
        ],
        "importance_probability_hash_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if not left["importance_probability_digest"]
            or not right["importance_probability_digest"]
            or left["importance_probability_digest"]
            != right["importance_probability_digest"]
        ],
        "sample_id_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["sample_ids"] != right["sample_ids"]
        ],
        "loss_component_mismatch_steps": [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if left["bce_loss_digest"] != right["bce_loss_digest"]
            or left["weighted_bce_loss_digest"] != right["weighted_bce_loss_digest"]
        ],
    }
    if first_record is not None and second_record is not None:
        envelope["scheduler_state_digest_equal"] = (
            first_record["scheduler_state_digest"]
            == second_record["scheduler_state_digest"]
        )
        envelope["scheduler_state_equal"] = (
            first_record["scheduler_state"] == second_record["scheduler_state"]
        )
        envelope["validation_state_equal"] = all(
            first_record[field] == second_record[field]
            for field in (
                "validation_executed",
                "validation_batches",
                "validation_loss",
            )
        )
        envelope["pre_model_rng_digest_equal"] = (
            first_record["pre_model_rng_digest"]
            == second_record["pre_model_rng_digest"]
        )
        envelope["pre_loader_rng_digest_equal"] = (
            first_record["pre_loader_rng_digest"]
            == second_record["pre_loader_rng_digest"]
        )
    return envelope


@pytest.mark.parametrize("dataset", DATASETS)
def test_s3_natural_lockstep_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = _s3_steps()
    _apply_s3_determinism()
    fixture = _fixture(dataset, device, seed=NATURAL_STREAM_SEED)
    (
        vendor_batches,
        package_batches,
        vendor_validation_batches,
        package_validation_batches,
        loader_record,
    ) = _natural_training_batches(dataset, fixture.vendor_tokenizer, steps)
    initial_state = deepcopy(fixture.vendor.model.module.state_dict())
    vendor_rows, vendor_state, vendor_record = _run_natural_side(
        fixture,
        vendor_batches,
        vendor_validation_batches,
        side="vendor",
        initial_state=initial_state,
        steps=steps,
    )
    vendor_repeat_rows, _, vendor_repeat_record = _run_natural_side(
        fixture,
        vendor_batches,
        vendor_validation_batches,
        side="vendor",
        initial_state=initial_state,
        steps=steps,
    )
    package_rows, package_state, package_record = _run_natural_side(
        fixture,
        package_batches,
        package_validation_batches,
        side="package",
        initial_state=initial_state,
        steps=steps,
    )
    package_repeat_rows, _, package_repeat_record = _run_natural_side(
        fixture,
        package_batches,
        package_validation_batches,
        side="package",
        initial_state=initial_state,
        steps=steps,
    )
    vendor_record["pre_loader_rng_digest"] = loader_record[
        "vendor_pre_loader_rng_digest"
    ]
    vendor_repeat_record["pre_loader_rng_digest"] = loader_record[
        "vendor_pre_loader_rng_digest"
    ]
    if package_repeat_record is None:
        raise AssertionError(
            "natural evidence did not use the expected production path"
        )
    comparison = _natural_comparison(
        dataset,
        vendor_rows,
        package_rows,
        vendor_record,
        package_record,
        loader_record,
    )
    final_parameter_diffs, final_parameter_first = _state_diffs(
        vendor_state, package_state
    )
    rows = [
        {
            "step": vendor["step"],
            "vendor_loss": vendor["loss"],
            "package_loss": package["loss"],
            "loss_abs_diff": abs(vendor["loss"] - package["loss"]),
            "loss_relative_diff": abs(vendor["loss"] - package["loss"])
            / max(abs(vendor["loss"]), 1.0e-12),
            "vendor_gradient_norm": vendor["gradient_norm"],
            "package_gradient_norm": package["gradient_norm"],
            "vendor_bce_loss_digest": vendor["bce_loss_digest"],
            "package_bce_loss_digest": package["bce_loss_digest"],
            "vendor_weighted_bce_loss_digest": vendor["weighted_bce_loss_digest"],
            "package_weighted_bce_loss_digest": package["weighted_bce_loss_digest"],
            "learning_rate_vendor": vendor["learning_rate"],
            "learning_rate_package": package["learning_rate"],
            "vendor_optimizer_state_digest": vendor["optimizer_state_digest"],
            "package_optimizer_state_digest": package["optimizer_state_digest"],
            "vendor_parameter_state_digest": vendor["parameter_state_digest"],
            "package_parameter_state_digest": package["parameter_state_digest"],
            "vendor_rng_digest": vendor["rng_digest"],
            "package_rng_digest": package["rng_digest"],
            "vendor_timesteps_digest": vendor["timesteps_digest"],
            "package_timesteps_digest": package["timesteps_digest"],
            "vendor_importance_probability_digest": vendor[
                "importance_probability_digest"
            ],
            "package_importance_probability_digest": package[
                "importance_probability_digest"
            ],
            "vendor_sample_ids": vendor["sample_ids"],
            "package_sample_ids": package["sample_ids"],
        }
        for vendor, package in zip(vendor_rows, package_rows, strict=True)
    ]
    stage = _s3_stage("s3-lockstep")
    trace_path = _write_jsonl(stage, dataset, rows)
    path = _write_json(
        stage,
        dataset,
        {
            "dataset": dataset,
            "steps": len(rows),
            "adamw_mode": os.environ.get("LAYOUT_CORRECTOR_ADAMW_MODE", "default"),
            "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "comparison": comparison,
            "final_parameter_max_abs_diffs": final_parameter_diffs,
            "final_parameter_first_difference": final_parameter_first,
            "repeat_run_envelope": {
                "vendor": _repeat_envelope(
                    vendor_rows,
                    vendor_repeat_rows,
                    vendor_record,
                    vendor_repeat_record,
                ),
                "package": _repeat_envelope(
                    package_rows,
                    package_repeat_rows,
                    package_record,
                    package_repeat_record,
                ),
            },
            "production_record": package_record,
            "vendor_record": vendor_record,
            "loader_record": loader_record,
            "trace_artifact": str(trace_path.relative_to(ROOT)),
            "runtime": _runtime_record(),
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()
    assert final_parameter_first is None, final_parameter_diffs
    assert max(final_parameter_diffs.values()) == 0.0, final_parameter_diffs
    vendor_repeat = _repeat_envelope(
        vendor_rows, vendor_repeat_rows, vendor_record, vendor_repeat_record
    )
    package_repeat = _repeat_envelope(
        package_rows, package_repeat_rows, package_record, package_repeat_record
    )
    assert vendor_repeat["max_abs_loss_diff"] == 0.0, vendor_repeat
    assert package_repeat["max_abs_loss_diff"] == 0.0, package_repeat
    assert not vendor_repeat["parameter_state_hash_mismatch_steps"], vendor_repeat
    assert not package_repeat["parameter_state_hash_mismatch_steps"], package_repeat
    assert not vendor_repeat["optimizer_state_hash_mismatch_steps"], vendor_repeat
    assert not package_repeat["optimizer_state_hash_mismatch_steps"], package_repeat
    assert not vendor_repeat["gradient_norm_mismatch_steps"], vendor_repeat
    assert not package_repeat["gradient_norm_mismatch_steps"], package_repeat
    assert not vendor_repeat["learning_rate_mismatch_steps"], vendor_repeat
    assert not package_repeat["learning_rate_mismatch_steps"], package_repeat
    assert not vendor_repeat["input_digest_mismatch_steps"], vendor_repeat
    assert not package_repeat["input_digest_mismatch_steps"], package_repeat
    assert not vendor_repeat["trace_digest_mismatch_steps"], vendor_repeat
    assert not package_repeat["trace_digest_mismatch_steps"], package_repeat
    assert not vendor_repeat["sample_id_mismatch_steps"], vendor_repeat
    assert not package_repeat["sample_id_mismatch_steps"], package_repeat
    assert not vendor_repeat["loss_component_mismatch_steps"], vendor_repeat
    assert not package_repeat["loss_component_mismatch_steps"], package_repeat
    assert not vendor_repeat["importance_probability_hash_mismatch_steps"], (
        vendor_repeat
    )
    assert vendor_repeat_record == vendor_record
    assert vendor_repeat["scheduler_state_digest_equal"] is True
    assert vendor_repeat["scheduler_state_equal"] is True
    assert vendor_repeat["validation_state_equal"] is True
    assert vendor_repeat["pre_model_rng_digest_equal"] is True
    assert vendor_repeat["pre_loader_rng_digest_equal"] is True
    assert not package_repeat["importance_probability_hash_mismatch_steps"], (
        package_repeat
    )
    assert package_repeat["scheduler_state_digest_equal"] is True
    assert package_repeat["scheduler_state_equal"] is True
    assert package_repeat["validation_state_equal"] is True
    assert package_repeat["pre_model_rng_digest_equal"] is True
    assert package_repeat["pre_loader_rng_digest_equal"] is True


@pytest.mark.parametrize("dataset", DATASETS)
def test_s3_synchronized_diagnostic_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    natural_path = (
        _evidence_root() / _s3_stage("s3-lockstep") / dataset / "summary.json"
    )
    if not natural_path.is_file():
        raise AssertionError(f"natural evidence is missing: {natural_path}")
    trace_path = natural_path.parent / "trace.jsonl"
    if not trace_path.is_file():
        raise AssertionError(f"natural trace is missing: {trace_path}")
    raw_rows = [
        cast(dict[str, Any], json.loads(line))
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(raw_rows) == _s3_steps()
    raw_fields = {
        "loss": ("vendor_loss", "package_loss"),
        "gradient_norm": ("vendor_gradient_norm", "package_gradient_norm"),
        "learning_rate": ("learning_rate_vendor", "learning_rate_package"),
        "optimizer_state_digest": (
            "vendor_optimizer_state_digest",
            "package_optimizer_state_digest",
        ),
        "parameter_state_digest": (
            "vendor_parameter_state_digest",
            "package_parameter_state_digest",
        ),
        "rng_digest": ("vendor_rng_digest", "package_rng_digest"),
        "timesteps_digest": (
            "vendor_timesteps_digest",
            "package_timesteps_digest",
        ),
        "importance_probability_digest": (
            "vendor_importance_probability_digest",
            "package_importance_probability_digest",
        ),
        "bce_loss_digest": ("vendor_bce_loss_digest", "package_bce_loss_digest"),
        "weighted_bce_loss_digest": (
            "vendor_weighted_bce_loss_digest",
            "package_weighted_bce_loss_digest",
        ),
    }
    raw_mismatches = {
        field: [
            index for index, row in enumerate(raw_rows) if row[pair[0]] != row[pair[1]]
        ]
        for field, pair in raw_fields.items()
    }
    assert not any(raw_mismatches.values()), raw_mismatches
    assert all(
        row["vendor_importance_probability_digest"]
        and row["package_importance_probability_digest"]
        for row in raw_rows
    )
    assert all(
        row["vendor_sample_ids"] == row["package_sample_ids"] for row in raw_rows
    )
    path = _write_json(
        "s3-lockstep-synchronized",
        dataset,
        {
            "dataset": dataset,
            "status": "not-needed",
            "basis": "natural production-path comparison is bitwise across raw trace rows",
            "natural_evidence": str(natural_path.relative_to(ROOT)),
            "trace_artifact": str(trace_path.relative_to(ROOT)),
            "raw_trace_rows": len(raw_rows),
            "raw_trace_mismatches": raw_mismatches,
            "runtime": _runtime_record(),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()


@pytest.mark.parametrize("dataset", DATASETS)
def test_parameter_registration_order_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=123, deterministic_algorithms=False))
    fixture = _fixture(dataset, torch.device("cpu"))
    vendor_names = [name for name, _ in fixture.vendor.model.module.named_parameters()]
    package_names = [name for name, _ in fixture.package.model.model.named_parameters()]
    assert package_names == vendor_names


def _loader_stream_rows(
    dataset: LayoutCorrectorTrainingDatasetName,
    split: LayoutCorrectorTrainingSplit,
    vendor_split: str,
    fixture: Fixture,
    *,
    random_order: bool,
    evidence_num_workers: int,
) -> list[dict[str, Any]]:
    from trainer.data.util import compose_transform, sparse_to_dense

    vendor_dataset = _vendor_dataset(
        dataset,
        vendor_split,
        transform=compose_transform(["RandomOrder"]) if random_order else None,
    )
    package_module = LayoutCorrectorDataModule(
        dataset_name=dataset,
        config=_package_layout_dm_config(dataset),
        processed_data_dir=_layout_dm_cache() / "datasets",
        batch_size=8,
        num_workers=evidence_num_workers,
        random_order=random_order,
        pin_memory=False,
    )
    package_module.setup("fit" if split in {"train", "validation"} else "test")
    package_dataset = package_module._datasets[cast(Any, split)]
    vendor_loader = GeometricDataLoader(
        vendor_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=evidence_num_workers,
    )
    package_loader = DataLoader(
        package_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=evidence_num_workers,
    )
    seed = 314159
    torch.manual_seed(seed)
    vendor_iterator = iter(vendor_loader)
    vendor_batches: list[Any] = []
    for _ in range(2):
        try:
            vendor_batches.append(next(vendor_iterator))
        except StopIteration:
            break
    torch.manual_seed(seed)
    package_iterator = iter(package_loader)
    package_batches: list[Any] = []
    for _ in range(2):
        try:
            package_batches.append(next(package_iterator))
        except StopIteration:
            break
    rows: list[dict[str, Any]] = []
    for index, (vendor_batch, package_batch) in enumerate(
        zip(vendor_batches, package_batches, strict=True)
    ):
        bbox, labels, _, mask = sparse_to_dense(vendor_batch)
        vendor_encoded = fixture.vendor_tokenizer.encode(
            {"bbox": bbox, "label": labels, "mask": mask}
        )
        package_input_ids = cast(torch.Tensor, package_batch["input_ids"])
        package_mask = cast(torch.Tensor, package_batch["attention_mask"])
        token_diff = (vendor_encoded["seq"] - package_input_ids).abs()
        mask_diff = (vendor_encoded["mask"] != package_mask).sum().item()
        sample_names = vendor_batch.attr["name"]
        rows.append(
            {
                "batch_index": index,
                "sample_names": [str(name) for name in sample_names],
                "sample_count": int(package_input_ids.shape[0]),
                "vendor_input_digest": _tensor_digest(vendor_encoded["seq"]),
                "package_input_digest": _tensor_digest(package_input_ids),
                "max_abs_input_id_diff": int(token_diff.max().item()),
                "attention_mask_mismatch_count": int(mask_diff),
                "vendor_loader_num_workers": vendor_loader.num_workers,
                "package_loader_num_workers": package_loader.num_workers,
            }
        )
        assert torch.equal(vendor_encoded["seq"], package_input_ids)
        assert torch.equal(vendor_encoded["mask"], package_mask)
    return rows


@pytest.mark.parametrize("dataset", DATASETS)
def test_s4_loader_stream_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=314159, deterministic_algorithms=False))
    fixture = _fixture(dataset, torch.device("cpu"))
    split_rows = {
        split: _loader_stream_rows(
            dataset,
            split,
            "val" if split == "validation" else split,
            fixture,
            random_order=split == "train",
            evidence_num_workers=_loader_worker_count(dataset),
        )
        for split in ("train", "validation", "test")
    }
    observed_loader_num_workers = sorted(
        {
            row["vendor_loader_num_workers"]
            for rows in split_rows.values()
            for row in rows
        }
        | {
            row["package_loader_num_workers"]
            for rows in split_rows.values()
            for row in rows
        }
    )
    batch_stream_comparison = all(
        row["max_abs_input_id_diff"] == 0 and row["attention_mask_mismatch_count"] == 0
        for rows in split_rows.values()
        for row in rows
    )
    assert all(len(rows) == 2 for rows in split_rows.values())
    assert observed_loader_num_workers == [_loader_worker_count(dataset)]
    assert batch_stream_comparison
    path = _write_json(
        "loader-stream",
        dataset,
        {
            "dataset": dataset,
            "splits": split_rows,
            "observed_loader_num_workers": observed_loader_num_workers,
            "batch_stream_comparison": batch_stream_comparison,
            "train_transform": "RandomOrder",
            "validation_transform": "none",
            "test_transform": "none (vendor evaluation-time preprocessing)",
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
            "runtime": _runtime_record(),
        },
    )
    assert path.exists()


def _prediction_records(
    predictions: list[tuple[Any, Any]],
) -> list[dict[str, list[Any]]]:
    return [
        {
            "bbox": torch.as_tensor(boxes).tolist(),
            "labels": torch.as_tensor(labels).tolist(),
        }
        for boxes, labels in predictions
    ]


def _out_of_bounds_count(predictions: list[dict[str, list[Any]]]) -> int:
    count = 0
    for prediction in predictions:
        boxes = torch.as_tensor(prediction["bbox"], dtype=torch.float32)
        if boxes.numel() == 0:
            continue
        corners = torch.cat(
            (boxes[:, :2] - boxes[:, 2:] / 2.0, boxes[:, :2] + boxes[:, 2:] / 2.0),
            dim=1,
        )
        count += int(torch.any((corners < 0.0) | (corners > 1.0), dim=1).sum())
    return count


def _prediction_difference(
    reference: list[dict[str, list[Any]]], target: list[dict[str, list[Any]]]
) -> tuple[float, dict[str, Any] | None]:
    maximum = 0.0
    first: dict[str, Any] | None = None
    for index, (left, right) in enumerate(zip(reference, target, strict=True)):
        left_boxes = torch.as_tensor(left["bbox"], dtype=torch.float32)
        right_boxes = torch.as_tensor(right["bbox"], dtype=torch.float32)
        if left_boxes.shape != right_boxes.shape:
            return float("inf"), {"layout": index, "tensor": "bbox"}
        difference = (left_boxes - right_boxes).abs()
        value = float(difference.max().item()) if difference.numel() else 0.0
        maximum = max(maximum, value)
        if first is None and value != 0.0:
            first = {"layout": index, "tensor": "bbox", "max_abs": value}
        if left["labels"] != right["labels"] and first is None:
            first = {"layout": index, "tensor": "labels"}
    return maximum, first


def _evaluation_asset_root() -> Path:
    value = os.environ.get("LAYOUT_CORRECTOR_EVAL_ASSET_ROOT")
    if value is None:
        skip_or_fail_vendor_parity(
            "full evaluation assets are local-only",
            missing_paths=[Path("$LAYOUT_CORRECTOR_EVAL_ASSET_ROOT")],
            regeneration_hint="set LAYOUT_CORRECTOR_EVAL_ASSET_ROOT to the extracted starter-kit download directory",
        )

    assert value is not None
    return Path(value)


def _evaluation_pipeline_root(dataset: str) -> Path:
    root = os.environ.get("LAYOUT_CORRECTOR_EVAL_PIPELINE_ROOT")
    if root is None:
        skip_or_fail_vendor_parity(
            "converted evaluation pipelines are local-only",
            missing_paths=[Path("$LAYOUT_CORRECTOR_EVAL_PIPELINE_ROOT")],
            regeneration_hint="convert the released seed-0 LayoutDM and Layout-Corrector checkpoints",
        )

    assert root is not None
    path = Path(root) / dataset
    if not path.is_dir():
        skip_or_fail_vendor_parity(
            "converted evaluation pipeline is missing",
            missing_paths=[path],
            regeneration_hint="run the conversion commands in TRAINING.md",
        )
    return path


def _evaluation_environment(scratch: Path) -> dict[str, str]:
    parent_hash_seed = os.environ.get("PYTHONHASHSEED")
    if parent_hash_seed != "0":
        raise RuntimeError(
            "S4 evaluator requires the parent process to set PYTHONHASHSEED=0"
        )

    environment = os.environ.copy()
    runtime = os.environ.get("LAYOUT_CORRECTOR_AUDIT_VENV")
    if runtime is not None:
        environment["PATH"] = f"{runtime}/bin:{environment['PATH']}"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(scratch), str(scratch / "src" / "trainer"))
    )
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONHASHSEED"] = parent_hash_seed
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return environment


def _vendor_command(dataset: str, job_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "vendor" / "layout-corrector" / "bin" / "corrector_test_eval.py"),
        str(job_dir),
        dataset,
        "--device",
        "0",
        "--batch_size",
        str(EVALUATION_BATCH_SIZE),
        "--test_only",
        "--timesteps",
        "100",
        "--corrector_t_list",
        *(str(value) for value in EVALUATION_CORRECTOR_T_LIST),
        "--force",
        "--cond_list",
        *EVALUATION_CONDITIONS,
        "--no_gumbel_noise",
    ]


def _vendor_unconditional_sample_count() -> int:
    """Read the unconditional count from the original evaluator entry point."""
    path = ROOT / "vendor" / "layout-corrector" / "bin" / "corrector_test_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        option_names = {
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        if "--num_uncond_samples" not in option_names:
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, int
            ):
                return keyword.value.value
    raise AssertionError("original evaluator has no integer unconditional count")


def _vendor_result_paths(dataset: str, scratch: Path) -> dict[str, Path]:
    result_root = scratch / "results" / dataset / "layout_corrector"
    result_dirs = {
        condition: sorted(result_root.glob(f"{condition}_*/"))[-1]
        for condition in EVALUATION_CONDITIONS
    }
    return {
        condition: next(result_dirs[condition].glob("seed_0.pkl"))
        for condition in result_dirs
    }


def _retained_vendor_scratch(dataset: str) -> Path | None:
    if os.environ.get("LAYOUT_CORRECTOR_S4_REUSE_VENDOR") != "1":
        return None
    template = os.environ.get("LAYOUT_CORRECTOR_S4_VENDOR_SCRATCH_ROOT")
    if template is None:
        raise RuntimeError(
            "LAYOUT_CORRECTOR_S4_VENDOR_SCRATCH_ROOT is required when "
            "LAYOUT_CORRECTOR_S4_REUSE_VENDOR=1"
        )
    return Path(template.format(dataset=dataset))


def _vendor_import_roots(scratch: Path) -> tuple[Path, ...]:
    return (scratch, scratch / "src", scratch / "src" / "trainer")


def _prepend_vendor_import_roots(scratch: Path) -> None:
    for import_root in reversed(_vendor_import_roots(scratch)):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))


def _load_vendor_metadata(vendor_path: Path, scratch: Path) -> dict[str, Any]:
    _prepend_vendor_import_roots(scratch)
    with vendor_path.open("rb") as handle:
        return cast(dict[str, Any], pickle.load(handle))


def test_retained_vendor_output_unpickles() -> None:
    scratch = _retained_vendor_scratch("rico25")
    if scratch is None:
        pytest.skip("retained vendor-output reuse is not enabled")
    vendor_path = _vendor_result_paths("rico25", scratch)["unconditional"]
    metadata = _load_vendor_metadata(vendor_path, scratch)
    assert len(metadata["results"]) == int(metadata["N_total"])


def _run_vendor_evaluation(
    dataset: str,
) -> tuple[Path, list[str], dict[str, Path], Path, bool]:
    asset_root = _evaluation_asset_root()
    job_dir = asset_root / "pretrained_weights" / dataset / "layout_corrector" / "0"
    command = _vendor_command(dataset, job_dir)
    retained_scratch = _retained_vendor_scratch(dataset)
    if retained_scratch is not None:
        pkl_paths = _vendor_result_paths(dataset, retained_scratch)
        for path in pkl_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "vendor" / "layout-corrector" / "bin" / "calc_metrics.py"),
                str(retained_scratch / "results" / dataset / "layout_corrector"),
                "--force",
            ],
            cwd=retained_scratch,
            env=_evaluation_environment(retained_scratch),
            check=True,
        )
        return retained_scratch, command, pkl_paths, job_dir, True

    scratch = Path(tempfile.mkdtemp(prefix=f"layout-corrector-evaluation-{dataset}-"))
    shutil.copytree(
        ROOT / "vendor" / "layout-corrector" / "src",
        scratch / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(ROOT / "vendor" / "layout-corrector" / "bin", scratch / "bin")
    shutil.copy2(ROOT / "vendor" / "layout-corrector" / "eval.py", scratch / "eval.py")
    (scratch / "download").symlink_to(asset_root, target_is_directory=True)
    if not (job_dir / "best_model.pt").is_file():
        skip_or_fail_vendor_parity(
            "released corrector checkpoint is missing",
            missing_paths=[job_dir / "best_model.pt"],
            regeneration_hint="download the Layout-Corrector starter kit",
        )
    subprocess.run(
        command,
        cwd=scratch,
        env=_evaluation_environment(scratch),
        check=True,
    )
    return scratch, command, _vendor_result_paths(dataset, scratch), job_dir, False


def _numeric_metrics(path: Path) -> dict[str, float]:
    values = json.loads(path.read_text())
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def _corrector_weight_identity(
    pipeline: Any, vendor_checkpoint: Path
) -> dict[str, Any]:
    from layout_corrector.conversion import load_original_corrector_state_dict

    vendor_state = load_original_corrector_state_dict(vendor_checkpoint)
    package_state = {
        name: value.detach().cpu()
        for name, value in pipeline.corrector.model.state_dict().items()
    }
    vendor_keys = set(vendor_state)
    package_keys = set(package_state)
    missing = sorted(vendor_keys - package_keys)
    extra = sorted(package_keys - vendor_keys)
    tensor_mismatches = [
        name
        for name in sorted(vendor_keys & package_keys)
        if not torch.equal(vendor_state[name].cpu(), package_state[name])
    ]
    return {
        "same_key_set": not missing and not extra,
        "missing_from_package": missing,
        "extra_in_package": extra,
        "tensor_mismatches": tensor_mismatches,
        "vendor_state_digest": _state_dict_digest(vendor_state),
        "package_state_digest": _state_dict_digest(package_state),
        "same_tensors": not tensor_mismatches,
    }


def _assert_metric_parity(
    dataset: str,
    condition: str,
    vendor_metrics: dict[str, float],
    package_metrics: dict[str, float],
) -> dict[str, float]:
    runtime_keys = {"N_total", "seed", "t_total"}
    vendor_keys = set(vendor_metrics) - runtime_keys
    package_keys = set(package_metrics) - runtime_keys
    assert vendor_keys == package_keys
    metric_diffs = {
        key: abs(vendor_metrics[key] - package_metrics[key])
        for key in sorted(vendor_keys)
    }
    assert not any(metric_diffs.values()), {
        "dataset": dataset,
        "condition": condition,
        "metric_diffs": metric_diffs,
        "vendor": vendor_metrics,
        "package": package_metrics,
    }
    return metric_diffs


def _package_evaluation(
    dataset: str,
    pipeline_path: Path,
    vendor_pkl_paths: dict[str, Path],
    scratch: Path,
) -> tuple[dict[str, Path], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from layout_corrector import LayoutCorrectorPipeline
    from layout_dm.training.dataset import LayoutDMProcessedDataset

    device = torch.device("cuda:0")
    pipeline = LayoutCorrectorPipeline.from_pretrained(pipeline_path)
    weight_identity = _corrector_weight_identity(
        pipeline,
        _evaluation_asset_root()
        / "pretrained_weights"
        / dataset
        / "layout_corrector"
        / "0"
        / "best_model.pt",
    )
    assert weight_identity["same_key_set"], weight_identity
    assert weight_identity["same_tensors"], weight_identity
    pipeline.to(device)
    processed_root = _evaluation_asset_root() / "datasets"
    dataset_config = pipeline.layout_dm.tokenizer.config
    package_dataset = LayoutDMProcessedDataset(
        dataset_name=cast(LayoutDMTrainingDatasetName, dataset),
        config=dataset_config,
        processed_data_dir=processed_root,
        split="test",
        max_seq_length=25,
        random_order=False,
    )
    loader = DataLoader(
        package_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    package_root = scratch / "results" / dataset
    package_dirs = {
        condition: package_root / f"package_{condition}"
        for condition in vendor_pkl_paths
    }
    package_inputs: dict[str, list[dict[str, Any]]] = {}
    expected_unconditional_count = _vendor_unconditional_sample_count()
    for condition, vendor_path in vendor_pkl_paths.items():
        vendor_meta = _load_vendor_metadata(vendor_path, scratch)
        generator = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
        package_predictions: list[tuple[torch.Tensor, torch.Tensor]] = []
        inputs: list[dict[str, Any]] = []
        started = time.perf_counter()
        if condition == "unconditional":
            vendor_total_layouts = int(vendor_meta["N_total"])
            assert vendor_total_layouts == expected_unconditional_count, {
                "dataset": dataset,
                "vendor_total_layouts": vendor_total_layouts,
                "expected_unconditional_count": expected_unconditional_count,
            }
            total_layouts = expected_unconditional_count
            for start in range(0, total_layouts, EVALUATION_BATCH_SIZE):
                output = cast(
                    LayoutGenerationOutput,
                    pipeline(
                        batch_size=min(EVALUATION_BATCH_SIZE, total_layouts - start),
                        generator=generator,
                        num_inference_steps=100,
                        sampling="random",
                        corrector_steps=1,
                        corrector_t_list=EVALUATION_CORRECTOR_T_LIST,
                        corrector_start=-1,
                        corrector_end=-1,
                        corrector_mask_mode="thresh",
                        corrector_mask_threshold=0.7,
                        use_gumbel_noise=False,
                    ),
                )
                package_predictions.extend(
                    (
                        output.bbox[index][output.mask[index]].detach().cpu().numpy(),
                        output.labels[index][output.mask[index]].detach().cpu().numpy(),
                    )
                    for index in range(output.bbox.shape[0])
                )
        else:
            condition_type = "label" if condition == "c" else "label_size"
            for batch in loader:
                input_ids = cast(torch.Tensor, batch["input_ids"])
                inputs.append(
                    {
                        "input_ids": input_ids.detach().cpu(),
                        "attention_mask": cast(torch.Tensor, batch["attention_mask"])
                        .detach()
                        .cpu(),
                        "sample_ids": [str(value) for value in batch.get("id", [])],
                    }
                )
                decoded = pipeline.layout_dm.tokenizer.decode_layout(input_ids)
                output = cast(
                    LayoutGenerationOutput,
                    pipeline(
                        generator=generator,
                        condition_type=condition_type,
                        labels=decoded["labels"],
                        bbox=decoded["bbox"],
                        mask=decoded["mask"],
                        num_inference_steps=100,
                        sampling="random",
                        corrector_steps=1,
                        corrector_t_list=EVALUATION_CORRECTOR_T_LIST,
                        corrector_start=-1,
                        corrector_end=-1,
                        corrector_mask_mode="thresh",
                        corrector_mask_threshold=0.7,
                        use_gumbel_noise=False,
                    ),
                )
                package_predictions.extend(
                    (
                        output.bbox[index][output.mask[index]].detach().cpu().numpy(),
                        output.labels[index][output.mask[index]].detach().cpu().numpy(),
                    )
                    for index in range(output.bbox.shape[0])
                )
        package_dir = scratch / "results" / dataset / f"package_{condition}"
        package_dir.mkdir(parents=True, exist_ok=True)
        package_meta = dict(vendor_meta)
        package_meta["results"] = package_predictions
        package_meta["t_total"] = time.perf_counter() - started
        package_meta["N_total"] = len(package_predictions)
        with (package_dir / "seed_0.pkl").open("wb") as handle:
            pickle.dump(package_meta, handle)
        package_dirs[condition] = package_dir
        package_inputs[condition] = inputs
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "vendor" / "layout-corrector" / "bin" / "calc_metrics.py"),
            str(package_root),
            "--force",
        ],
        cwd=scratch,
        env=_evaluation_environment(scratch),
        check=True,
    )
    return package_dirs, package_inputs, weight_identity


def _vendor_evaluation_inputs(
    vendor_pkl_paths: dict[str, Path], scratch: Path
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct the conditioning tensors from the original evaluator path."""
    _prepend_vendor_import_roots(scratch)
    from hydra.utils import instantiate
    from trainer.data.util import sparse_to_dense
    from trainer.corrector_test import build_tokenizer

    vendor_meta = _load_vendor_metadata(vendor_pkl_paths["unconditional"], scratch)
    train_config = vendor_meta["train_cfg"]
    dataset_config = train_config.dataset
    dataset_config.dir = str(scratch / "download" / "datasets")
    dataset = instantiate(dataset_config)(split="test", transform=None)
    loader = GeometricDataLoader(
        dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    tokenizer = build_tokenizer(train_config.data, dataset_config)
    inputs: dict[str, list[dict[str, Any]]] = {
        "unconditional": [],
        "c": [],
        "cwh": [],
    }
    for batch in loader:
        sample_ids = [str(value) for value in batch.attr["name"]]
        bbox, label, _, mask = sparse_to_dense(batch)
        encoded = tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        for condition in ("c", "cwh"):
            inputs[condition].append(
                {
                    "input_ids": encoded["seq"].detach().cpu(),
                    "attention_mask": encoded["mask"].detach().cpu(),
                    "sample_ids": sample_ids,
                }
            )
    return inputs


def _input_bytes(records: list[dict[str, Any]]) -> bytes:
    parts: list[bytes] = []
    for record in records:
        input_ids = cast(torch.Tensor, record["input_ids"])
        attention_mask = cast(torch.Tensor, record["attention_mask"])
        parts.extend(
            (
                input_ids.contiguous().numpy().tobytes(),
                attention_mask.contiguous().numpy().tobytes(),
            )
        )
    return b"".join(parts)


def _write_input_artifact(
    evidence_dir: Path,
    condition: str,
    side: str,
    records: list[dict[str, Any]],
) -> Path:
    path = evidence_dir / f"{condition}-{side}-evaluation-inputs.bin"
    path.write_bytes(_input_bytes(records))
    return path


def _state_artifact_bytes(state: dict[str, torch.Tensor]) -> bytes:
    parts: list[bytes] = []
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        parts.extend(
            (
                name.encode(),
                b"\0",
                str(value.dtype).encode(),
                b"\0",
                repr(tuple(value.shape)).encode(),
                b"\0",
                value.view(torch.uint8).numpy().tobytes(),
            )
        )
    return b"".join(parts)


@pytest.mark.parametrize("dataset", DATASETS)
def test_s4_test_evaluation_path_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("S4 evaluation-path parity requires the selected GPU")
    expected_unconditional_count = _vendor_unconditional_sample_count()
    scratch, vendor_command, vendor_pkl_paths, corrector_checkpoint, vendor_reused = (
        _run_vendor_evaluation(dataset)
    )
    evidence_dir = _evidence_root() / "evaluation-path" / dataset
    evidence_dir.mkdir(parents=True, exist_ok=True)
    package_dirs, package_inputs, weight_identity = _package_evaluation(
        dataset,
        _evaluation_pipeline_root(dataset),
        vendor_pkl_paths,
        scratch,
    )
    vendor_inputs = _vendor_evaluation_inputs(vendor_pkl_paths, scratch)
    pipeline_path = _evaluation_pipeline_root(dataset)
    from layout_corrector import LayoutCorrectorPipeline
    from layout_corrector.conversion import load_original_corrector_state_dict

    package_pipeline = LayoutCorrectorPipeline.from_pretrained(pipeline_path)
    vendor_state = load_original_corrector_state_dict(
        corrector_checkpoint / "best_model.pt"
    )
    package_state = {
        name: value.detach().cpu()
        for name, value in package_pipeline.corrector.model.state_dict().items()
    }
    vendor_weight_path = evidence_dir / "vendor-corrector-weights.bin"
    package_weight_path = evidence_dir / "package-corrector-weights.bin"
    vendor_weight_path.write_bytes(_state_artifact_bytes(vendor_state))
    package_weight_path.write_bytes(_state_artifact_bytes(package_state))
    assert vendor_weight_path.read_bytes() == package_weight_path.read_bytes()
    weight_identity["vendor_artifact_sha256"] = _sha256(vendor_weight_path)
    weight_identity["package_artifact_sha256"] = _sha256(package_weight_path)
    weight_identity["artifact_equal"] = (
        vendor_weight_path.read_bytes() == package_weight_path.read_bytes()
    )
    assert weight_identity["artifact_equal"] is True
    vendor_sweep_source_commit = os.environ.get(
        "LAYOUT_CORRECTOR_S4_VENDOR_SWEEP_COMMIT"
    )
    if vendor_reused and vendor_sweep_source_commit is None:
        raise RuntimeError(
            "LAYOUT_CORRECTOR_S4_VENDOR_SWEEP_COMMIT is required when reusing "
            "vendor outputs"
        )
    evaluator_environment = _evaluation_environment(scratch)
    results: dict[str, Any] = {
        "dataset": dataset,
        "split": "test",
        "conditions": {},
        "corrector_checkpoint": str(corrector_checkpoint),
        "corrector_checkpoint_sha256": _sha256(corrector_checkpoint / "best_model.pt"),
        "corrector_weight_identity": weight_identity,
        "pipeline_path": str(pipeline_path),
        "vendor_command": vendor_command,
        "vendor_evaluation_mode": "reused" if vendor_reused else "executed",
        "package_evaluation_mode": "executed",
        "vendor_sweep_source_commit": vendor_sweep_source_commit
        if vendor_reused
        else _source_commit(ROOT),
        "vendor_result_root": str(scratch / "results" / dataset / "layout_corrector"),
        "vendor_source_commit": _source_commit(ROOT / "vendor" / "layout-corrector"),
        "vendor_evaluator_commit": _source_commit(ROOT / "vendor" / "layout-corrector"),
        "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
        "evaluator_unconditional_sample_count": expected_unconditional_count,
        "evaluator_environment": {
            "PYTHONHASHSEED": evaluator_environment["PYTHONHASHSEED"]
        },
        "source_commit": _source_commit(ROOT),
        "runtime": _runtime_record(),
    }
    if vendor_reused:
        assert vendor_sweep_source_commit == "5ee3af2", vendor_sweep_source_commit
        results["retained_vendor_provenance"] = {
            "sweep_source_commit": vendor_sweep_source_commit,
            "vendor_result_sha256": {
                condition: _sha256(path) for condition, path in vendor_pkl_paths.items()
            },
            "vendor_weight_sha256": _sha256(corrector_checkpoint / "best_model.pt"),
        }
    for condition, vendor_path in vendor_pkl_paths.items():
        vendor_meta = _load_vendor_metadata(vendor_path, scratch)
        with (package_dirs[condition] / "seed_0.pkl").open("rb") as handle:
            package_meta = pickle.load(handle)
        vendor_records = _prediction_records(vendor_meta["results"])
        package_records = _prediction_records(package_meta["results"])
        vendor_prediction_path = evidence_dir / f"{condition}-vendor-predictions.json"
        package_prediction_path = evidence_dir / f"{condition}-package-predictions.json"
        vendor_prediction_path.write_text(
            json.dumps(vendor_records, sort_keys=True, separators=(",", ":")) + "\n"
        )
        package_prediction_path.write_text(
            json.dumps(package_records, sort_keys=True, separators=(",", ":")) + "\n"
        )
        vendor_metrics = _numeric_metrics(
            vendor_path.parent / "scores_fake_seed_0.json"
        )
        package_metrics = _numeric_metrics(
            package_dirs[condition] / "scores_fake_seed_0.json"
        )
        metric_diffs = _assert_metric_parity(
            dataset, condition, vendor_metrics, package_metrics
        )
        prediction_max_abs_diff, prediction_first = _prediction_difference(
            vendor_records, package_records
        )
        input_path = evidence_dir / f"{condition}-input-ids.bin"
        package_input_records = package_inputs[condition]
        vendor_input_records = vendor_inputs[condition]
        vendor_input_path = _write_input_artifact(
            evidence_dir, condition, "vendor", vendor_input_records
        )
        package_input_path = _write_input_artifact(
            evidence_dir, condition, "package", package_input_records
        )
        assert vendor_input_path.read_bytes() == package_input_path.read_bytes()
        input_path.write_bytes(_input_bytes(package_input_records))
        assert input_path.read_bytes() == package_input_path.read_bytes()
        assert len(vendor_input_records) == len(package_input_records)
        for vendor_input, package_input in zip(
            vendor_input_records, package_input_records, strict=True
        ):
            assert torch.equal(vendor_input["input_ids"], package_input["input_ids"])
            assert torch.equal(
                vendor_input["attention_mask"], package_input["attention_mask"]
            )
            assert vendor_input["sample_ids"] == package_input["sample_ids"]
        vendor_counts = [len(item["bbox"]) for item in vendor_records]
        package_counts = [len(item["bbox"]) for item in package_records]
        assert len(vendor_records) == len(package_records)
        assert vendor_counts == package_counts
        assert vendor_records == package_records
        assert prediction_max_abs_diff == 0.0
        assert prediction_first is None
        assert (
            vendor_prediction_path.read_bytes() == package_prediction_path.read_bytes()
        )
        assert _sha256(vendor_prediction_path) == _sha256(package_prediction_path)
        assert _out_of_bounds_count(vendor_records) == _out_of_bounds_count(
            package_records
        )
        if condition == "unconditional":
            assert int(vendor_meta["N_total"]) == expected_unconditional_count
            assert int(package_meta["N_total"]) == expected_unconditional_count
        assert vendor_meta["N_total"] == package_meta["N_total"] == len(vendor_records)
        assert weight_identity["same_key_set"] is True
        assert weight_identity["same_tensors"] is True
        assert weight_identity["artifact_equal"] is True
        results["conditions"][condition] = {
            "command_arguments": vendor_command[2:],
            "sampling_seed": EVALUATION_SEED,
            "corrector_t_list": list(EVALUATION_CORRECTOR_T_LIST),
            "vendor_results_pickle": str(vendor_path),
            "vendor_results_sha256": _sha256(vendor_path),
            "package_results_pickle": str(package_dirs[condition] / "seed_0.pkl"),
            "vendor_input_artifact": str(vendor_input_path.relative_to(ROOT)),
            "package_input_artifact": str(package_input_path.relative_to(ROOT)),
            "vendor_input_sha256": _sha256(vendor_input_path),
            "package_input_sha256": _sha256(package_input_path),
            "input_artifact": str(input_path.relative_to(ROOT)),
            "input_sha256": _sha256(input_path),
            "input_batch_count": len(vendor_input_records),
            "vendor_prediction_artifact": str(vendor_prediction_path.relative_to(ROOT)),
            "package_prediction_artifact": str(
                package_prediction_path.relative_to(ROOT)
            ),
            "vendor_predictions_sha256": _sha256(vendor_prediction_path),
            "package_predictions_sha256": _sha256(package_prediction_path),
            "expected_layout_count": (
                expected_unconditional_count
                if condition == "unconditional"
                else len(vendor_records)
            ),
            "num_layouts": len(vendor_records),
            "vendor_prediction_count_elements": sum(vendor_counts),
            "package_prediction_count_elements": sum(package_counts),
            "vendor_prediction_counts_per_layout": vendor_counts,
            "package_prediction_counts_per_layout": package_counts,
            "coordinate_frame": "original normalized center-xywh frame",
            "vendor_out_of_bounds_count_elements": _out_of_bounds_count(vendor_records),
            "package_out_of_bounds_count_elements": _out_of_bounds_count(
                package_records
            ),
            "vendor_metrics": vendor_metrics,
            "package_metrics": package_metrics,
            "metric_max_abs_diffs": metric_diffs,
            "max_abs_prediction_diff": prediction_max_abs_diff,
            "first_prediction_divergence": prediction_first,
            "evaluator_command": [
                sys.executable,
                str(ROOT / "vendor" / "layout-corrector" / "bin" / "calc_metrics.py"),
                str(package_dirs[condition].parent),
                "--force",
            ],
        }
    evaluation = _write_json("evaluation-path", dataset, results)
    assert evaluation.exists()
    assert results["corrector_weight_identity"]["same_key_set"] is True
    assert results["corrector_weight_identity"]["same_tensors"] is True
    assert results["corrector_weight_identity"]["artifact_equal"] is True
