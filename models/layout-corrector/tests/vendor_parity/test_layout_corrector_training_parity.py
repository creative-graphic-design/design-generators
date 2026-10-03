from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Protocol, cast  # noqa: TID251 - vendor APIs are dynamic.

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")
pytest.importorskip("omegaconf")
pytest.importorskip("torch_geometric")

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from laygen.common.testing import skip_or_fail_vendor_parity
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
    build_layout_corrector_step_trace,
    compare_layout_corrector_step,
)
from layout_corrector.training.config import (
    LayoutCorrectorTrainingDatasetName,
    LayoutCorrectorTrainingSplit,
)
from layout_dm.configuration_layout_dm import LayoutDMConfig

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT: Final = Path(__file__).resolve().parents[4]
DATASETS: Final[tuple[LayoutCorrectorTrainingDatasetName, ...]] = (
    "rico25",
    "publaynet",
)
CORRECTOR_HIDDEN_SIZE: Final = 432
CORRECTOR_INTERMEDIATE_SIZE: Final = 1728
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
        return {"environment": "lockfile"}

    python = Path(runtime) / "bin" / "python"
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
    if "prdc" not in sys.modules:
        prdc_module = types.ModuleType("prdc")
        setattr(prdc_module, "compute_prdc", lambda **_: {})
        sys.modules["prdc"] = prdc_module
    if "pytorch_fid.fid_score" not in sys.modules:
        fid_module = types.ModuleType("pytorch_fid.fid_score")
        setattr(
            fid_module,
            "calculate_frechet_distance",
            lambda *_args, **_kwargs: 0.0,
        )
        sys.modules["pytorch_fid"] = types.ModuleType("pytorch_fid")
        sys.modules["pytorch_fid.fid_score"] = fid_module
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
        dataset_name=dataset,
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
    vendor_initialization_device: str
    package_initialization_device: str


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
    package_reference = FrozenLayoutDMReference.from_checkpoint(
        dataset_name=dataset,
        checkpoint_path=_checkpoint_path(dataset),
        cluster_centers_path=_cluster_path(dataset),
    )
    vendor_diffusion, vendor_tokenizer = _vendor_reference(
        dataset, tokenizer_cls, layout_dm_cls, backbone
    )
    torch.manual_seed(seed)
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
    package = LayoutCorrectorTrainingModule(
        config=_corrector_config(dataset, package_config.vocab_size),
        layout_dm_checkpoint_path=_checkpoint_path(dataset),
        cluster_centers_path=_cluster_path(dataset),
        learning_rate=5.0e-4,
        weight_decay=0.1,
        betas=(0.9, 0.98),
        gradient_clip_norm=1.0,
    )
    vendor_initialization_device = str(next(vendor_corrector.parameters()).device)
    package_initialization_device = str(next(package.model.parameters()).device)
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
        vendor_initialization_device=vendor_initialization_device,
        package_initialization_device=package_initialization_device,
    )


def _loader_worker_count() -> int:
    return int(os.environ.get("LAYOUT_CORRECTOR_EVIDENCE_WORKERS", "16"))


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

    workers = _loader_worker_count() if num_workers is None else num_workers
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
    return {
        "input_ids": vendor_encoded["seq"],
        "attention_mask": vendor_encoded["mask"],
    }, package_batch


def _vendor_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "seq": cast(torch.Tensor, batch["input_ids"]).to(device),
        "mask": cast(torch.Tensor, batch["attention_mask"]).to(device),
    }


def _vendor_trace(fixture: Fixture, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    from omegaconf import OmegaConf

    prepared = fixture.vendor.preprocess(
        _vendor_batch(batch, next(fixture.vendor.parameters()).device),
        fixture.vendor_diffusion,
        OmegaConf.create({"name": "random", "temperature": 1.0}),
    )
    outputs = fixture.vendor.model.module(
        prepared["x0_recon"],
        timestep=prepared["t"],
        self_cond=None,
        src_key_padding_mask=None,
        attention_bias=None,
    )
    logits = outputs["logits"].squeeze(-1)
    target = prepared["recon_acc"].float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.ones(5, device=logits.device, dtype=logits.dtype).repeat(25)
    loss = bce.mul(weights.unsqueeze(0)).mean()
    return {
        **prepared,
        "logits": logits.detach(),
        "bce_loss": bce.detach(),
        "weighted_bce_loss": (bce * weights).detach(),
        "train_loss": loss,
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
    report = compare_optimizer_step(reference, target)
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
    return int(os.environ.get("LAYOUT_CORRECTOR_S3_STEPS", "300"))


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


def _nested_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            digest.update(_tensor_digest(item).encode())
            return
        if isinstance(item, dict):
            digest.update(b"dict")
            pairs = sorted(item.items(), key=lambda pair: str(pair[0]))
            for key, child in pairs:
                digest.update(str(key).encode())
                update(child)
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                update(child)
            return
        digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _optimizer_defaults(optimizer: torch.optim.Optimizer) -> dict[str, str]:
    return {key: repr(value) for key, value in sorted(optimizer.defaults.items())}


def _assert_optimizer_state_independent(
    reference: dict[str, Any], target: dict[str, Any]
) -> None:
    """Fail if same-device optimizer loading aliases tensor storage."""
    for state_id, reference_state in reference.get("state", {}).items():
        target_state = target.get("state", {}).get(state_id, {})
        for name, value in reference_state.items():
            target_value = target_state.get(name)
            if isinstance(value, torch.Tensor) and isinstance(
                target_value, torch.Tensor
            ):
                if value.data_ptr() == target_value.data_ptr():
                    raise AssertionError(
                        f"optimizer state storage is shared for {state_id}:{name}"
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
    assert not dataset_first, dataset_input_diffs
    assert not corrector_first, corrector_diffs
    expected_initialization_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert fixture.vendor_initialization_device == expected_initialization_device
    assert fixture.package_initialization_device == expected_initialization_device
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
            "vendor_initialization_device": fixture.vendor_initialization_device,
            "package_initialization_device": fixture.package_initialization_device,
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
    fixture = _fixture(dataset, torch.device("cpu"))
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
    fixture = _fixture(dataset, torch.device("cpu"))
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
    vendor_optimizer.zero_grad()
    package_optimizer.zero_grad()
    vendor_loss = vendor_trace["train_loss"]
    package_loss = package_trace["train_loss"]
    vendor_loss.backward()
    package_loss.backward()
    vendor_grad_norm = torch.nn.utils.clip_grad_norm_(fixture.vendor.parameters(), 1.0)
    package_grad_norm = torch.nn.utils.clip_grad_norm_(
        fixture.package.model.parameters(), 1.0
    )
    vendor_optimizer.step()
    package_optimizer.step()
    diffs, first = _state_diffs(
        fixture.vendor.model.module.state_dict(),
        fixture.package.model.model.state_dict(),
    )
    assert not first, diffs
    assert torch.allclose(vendor_grad_norm, package_grad_norm, atol=1e-5, rtol=1e-5)
    path = _write_json(
        "s2-optimizer-step",
        dataset,
        {
            "dataset": dataset,
            "max_abs_parameter_diffs": diffs,
            "first_divergence": first,
            "vendor_gradient_norm": float(vendor_grad_norm.item()),
            "package_gradient_norm": float(package_grad_norm.item()),
            "scheduler_step": "not before validation boundary",
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
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Materialize paired production loader streams for one natural run."""
    from trainer.data.util import compose_transform, sparse_to_dense

    workers = _loader_worker_count()
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
    vendor_batches: list[dict[str, torch.Tensor]] = []
    vendor_ids: list[list[str]] = []
    for batch_index, batch in enumerate(vendor_loader):
        if batch_index >= steps:
            break
        bbox, labels, _, mask = sparse_to_dense(batch)
        encoded = tokenizer.encode({"bbox": bbox, "label": labels, "mask": mask})
        vendor_batches.append(
            {"input_ids": encoded["seq"], "attention_mask": encoded["mask"]}
        )
        vendor_ids.append([str(value) for value in batch.attr["name"]])
    torch.manual_seed(seed)
    package_batches: list[dict[str, Any]] = []
    package_ids: list[list[str]] = []
    for batch_index, batch in enumerate(package_loader):
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
    return vendor_batches, package_batches


def _run_natural_side(
    fixture: Fixture,
    batches: list[dict[str, Any]],
    *,
    side: str,
    initial_state: dict[str, torch.Tensor],
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Run one system without restoring RNG inside the multi-step trajectory."""
    if side == "vendor":
        model = fixture.vendor.model.module
        optimizer = _adamw(fixture.vendor.optim_groups(weight_decay=0.1))
    else:
        model = fixture.package.model.model
        optimizer = _adamw(fixture.package.optim_groups())
    model.load_state_dict(initial_state, strict=True)
    _apply_s3_determinism()
    rows: list[dict[str, Any]] = []
    for step, batch in enumerate(batches[:steps]):
        optimizer.zero_grad()
        rng_digest = _state_digest(capture_rng_state())
        trace = (
            _vendor_trace(fixture, batch)
            if side == "vendor"
            else _package_trace(fixture, batch)
        )
        trace["train_loss"].backward()
        parameters = (
            fixture.vendor.parameters()
            if side == "vendor"
            else fixture.package.model.parameters()
        )
        grad_norm = _clip_grad_norm(parameters)
        optimizer.step()
        loss = float(trace["train_loss"].detach().cpu().item())
        importance_probability = trace.get("pt")
        rows.append(
            {
                "step": step,
                "loss": loss,
                "gradient_norm": float(grad_norm.detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_state_digest": _nested_digest(optimizer.state_dict()),
                "parameter_state_digest": _state_dict_digest(model.state_dict()),
                "rng_digest": rng_digest,
                "timesteps_digest": _tensor_digest(trace["t"]),
                "importance_probability_digest": (
                    None
                    if importance_probability is None
                    else _tensor_digest(importance_probability)
                ),
            }
        )
    return rows, deepcopy(model.state_dict())


def _natural_comparison(
    vendor_rows: list[dict[str, Any]], package_rows: list[dict[str, Any]]
) -> dict[str, Any]:
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
            if relative > 1.0e-3
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
    return {
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
        "first_divergence": first,
        "relative_loss_criterion_passed": first is None,
    }


def _repeat_envelope(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> dict[str, Any]:
    loss_diffs = [
        abs(left["loss"] - right["loss"])
        for left, right in zip(first, second, strict=True)
    ]
    return {
        "max_abs_loss_diff": max(loss_diffs),
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
    }


@pytest.mark.parametrize("dataset", DATASETS)
def test_s3_natural_lockstep_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = _s3_steps()
    _apply_s3_determinism()
    fixture = _fixture(dataset, device)
    vendor_batches, package_batches = _natural_training_batches(
        dataset, fixture.vendor_tokenizer, steps
    )
    initial_state = deepcopy(fixture.vendor.model.module.state_dict())
    vendor_rows, vendor_state = _run_natural_side(
        fixture,
        vendor_batches,
        side="vendor",
        initial_state=initial_state,
        steps=steps,
    )
    vendor_repeat_rows, _ = _run_natural_side(
        fixture,
        vendor_batches,
        side="vendor",
        initial_state=initial_state,
        steps=steps,
    )
    package_rows, package_state = _run_natural_side(
        fixture,
        package_batches,
        side="package",
        initial_state=initial_state,
        steps=steps,
    )
    package_repeat_rows, _ = _run_natural_side(
        fixture,
        package_batches,
        side="package",
        initial_state=initial_state,
        steps=steps,
    )
    comparison = _natural_comparison(vendor_rows, package_rows)
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
            "num_workers": _loader_worker_count(),
            "adamw_mode": os.environ.get("LAYOUT_CORRECTOR_ADAMW_MODE", "default"),
            "gradient_clip_call": "torch.nn.utils.clip_grad_norm_(parameters, 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=None)",
            "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "comparison": comparison,
            "final_parameter_max_abs_diffs": final_parameter_diffs,
            "final_parameter_first_difference": final_parameter_first,
            "repeat_run_envelope": {
                "vendor": _repeat_envelope(vendor_rows, vendor_repeat_rows),
                "package": _repeat_envelope(package_rows, package_repeat_rows),
            },
            "rng_restore_inside_run": False,
            "shared_injected_random_tensors": False,
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
    assert comparison["relative_loss_criterion_passed"], comparison


@pytest.mark.parametrize("dataset", DATASETS)
def test_s3_synchronized_diagnostic_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _apply_s3_determinism()
    fixture = _fixture(dataset, device)
    vendor_batches, package_batches = _natural_training_batches(
        dataset, fixture.vendor_tokenizer, _s3_steps()
    )
    vendor_optimizer = _adamw(fixture.vendor.optim_groups(weight_decay=0.1))
    package_optimizer = _adamw(fixture.package.optim_groups())
    vendor_named_parameters = list(fixture.vendor.model.module.named_parameters())
    package_named_parameters = list(fixture.package.model.model.named_parameters())
    package_parameters_by_name = dict(package_named_parameters)
    aligned_clip = os.environ.get("LAYOUT_CORRECTOR_ALIGNED_CLIP") == "1"
    if aligned_clip:
        vendor_clip_parameters = [parameter for _, parameter in vendor_named_parameters]
        package_clip_parameters = [
            package_parameters_by_name[name] for name, _ in vendor_named_parameters
        ]
        clip_foreach: bool | None = False
    else:
        vendor_clip_parameters = list(fixture.vendor.parameters())
        package_clip_parameters = list(fixture.package.model.parameters())
        clip_foreach = None
    first_loss_divergence: dict[str, Any] | None = None
    first_trace_difference: dict[str, Any] | None = None
    first_gradient_difference: dict[str, Any] | None = None
    first_parameter_difference: dict[str, Any] | None = None
    max_trace_diff = 0.0
    max_gradient_diff = 0.0
    max_parameter_diff = 0.0
    trace_rows: list[dict[str, Any]] = []
    steps = _s3_steps()
    for step, (vendor_batch, package_batch) in enumerate(
        zip(vendor_batches, package_batches, strict=True)
    ):
        pre_model_state = deepcopy(fixture.vendor.model.module.state_dict())
        pre_optimizer_state = deepcopy(vendor_optimizer.state_dict())
        rng_state = capture_rng_state()
        vendor_optimizer.zero_grad()
        vendor_trace = _vendor_trace(fixture, vendor_batch)
        vendor_trace["train_loss"].backward()

        fixture.package.model.model.load_state_dict(pre_model_state, strict=True)
        package_optimizer.load_state_dict(deepcopy(pre_optimizer_state))
        _assert_optimizer_state_independent(
            pre_optimizer_state, package_optimizer.state_dict()
        )
        package_optimizer.zero_grad()
        restore_rng_state(rng_state)
        package_trace = _package_trace(fixture, package_batch)
        package_trace["train_loss"].backward()
        package_trace_diffs, package_trace_first = _tensor_diffs(
            vendor_trace, package_trace
        )
        trace_diff = max(package_trace_diffs.values())
        if first_trace_difference is None and package_trace_first:
            first_trace_difference = {
                "step": step,
                "tensor": package_trace_first,
                "max_abs": trace_diff,
            }
        gradient_diff, gradient_name = _gradient_diffs(
            fixture.vendor.model.module, fixture.package.model.model
        )
        vendor_grad_norm = _clip_grad_norm(vendor_clip_parameters, foreach=clip_foreach)
        package_grad_norm = _clip_grad_norm(
            package_clip_parameters, foreach=clip_foreach
        )
        vendor_optimizer.step()
        package_optimizer.step()

        parameter_diffs, parameter_first = _state_diffs(
            fixture.vendor.model.module.state_dict(),
            fixture.package.model.model.state_dict(),
        )
        parameter_diff = max(parameter_diffs.values())
        max_trace_diff = max(max_trace_diff, trace_diff)
        max_gradient_diff = max(max_gradient_diff, gradient_diff)
        max_parameter_diff = max(max_parameter_diff, parameter_diff)
        if first_gradient_difference is None and gradient_name:
            first_gradient_difference = {
                "step": step,
                "tensor": gradient_name,
                "max_abs": gradient_diff,
            }
        if first_parameter_difference is None and parameter_first:
            first_parameter_difference = {
                "step": step,
                "tensor": parameter_first,
                "max_abs": parameter_diff,
            }
        vendor_loss = float(vendor_trace["train_loss"].detach().cpu().item())
        package_loss = float(package_trace["train_loss"].detach().cpu().item())
        loss_diff = abs(vendor_loss - package_loss)
        relative_loss = loss_diff / max(abs(vendor_loss), 1.0e-12)
        if first_loss_divergence is None and relative_loss > 1.0e-3:
            first_loss_divergence = {
                "step": step,
                "vendor_loss": vendor_loss,
                "package_loss": package_loss,
                "absolute": loss_diff,
                "relative": relative_loss,
            }
        trace_rows.append(
            {
                "step": step,
                "vendor_loss": vendor_loss,
                "package_loss": package_loss,
                "loss_abs_diff": loss_diff,
                "loss_relative_diff": relative_loss,
                "vendor_gradient_norm": float(vendor_grad_norm.detach().cpu().item()),
                "package_gradient_norm": float(package_grad_norm.detach().cpu().item()),
                "vendor_optimizer_state_digest": _nested_digest(
                    vendor_optimizer.state_dict()
                ),
                "package_optimizer_state_digest": _nested_digest(
                    package_optimizer.state_dict()
                ),
                "max_abs_trace_diff": trace_diff,
                "max_abs_gradient_diff": gradient_diff,
                "max_abs_parameter_diff": parameter_diff,
                "rng_digest": _state_digest(rng_state),
                "timesteps_digest": _tensor_digest(vendor_trace["t"]),
                "importance_probability_digest": _tensor_digest(package_trace["pt"]),
            }
        )
    assert len(trace_rows) == steps
    stage = _s3_stage("s3-lockstep-synchronized")
    trace_path = _write_jsonl(stage, dataset, trace_rows)
    path = _write_json(
        stage,
        dataset,
        {
            "dataset": dataset,
            "steps": len(trace_rows),
            "num_workers": _loader_worker_count(),
            "adamw_mode": os.environ.get("LAYOUT_CORRECTOR_ADAMW_MODE", "default"),
            "gradient_clip_mode": (
                "disabled"
                if os.environ.get("LAYOUT_CORRECTOR_DISABLE_CLIP") == "1"
                else "aligned_parameter_order_norm_1.0"
                if aligned_clip
                else "norm_1.0"
            ),
            "gradient_clip_call": (
                "torch.nn.utils.clip_grad_norm_(parameters, 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=False)"
                if aligned_clip
                else "torch.nn.utils.clip_grad_norm_(parameters, 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=None)"
            ),
            "vendor_parameter_order_digest": _parameter_order_digest(
                [name for name, _ in vendor_named_parameters]
            ),
            "package_parameter_order_digest": _parameter_order_digest(
                [name for name, _ in package_named_parameters]
            ),
            "clip_parameter_order": "vendor-name-order" if aligned_clip else "native",
            "registration_order_equal": [name for name, _ in vendor_named_parameters]
            == [name for name, _ in package_named_parameters],
            "max_abs_trace_diff": max_trace_diff,
            "max_abs_gradient_diff": max_gradient_diff,
            "max_abs_parameter_diff": max_parameter_diff,
            "trace_comparison_passed": first_trace_difference is None,
            "gradient_comparison_passed": first_gradient_difference is None,
            "parameter_comparison_passed": first_parameter_difference is None,
            "relative_loss_criterion_passed": first_loss_divergence is None,
            "first_divergence": first_loss_divergence,
            "first_trace_difference": first_trace_difference,
            "first_gradient_difference": first_gradient_difference,
            "first_parameter_difference": first_parameter_difference,
            "trace_artifact": str(trace_path.relative_to(ROOT)),
            "runtime": _runtime_record(),
            "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert path.exists()
    assert first_loss_divergence is None, first_loss_divergence


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
        num_workers=16,
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
            evidence_num_workers=_loader_worker_count(),
        )
        for split in ("train", "validation", "test")
    }
    path = _write_json(
        "loader-stream",
        dataset,
        {
            "dataset": dataset,
            "splits": split_rows,
            "configured_num_workers": 16,
            "evidence_num_workers": _loader_worker_count(),
            "num_workers_override": "none; vendor-configured 16 workers retained",
            "batch_stream_comparison": all(
                row["max_abs_input_id_diff"] == 0
                and row["attention_mask_mismatch_count"] == 0
                for rows in split_rows.values()
                for row in rows
            ),
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


def _write_weight_artifact(path: Path, state_dict: dict[str, torch.Tensor]) -> str:
    with path.open("wb") as handle:
        for name in sorted(state_dict):
            value = state_dict[name].detach().cpu().contiguous()
            encoded_name = name.encode()
            raw = value.view(torch.uint8).numpy().tobytes()
            handle.write(len(encoded_name).to_bytes(8, "little"))
            handle.write(encoded_name)
            handle.write(str(value.dtype).encode() + b"\0")
            handle.write(len(raw).to_bytes(8, "little"))
            handle.write(raw)
    return _sha256(path)


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


def _evaluation_metric_values(
    predictions: list[dict[str, list[Any]]],
) -> dict[str, float]:
    from trainer.helpers.metric import (
        compute_alignment,
        compute_average_iou,
        compute_docsim,
        compute_overlap,
    )

    layouts = [
        (
            torch.as_tensor(item["bbox"], dtype=torch.float32).numpy(),
            torch.as_tensor(item["labels"], dtype=torch.long).numpy(),
        )
        for item in predictions
    ]
    max_length = max((len(item["bbox"]) for item in predictions), default=0)
    bbox = torch.zeros((len(predictions), max_length, 4), dtype=torch.float32)
    mask = torch.zeros((len(predictions), max_length), dtype=torch.bool)
    for index, item in enumerate(predictions):
        values = torch.as_tensor(item["bbox"], dtype=torch.float32)
        if values.numel():
            bbox[index, : values.shape[0]] = values
            mask[index, : values.shape[0]] = True
    metrics: dict[str, float] = {}
    for function in (compute_alignment, compute_overlap):
        metrics.update(
            {
                key: float(value.mean().item())
                for key, value in function(bbox, mask).items()
            }
        )
    metrics.update(compute_average_iou(layouts))
    metrics["self_docsim"] = float(compute_docsim(layouts, layouts))
    return metrics


@pytest.mark.parametrize("dataset", DATASETS)
def test_s4_test_evaluation_path_matches_vendor(
    dataset: LayoutCorrectorTrainingDatasetName,
) -> None:
    apply_determinism(DeterminismConfig(seed=314159, deterministic_algorithms=False))
    device = torch.device(os.environ.get("LAYOUT_CORRECTOR_S4_DEVICE", "cpu"))
    fixture = _fixture(dataset, device)
    from trainer import corrector_test

    vendor_dataset = _vendor_dataset(dataset, "test")
    vendor_loader = GeometricDataLoader(
        vendor_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    class EvaluationProbeDiffusion:
        tokenizer = fixture.vendor_tokenizer

        def __init__(self) -> None:
            self.input_batches: list[torch.Tensor] = []
            self.score_batches: list[torch.Tensor] = []
            self.batch_sizes: list[int] = []

        def sample(
            self,
            *,
            batch_size: int,
            cond: dict[str, torch.Tensor],
            sampling_cfg: dict[str, Any],
            cond_type: str,
            corrector: Any,
        ) -> dict[str, torch.Tensor]:
            del batch_size, sampling_cfg
            assert cond_type == "gt"
            probe_ids = cond["seq"].to(device)
            timestep = torch.full(
                (probe_ids.shape[0],), 10, dtype=torch.long, device=probe_ids.device
            )
            scores = corrector.calc_confidence_score(probe_ids, timestep)
            self.input_batches.append(probe_ids.detach().cpu())
            self.score_batches.append(scores.detach().cpu())
            self.batch_sizes.append(int(probe_ids.shape[0]))
            return self.tokenizer.decode(probe_ids.cpu())

    probe_diffusion = EvaluationProbeDiffusion()
    _, vendor_predictions, _, vendor_total = corrector_test.run(
        probe_diffusion,
        fixture.vendor,
        vendor_loader,
        SimpleNamespace(cond="gt", num_run=1, refine_noise_std=0.1),
        {},
    )
    vendor_inputs = torch.cat(probe_diffusion.input_batches)
    vendor_scores = torch.cat(probe_diffusion.score_batches)
    vendor_records = _prediction_records(vendor_predictions)

    # Use the vendor evaluator's complete TEST preprocessing output as the
    # canonical input for both systems. The loader-stream artifact separately
    # checks package-side TEST loading on two batches.
    package_inputs: list[torch.Tensor] = []
    package_scores_batches: list[torch.Tensor] = []
    package_predictions: list[tuple[Any, Any]] = []
    offset = 0
    for batch_size in probe_diffusion.batch_sizes:
        input_batch = vendor_inputs[offset : offset + batch_size]
        offset += batch_size
        input_ids = input_batch.to(device)
        timesteps = torch.full(
            (input_ids.shape[0],), 10, dtype=torch.long, device=device
        )
        scores = fixture.package.model.calc_confidence_score(input_ids, timesteps)
        decoded = fixture.package_reference.tokenizer.decode_layout(input_ids.cpu())
        package_inputs.append(input_ids.detach().cpu())
        package_scores_batches.append(scores.detach().cpu())
        package_predictions.extend(
            (
                decoded["bbox"][index][valid].numpy(),
                decoded["labels"][index][valid].numpy(),
            )
            for index, valid in enumerate(decoded["mask"])
        )
    package_inputs_tensor = torch.cat(package_inputs)
    package_scores = torch.cat(package_scores_batches)
    package_records = _prediction_records(package_predictions)
    confidence_score_diff = (vendor_scores - package_scores).abs()
    assert torch.equal(vendor_inputs, package_inputs_tensor)
    assert vendor_records == package_records
    assert vendor_total == len(vendor_records) == len(package_records)
    vendor_prediction_counts = [len(item["bbox"]) for item in vendor_records]
    package_prediction_counts = [len(item["bbox"]) for item in package_records]
    vendor_metric = float(sum(vendor_prediction_counts) / max(vendor_total, 1))
    package_metric = float(
        sum(package_prediction_counts) / max(len(package_records), 1)
    )
    prediction_max_abs_diff, prediction_first = _prediction_difference(
        vendor_records, package_records
    )
    vendor_metrics = _evaluation_metric_values(vendor_records)
    package_metrics = _evaluation_metric_values(package_records)
    metric_diffs, metric_first = _mapping_diffs(
        {key: torch.tensor(value) for key, value in vendor_metrics.items()},
        {key: torch.tensor(value) for key, value in package_metrics.items()},
    )
    assert prediction_first is None
    assert not metric_first, metric_diffs

    evidence_dir = _evidence_root() / "evaluation-path" / dataset
    evidence_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = evidence_dir / "test-input-ids.bin"
    inputs_path.write_bytes(vendor_inputs.contiguous().numpy().tobytes())
    vendor_prediction_path = evidence_dir / "vendor-predictions.json"
    package_prediction_path = evidence_dir / "package-predictions.json"
    vendor_prediction_path.write_text(
        json.dumps(vendor_records, sort_keys=True, separators=(",", ":")) + "\n"
    )
    package_prediction_path.write_text(
        json.dumps(package_records, sort_keys=True, separators=(",", ":")) + "\n"
    )
    vendor_weights_path = evidence_dir / "vendor-weights.bin"
    package_weights_path = evidence_dir / "package-weights.bin"
    vendor_state_digest = _state_dict_digest(fixture.vendor.model.module.state_dict())
    package_state_digest = _state_dict_digest(fixture.package.model.model.state_dict())
    assert vendor_state_digest == package_state_digest
    vendor_weights_sha256 = _write_weight_artifact(
        vendor_weights_path, fixture.vendor.model.module.state_dict()
    )
    package_weights_sha256 = _write_weight_artifact(
        package_weights_path, fixture.package.model.model.state_dict()
    )
    assert vendor_weights_sha256 == package_weights_sha256
    vendor_predictions_sha256 = _sha256(vendor_prediction_path)
    package_predictions_sha256 = _sha256(package_prediction_path)
    assert vendor_predictions_sha256 == package_predictions_sha256
    vendor_out_of_bounds = _out_of_bounds_count(vendor_records)
    package_out_of_bounds = _out_of_bounds_count(package_records)
    assert vendor_out_of_bounds == package_out_of_bounds
    evaluation = _write_json(
        "evaluation-path",
        dataset,
        {
            "dataset": dataset,
            "vendor_entry_point": "vendor/layout-corrector/src/trainer/trainer/corrector_test.py::run",
            "vendor_entry_point_wrapper": "vendor/layout-corrector/bin/corrector_test_eval.py",
            "vendor_entry_point_adapter": "direct call to trainer.corrector_test.run with the script default max_batch_size=512",
            "package_entry_point": "LayoutCorrectorModel.calc_confidence_score + LayoutDMTokenizer.decode_layout",
            "split": "test",
            "evaluator_settings": {
                "condition": "gt",
                "timestep": 10,
                "batch_size": 512,
                "random_order": False,
                "gumbel_noise": False,
                "sampling_seed": 314159,
            },
            "num_test_layouts": len(vendor_records),
            "same_input_ids_digest": _tensor_digest(vendor_inputs),
            "inputs_artifact": str(inputs_path.relative_to(ROOT)),
            "inputs_sha256": _sha256(inputs_path),
            "same_corrector_weights": vendor_state_digest == package_state_digest,
            "corrector_state_digest_vendor": vendor_state_digest,
            "corrector_state_digest_package": package_state_digest,
            "vendor_weights_artifact": str(vendor_weights_path.relative_to(ROOT)),
            "package_weights_artifact": str(package_weights_path.relative_to(ROOT)),
            "vendor_weights_sha256": vendor_weights_sha256,
            "package_weights_sha256": package_weights_sha256,
            "vendor_predictions_artifact": str(
                vendor_prediction_path.relative_to(ROOT)
            ),
            "package_predictions_artifact": str(
                package_prediction_path.relative_to(ROOT)
            ),
            "vendor_predictions_sha256": vendor_predictions_sha256,
            "package_predictions_sha256": package_predictions_sha256,
            "vendor_prediction_count": int(sum(vendor_prediction_counts)),
            "package_prediction_count": int(sum(package_prediction_counts)),
            "vendor_prediction_counts_per_layout": vendor_prediction_counts,
            "package_prediction_counts_per_layout": package_prediction_counts,
            "vendor_out_of_bounds_count_original_frame": vendor_out_of_bounds,
            "package_out_of_bounds_count_original_frame": package_out_of_bounds,
            "coordinate_frame": "normalized center xywh in [0, 1]",
            "max_abs_prediction_diff": prediction_max_abs_diff,
            "max_abs_confidence_score_diff": float(confidence_score_diff.max().item()),
            "metrics": {
                "identical": metric_first is None,
                "vendor": vendor_metrics,
                "package": package_metrics,
                "max_abs_diffs": metric_diffs,
                "valid_elements_per_prediction": {
                    "vendor": vendor_metric,
                    "package": package_metric,
                },
            },
            "same_inputs": torch.equal(vendor_inputs, package_inputs_tensor),
            "prediction_files_identical": vendor_predictions_sha256
            == package_predictions_sha256,
            "out_of_bounds_identical": vendor_out_of_bounds == package_out_of_bounds,
            "first_divergence": {
                "prediction": prediction_first,
                "metric": metric_first,
            },
            "vendor_source_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "vendor_evaluator_commit": _source_commit(
                ROOT / "vendor" / "layout-corrector"
            ),
            "layoutdm_source_commit": _source_commit(ROOT / "vendor" / "layout-dm"),
            "source_commit": _source_commit(ROOT),
        },
    )
    assert evaluation.exists()
