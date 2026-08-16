"""Real-scale 300-step source/package lockstep preflight for CGL."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - source adapter APIs are dynamic.

import pytest
import torch

from radm import RADMConfig, RADMDenoiser
from radm.training.lightning_module import RADMTrainingModule
from radm.training.topology import (
    build_reviewed_state_key_map,
    copy_reviewed_state_dict,
)
from reference_adapter import (
    RADMReferenceAdapter,
    ReferenceTrainingState,
    ReferenceUnavailable,
    _legacy_pillow_compat,
    _vendor_import_root,
)
from test_s1_radm_training import (
    _gradient_norm,
    _rng_digest,
    _snapshot_gradients,
    _snapshot_optimizer_state,
)
from traingen_parity import capture_rng_state, restore_rng_state, tensor_sha256


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

_STEPS = 300
_LOSS_RELATIVE_LIMIT = 1e-3
_S2_ATOL = 5e-5
_S2_RTOL = 2e-4


def _max_relative_error(source: float, package: float) -> float:
    return abs(source - package) / max(abs(source), abs(package), 1e-12)


def _within_contract(source: float, package: float) -> bool:
    return abs(source - package) <= _S2_ATOL + _S2_RTOL * abs(source)


def _mapped_parameter_errors(
    source: Mapping[str, torch.Tensor],
    package: Mapping[str, torch.Tensor],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name, source_name in sorted(key_map.items()):
        source_value = source[source_name]
        package_value = package[package_name]
        if source_value.shape != package_value.shape:
            return math.inf, math.inf, package_name
        difference = (source_value - package_value).abs()
        current_abs = float(difference.max()) if difference.numel() else 0.0
        current_rel = _max_relative_error(
            float(source_value.abs().max()) if source_value.numel() else 0.0,
            float(package_value.abs().max()) if package_value.numel() else 0.0,
        )
        if current_abs > max_abs:
            max_abs = current_abs
            first_name = package_name
        max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _mapped_gradient_errors(
    source: Mapping[str, torch.Tensor | None],
    package: Mapping[str, torch.Tensor | None],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    source_by_package = {
        package_name: source[source_name]
        for package_name, source_name in key_map.items()
    }
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name in sorted(source_by_package):
        source_value = source_by_package[package_name]
        package_value = package[package_name]
        if (source_value is None) != (package_value is None):
            return math.inf, math.inf, package_name
        if source_value is None or package_value is None:
            continue
        current_abs, current_rel, _ = _mapped_parameter_errors(
            {"value": source_value},
            {"value": package_value},
            {"value": "value"},
        )
        if current_abs > max_abs:
            max_abs = current_abs
            first_name = package_name
        max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _optimizer_state_errors(
    source: Mapping[str, Mapping[str, Any]],
    package: Mapping[str, Mapping[str, Any]],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name, source_name in sorted(key_map.items()):
        source_values = source[source_name]
        package_values = package[package_name]
        if source_values.keys() != package_values.keys():
            return math.inf, math.inf, package_name
        for state_name in sorted(source_values):
            source_value = source_values[state_name]
            package_value = package_values[state_name]
            if isinstance(source_value, torch.Tensor) and isinstance(
                package_value, torch.Tensor
            ):
                current_abs = float((source_value - package_value).abs().max())
                current_rel = _max_relative_error(
                    float(source_value.abs().max()), float(package_value.abs().max())
                )
            elif source_value == package_value:
                current_abs = current_rel = 0.0
            else:
                current_abs = current_rel = math.inf
            if current_abs > max_abs:
                max_abs = current_abs
                first_name = f"{package_name}.{state_name}"
            max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _state_digest(
    source_model: torch.nn.Module,
    package_model: torch.nn.Module,
    key_map: Mapping[str, str],
    *,
    side: str,
) -> str:
    state = (
        source_model.state_dict() if side == "source" else package_model.state_dict()
    )
    digest = hashlib.sha256()
    for package_name, source_name in sorted(key_map.items()):
        name = source_name if side == "source" else package_name
        digest.update(name.encode())
        digest.update(
            tensor_sha256(
                state[source_name if side == "source" else package_name]
            ).encode()
        )
    return digest.hexdigest()


def _optimizer_digest(
    values: Mapping[str, Mapping[str, Any]],
    key_map: Mapping[str, str],
    *,
    package: bool,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(
        values, key=lambda value: key_map.get(value, value) if package else value
    ):
        canonical = key_map.get(name, name) if package else name
        for state_name, value in sorted(values[name].items()):
            digest.update(f"{canonical}.{state_name}".encode())
            digest.update(
                tensor_sha256(value).encode()
                if isinstance(value, torch.Tensor)
                else repr(value).encode()
            )
    return digest.hexdigest()


def _package_batch(
    source_batch: Sequence[Mapping[str, Any]],
    source_model: torch.nn.Module,
    effective: Any,
) -> dict[str, torch.Tensor]:
    model = cast(Any, source_model)
    images, image_scales = model.preprocess_image(list(source_batch))
    device = image_scales.device
    batch_size = len(source_batch)
    boxes = image_scales.new_zeros(batch_size, effective.num_proposals, 4)
    labels = torch.zeros(
        batch_size, effective.num_proposals, dtype=torch.long, device=device
    )
    mask = torch.zeros(
        batch_size, effective.num_proposals, dtype=torch.bool, device=device
    )
    text_features = torch.stack(
        [item["text_fea"]["feats"] for item in source_batch], dim=0
    ).to(device=device, dtype=images.tensor.dtype)
    text_mask = torch.stack([item["text_mask"] for item in source_batch], dim=0).to(
        device
    )
    for index, item in enumerate(source_batch):
        instances = item["instances"].to(device)
        count = min(len(instances), effective.num_proposals)
        if count:
            absolute = instances.gt_boxes.tensor[:count].to(dtype=images.tensor.dtype)
            boxes[index, :count] = absolute / image_scales[index]
            labels[index, :count] = instances.gt_classes[:count]
            mask[index, :count] = True
    return {
        "images": images.tensor,
        "image_scales": image_scales,
        "boxes_xyxy": boxes,
        "labels": labels,
        "mask": mask,
        "text_features": text_features,
        "text_mask": text_mask,
    }


def _source_loader(state: ReferenceTrainingState) -> Any:
    data = importlib.import_module("detectron2.data")
    mapper_class = importlib.import_module("RADM.dataset_mapper").RADMDatasetMapper
    mapper = mapper_class(state.config, is_train=True)
    return data.build_detection_train_loader(state.config, mapper=mapper)


def _next_batch(loader: Any, iterator: Any) -> tuple[list[dict[str, Any]], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _run_lockstep(
    state: ReferenceTrainingState,
    output_path: Path,
    *,
    steps: int = _STEPS,
) -> dict[str, object]:
    device = next(state.model.parameters()).device
    package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs())).to(device)
    key_map = build_reviewed_state_key_map(state.model, package)
    copy_reviewed_state_dict(
        state.model,
        package,
        key_map,
        allowlist=state.reviewed_state_allowlist,
    )
    module = RADMTrainingModule(
        config=package.radm_config,
        model=package,
        effective=state.effective,
    ).to(device)
    configured = cast(dict[str, Any], module.configure_optimizers())
    package_optimizer = cast(torch.optim.Optimizer, configured["optimizer"])
    package_scheduler = cast(
        torch.optim.lr_scheduler.LRScheduler,
        cast(dict[str, Any], configured["lr_scheduler"])["scheduler"],
    )
    loader = _source_loader(state)
    iterator = iter(loader)
    source_model = cast(Any, state.model)
    source_model.train()
    module.train()
    lines: list[dict[str, object]] = []
    first_divergence: dict[str, object] | None = None

    for step in range(1, steps + 1):
        source_batch, iterator = _next_batch(loader, iterator)
        package_batch = _package_batch(source_batch, source_model, state.effective)
        batch_ids = [int(item["image_id"]) for item in source_batch]

        state.optimizer.zero_grad()
        package_optimizer.zero_grad()
        rng_before = capture_rng_state()
        source_losses = source_model(source_batch)
        source_total = source_losses["loss_ce"] * 0
        for value in source_losses.values():
            source_total = source_total + value
        source_rng_after = capture_rng_state()
        restore_rng_state(rng_before)
        package_total = module._compute_step_loss(package_batch, record_trace=True)
        package_losses = {
            name: value
            for name, value in module.latest_step_trace.items()
            if name.startswith("loss_")
        }
        package_rng_after = capture_rng_state()
        source_rng_digest = _rng_digest(source_rng_after)
        package_rng_digest = _rng_digest(package_rng_after)
        source_total.backward()
        package_total.backward()
        source_gradients = _snapshot_gradients(state.model, state.optimizer)
        package_gradients = _snapshot_gradients(package, package_optimizer)
        source_preclip_norm = _gradient_norm(source_gradients)
        package_preclip_norm = _gradient_norm(package_gradients)

        state.optimizer.step()
        package_optimizer.step()
        source_postclip = _snapshot_gradients(state.model, state.optimizer)
        package_postclip = _snapshot_gradients(package, package_optimizer)
        source_optimizer_state = _snapshot_optimizer_state(state.model, state.optimizer)
        package_optimizer_state = _snapshot_optimizer_state(package, package_optimizer)
        state.scheduler.step()
        package_scheduler.step()

        source_loss_values = {
            name: float(value.detach()) for name, value in source_losses.items()
        }
        source_loss_values["total"] = float(source_total.detach())
        package_loss_values = {
            name: float(value.detach()) for name, value in package_losses.items()
        }
        package_loss_values["total"] = float(package_total.detach())
        loss_errors = {
            name: {
                "source": source_loss_values[name],
                "package": package_loss_values.get(name, float("nan")),
                "max_abs": abs(
                    source_loss_values[name]
                    - package_loss_values.get(name, float("nan"))
                ),
                "max_rel": _max_relative_error(
                    source_loss_values[name],
                    package_loss_values.get(name, float("nan")),
                ),
            }
            for name in source_loss_values
        }
        parameter_abs, parameter_rel, parameter_name = _mapped_parameter_errors(
            state.model.state_dict(), package.state_dict(), key_map
        )
        gradient_abs, gradient_rel, gradient_name = _mapped_gradient_errors(
            source_gradients, package_gradients, key_map
        )
        postclip_abs, postclip_rel, postclip_name = _mapped_gradient_errors(
            source_postclip, package_postclip, key_map
        )
        optimizer_abs, optimizer_rel, optimizer_name = _optimizer_state_errors(
            source_optimizer_state, package_optimizer_state, key_map
        )
        row: dict[str, object] = {
            "step": step,
            "batch_image_ids": batch_ids,
            "loss": loss_errors,
            "preclip_gradient_norm": {
                "source": source_preclip_norm,
                "package": package_preclip_norm,
                "max_rel": _max_relative_error(
                    source_preclip_norm, package_preclip_norm
                ),
            },
            "postclip_gradient_max_abs": postclip_abs,
            "postclip_gradient_max_rel": postclip_rel,
            "parameter_max_abs": parameter_abs,
            "parameter_max_rel": parameter_rel,
            "parameter_first_name": parameter_name,
            "gradient_max_abs": gradient_abs,
            "gradient_max_rel": gradient_rel,
            "gradient_first_name": gradient_name,
            "postclip_gradient_first_name": postclip_name,
            "optimizer_state_max_abs": optimizer_abs,
            "optimizer_state_max_rel": optimizer_rel,
            "optimizer_state_first_name": optimizer_name,
            "scheduler": {
                "source_last_epoch": state.scheduler.last_epoch,
                "package_last_epoch": package_scheduler.last_epoch,
                "source_lr": [float(value) for value in state.scheduler.get_last_lr()],
                "package_lr": [
                    float(value) for value in package_scheduler.get_last_lr()
                ],
            },
            "rng_equal": source_rng_digest == package_rng_digest,
            "model_sha256": {
                "source": _state_digest(state.model, package, key_map, side="source"),
                "package": _state_digest(state.model, package, key_map, side="package"),
            },
            "optimizer_sha256": {
                "source": _optimizer_digest(
                    source_optimizer_state, key_map, package=False
                ),
                "package": _optimizer_digest(
                    package_optimizer_state, key_map, package=True
                ),
            },
        }
        lines.append(row)

        if first_divergence is None:
            for name, error in loss_errors.items():
                if not _within_contract(
                    float(error["source"]), float(error["package"])
                ) or (
                    name == "total" and float(error["max_rel"]) > _LOSS_RELATIVE_LIMIT
                ):
                    first_divergence = {
                        "step": step,
                        "surface": f"loss.{name}",
                        "max_abs": error["max_abs"],
                        "max_rel": error["max_rel"],
                    }
                    break
            if first_divergence is None and not _within_contract(
                source_preclip_norm, package_preclip_norm
            ):
                first_divergence = {
                    "step": step,
                    "surface": "preclip_gradient_norm",
                    "max_rel": row["preclip_gradient_norm"]["max_rel"],
                }
            if first_divergence is None and gradient_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"preclip_gradients.{gradient_name}",
                    "max_abs": gradient_abs,
                    "max_rel": gradient_rel,
                }
            if first_divergence is None and parameter_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"parameters.{parameter_name}",
                    "max_abs": parameter_abs,
                    "max_rel": parameter_rel,
                }
            if first_divergence is None and optimizer_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"optimizer_state.{optimizer_name}",
                    "max_abs": optimizer_abs,
                    "max_rel": optimizer_rel,
                }
            if first_divergence is None and not bool(row["rng_equal"]):
                first_divergence = {"step": step, "surface": "rng_after_step"}

    report: dict[str, object] = {
        "mode": "real_scale_300_step_lockstep",
        "steps": steps,
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
        "first_divergence": first_divergence,
        "records": len(lines),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite lockstep evidence: {output_path}")
    output_path.write_text(
        "".join(
            json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
            for line in lines
        )
        + json.dumps({"summary": report}, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def test_radm_300_step_cgl_lockstep() -> None:
    """Run and enforce the real CGL lockstep only when explicitly enabled."""
    if os.environ.get("RADM_RUN_300_LOCKSTEP") != "1":
        pytest.skip("set RADM_RUN_300_LOCKSTEP=1 to launch the real 300-step preflight")
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.fail("PARITY_REQUIRE=1 is required for the real-scale lockstep")
    if not torch.cuda.is_available():
        pytest.fail("the real-scale lockstep requires CUDA")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    weights_path = Path(
        os.environ.get("RADM_R50_WEIGHTS", ".cache/radm/weights/R-50.pkl")
    )
    required = (
        data_root / "annotations" / "train.json",
        data_root / "images" / "train",
        data_root / "text_features" / "train",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.fail(f"CGL lockstep assets are missing: {missing}")
    if not weights_path.is_file():
        pytest.fail(f"R-50 initialization weights are missing: {weights_path}")
    device = os.environ.get("RADM_REFERENCE_DEVICE", "cuda:0")
    output_path = Path(
        os.environ.get(
            "RADM_300_LOCKSTEP_EVIDENCE_PATH",
            ".cache/radm/s5-preflight/lockstep-300.jsonl",
        )
    )
    try:
        with _vendor_import_root(Path("vendor/radm")), _legacy_pillow_compat():
            state = RADMReferenceAdapter(
                vendor_root=Path("vendor/radm"),
                dataset_root=data_root,
                text_feature_root=data_root / "text_features",
                device=device,
            ).build_initialized_state()
            checkpointer = importlib.import_module(
                "detectron2.checkpoint"
            ).DetectionCheckpointer(state.model)
            checkpointer.load(str(weights_path))
            report = _run_lockstep(state, output_path)
    except ReferenceUnavailable as exc:
        pytest.fail(str(exc))
    assert report["records"] == _STEPS
    assert report["first_divergence"] is None, json.dumps(report, sort_keys=True)
