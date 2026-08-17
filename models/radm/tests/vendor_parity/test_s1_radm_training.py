"""S1 fixed-batch training parity against the pinned RADM source."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - Detectron2 outputs are dynamic.
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from radm import RADMConfig, RADMDenoiser
from radm.training.lightning_module import (
    RADMTarget,
    RADMTrainingModule,
    _classification_loss,
    _dynamic_k_match,
    _sigmoid_focal_loss,
    _xyxy_to_cxcywh,
)
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
from traingen_parity import (
    DeterminismConfig,
    TensorTolerance,
    apply_determinism,
    build_step_trace,
    capture_rng_state,
    compare_step_trace,
    restore_rng_state,
    tensor_sha256,
)


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

_SEED = 261
_VENDOR_REVISION = "413f87a45760ceac5635b6a08c8047f86478acf5"
_FLOAT_TOLERANCE = TensorTolerance(atol=2e-5, rtol=2e-5)
_S2_FLOAT_TOLERANCE = TensorTolerance(atol=5e-5, rtol=2e-4)


def _write_s1_guard_evidence(path: Path, payload: Mapping[str, Any]) -> Path:
    artifact = dict(payload)
    artifact["sha256"] = hashlib.sha256(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_s1_guard_evidence_records_a_payload_digest(tmp_path: Path) -> None:
    path = _write_s1_guard_evidence(
        tmp_path / "batch16-guard.json",
        {"per_surface": {"backbone": {"bitwise": True}}},
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    digest_payload = dict(artifact)
    digest = digest_payload.pop("sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    assert artifact["per_surface"]["backbone"]["bitwise"] is True


def test_classification_loss_normalizes_by_matched_targets() -> None:
    logits = torch.tensor([[[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]]])
    targets: list[RADMTarget] = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.zeros(2, 4),
            "boxes_xyxy": torch.zeros(2, 4),
            "image_size_xyxy": torch.ones(4),
            "image_size_xyxy_tgt": torch.ones(2, 4),
        }
    ]
    indices = [(torch.tensor([True, False, False]), torch.tensor([1]))]

    actual = _classification_loss(
        logits,
        targets,
        indices,
        num_classes=2,
        alpha=0.25,
        gamma=2.0,
        no_object_weight=0.1,
    )
    target = torch.zeros(1, 3, 3)
    target[0, 0, 1] = 1
    expected = _sigmoid_focal_loss(
        logits.flatten(0, 1), target[..., :2].flatten(0, 1), 0.25, 2.0
    ).sum()

    torch.testing.assert_close(actual, expected)


@dataclass
class _ParityCase:
    """One source-generated fixture and its independently built package graph."""

    state: ReferenceTrainingState
    sample: dict[str, Any]
    package_batch: dict[str, torch.Tensor]
    fixture_tensors: dict[str, torch.Tensor]
    package: RADMDenoiser
    key_map: dict[str, str]
    module: RADMTrainingModule
    package_optimizer: torch.optim.Optimizer
    package_scheduler: torch.optim.lr_scheduler.LRScheduler
    state_before: str


@dataclass
class _PairedForward:
    """Source/package forward results run from one captured RNG state."""

    source: dict[str, Any]
    package_loss: torch.Tensor
    package_trace: dict[str, torch.Tensor]
    source_capture: dict[str, Any]
    package_capture: dict[str, Any]
    rng_before: str
    source_forward_rng: str
    package_forward_rng: str


@dataclass
class _PairedOptimizerStep:
    """One source/package backward, clipped update, and scheduler step."""

    paired: _PairedForward
    source_preclip: dict[str, torch.Tensor | None]
    package_preclip: dict[str, torch.Tensor | None]
    source_postclip: dict[str, torch.Tensor | None]
    package_postclip: dict[str, torch.Tensor | None]
    source_optimizer_state: dict[str, dict[str, Any]]
    package_optimizer_state: dict[str, dict[str, Any]]
    source_parameters_after: dict[str, torch.Tensor]
    package_parameters_after: dict[str, torch.Tensor]
    source_preclip_norm: float
    package_preclip_norm: float
    source_postclip_norm: float
    package_postclip_norm: float
    source_after_rng: str
    package_after_rng: str


def _assert_vendor_revision() -> None:
    actual = subprocess.run(
        ["git", "-C", "vendor/radm", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual == _VENDOR_REVISION, (actual, _VENDOR_REVISION)


def _require_reference_state(text_root: Path) -> ReferenceTrainingState:
    _assert_vendor_revision()
    try:
        return RADMReferenceAdapter(
            vendor_root=Path("vendor/radm"),
            text_feature_root=text_root,
            device=os.environ.get("RADM_REFERENCE_DEVICE", "cpu"),
        ).build_initialized_state()
    except ReferenceUnavailable as exc:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _build_source_fixture(
    state: ReferenceTrainingState,
    root: Path,
    *,
    fixture_name: str = "s1_fixture",
    annotations: list[dict[str, Any]] | None = None,
    feature_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.25, 0.25),
        (0.5, 1.0),
    ),
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Generate one source-shaped sample through the original mapper."""
    model = cast(Any, state.model)
    train_root = root / "train"
    train_root.mkdir(exist_ok=True)
    image_path = root / f"{fixture_name}.png"
    pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(pixels, mode="RGB").save(image_path)
    features = [
        torch.linspace(start, end, 768, dtype=torch.float32).reshape(1, 768)
        for start, end in feature_ranges
    ]
    torch.save({"feats": features}, train_root / f"{fixture_name}_feats.pth")
    if annotations is None:
        annotations = [
            {
                "bbox": [8, 8, 24, 24],
                "bbox_mode": 0,
                "category_id": 0,
                "iscrowd": 0,
            },
            {
                "bbox": [32, 32, 16, 16],
                "bbox_mode": 0,
                "category_id": 3,
                "iscrowd": 0,
            },
        ]

    with _vendor_import_root(Path("vendor/radm")), _legacy_pillow_compat():
        import importlib

        mapper = importlib.import_module("RADM.dataset_mapper").RADMDatasetMapper(
            state.config, is_train=True
        )
        sample = mapper(
            {
                "file_name": str(image_path),
                "height": 64,
                "width": 64,
                "annotations": annotations,
            }
        )

    images, image_scales = model.preprocess_image([sample])
    instances = sample["instances"].to(model.device)
    absolute_boxes = instances.gt_boxes.tensor.to(dtype=torch.float32)
    normalized_boxes = absolute_boxes / image_scales[0]
    labels = instances.gt_classes.to(dtype=torch.long)
    text_features = (
        sample["text_fea"]["feats"].to(model.device).unsqueeze(0).contiguous()
    )
    text_mask = sample["text_mask"].to(model.device).unsqueeze(0).contiguous()
    assert labels.tolist() == [
        int(annotation["category_id"]) for annotation in annotations
    ]
    assert int(labels.max()) < state.effective.num_classes
    assert state.effective.vocabulary_size == 5
    assert state.effective.predicted_class_id_to_label == {
        0: "Logo",
        1: "文字",
        2: "衬底",
        3: "符号元素",
    }
    package_batch = {
        "images": images.tensor,
        "image_scales": image_scales,
        "forward_image_scales": image_scales.new_tensor(
            (images.tensor.shape[-2], images.tensor.shape[-1])
        ).repeat(1, 2),
        "boxes_xyxy": normalized_boxes.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "mask": torch.ones(1, labels.numel(), dtype=torch.bool),
        "text_features": text_features,
        "text_mask": text_mask,
    }
    fixture_tensors = {
        "prepared_image": images.tensor,
        "boxes_xyxy": normalized_boxes,
        "labels": labels,
        "text_features": text_features,
        "text_mask": text_mask,
        "image_scales": image_scales,
    }
    return sample, package_batch, fixture_tensors


def _build_parity_case(
    state: ReferenceTrainingState,
    root: Path,
    *,
    fixture_name: str = "s1_fixture",
    annotations: list[dict[str, Any]] | None = None,
    feature_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.25, 0.25),
        (0.5, 1.0),
    ),
) -> _ParityCase:
    """Build the reviewed source/package graphs and one source-shaped batch."""
    sample, package_batch, fixture_tensors = _build_source_fixture(
        state,
        root,
        fixture_name=fixture_name,
        annotations=annotations,
        feature_ranges=feature_ranges,
    )
    package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs()))
    key_map = build_reviewed_state_key_map(state.model, package)
    copy_reviewed_state_dict(
        state.model,
        package,
        key_map,
        allowlist=state.reviewed_state_allowlist,
    )
    package.to(next(state.model.parameters()).device)
    state_before = _mapped_state_digest(state.model, package, key_map)
    module = RADMTrainingModule(
        config=package.radm_config,
        model=package,
        effective=state.effective,
    )
    configured = cast(dict[str, Any], module.configure_optimizers())
    package_optimizer = cast(torch.optim.Optimizer, configured["optimizer"])
    package_scheduler = cast(
        torch.optim.lr_scheduler.LRScheduler,
        cast(dict[str, Any], configured["lr_scheduler"])["scheduler"],
    )
    return _ParityCase(
        state=state,
        sample=sample,
        package_batch=package_batch,
        fixture_tensors=fixture_tensors,
        package=package,
        key_map=key_map,
        module=module,
        package_optimizer=package_optimizer,
        package_scheduler=package_scheduler,
        state_before=state_before,
    )


_REAL_BACKBONE_IMAGE_IDS = (
    41061,
    14431,
    21733,
    52182,
    35920,
    79726,
    9908,
    67798,
    8097,
    6364,
    251,
    1140,
    34124,
    59245,
    55077,
    27566,
)


@pytest.mark.vendor_parity
def test_s1_radm_real_batch16_backbone_bitwise_parity() -> None:
    """Compare the five source/package backbone outputs on the R-50 fixture."""
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    text_root = data_root / "text_features"
    weights_path = Path(
        os.environ.get("RADM_R50_WEIGHTS", ".cache/radm/weights/R-50.pkl")
    )
    evidence_path = Path(
        os.environ.get(
            "RADM_S1_BATCH16_EVIDENCE_PATH", ".cache/radm/s1/batch16_guard.json"
        )
    )
    if not data_root.exists() or not text_root.exists() or not weights_path.is_file():
        message = f"CGL/R-50 assets are missing: {data_root}, {weights_path}"
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    state = RADMReferenceAdapter(
        vendor_root=Path("vendor/radm"),
        dataset_root=data_root,
        text_feature_root=text_root,
        device=os.environ.get("RADM_REFERENCE_DEVICE", "cpu"),
    ).build_initialized_state()
    with _vendor_import_root(Path("vendor/radm")), _legacy_pillow_compat():
        checkpointer = importlib.import_module(
            "detectron2.checkpoint"
        ).DetectionCheckpointer(state.model)
        checkpointer.load(str(weights_path))
        data = importlib.import_module("detectron2.data")
        mapper_class = importlib.import_module("RADM.dataset_mapper").RADMDatasetMapper
        mapper = mapper_class(state.config, is_train=True)
        rows = data.DatasetCatalog.get("layout_train")
        by_id = {int(row["image_id"]): row for row in rows}
        missing = [
            image_id for image_id in _REAL_BACKBONE_IMAGE_IDS if image_id not in by_id
        ]
        if missing:
            pytest.fail(f"recorded CGL batch IDs are missing: {missing}")
        apply_determinism(DeterminismConfig(seed=_SEED))
        batch = [
            mapper(copy.deepcopy(by_id[image_id]))
            for image_id in _REAL_BACKBONE_IMAGE_IDS
        ]

    package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs())).to(
        next(state.model.parameters()).device
    )
    key_map = build_reviewed_state_key_map(state.model, package)
    copy_reviewed_state_dict(
        state.model,
        package,
        key_map,
        allowlist=state.reviewed_state_allowlist,
    )
    state.model.train()
    package.train()
    with torch.no_grad():
        images, _ = cast(Any, state.model).preprocess_image(batch)
        source_features = list(cast(Any, state.model).backbone(images.tensor).values())
        package_features = list(
            cast(Any, package).backbone.body(images.tensor).values()
        )
    backbone_levels = [
        torch.equal(source_value, package_value)
        for source_value, package_value in zip(
            source_features, package_features, strict=False
        )
    ]
    backbone_bitwise = len(source_features) == len(package_features) == 5 and all(
        backbone_levels
    )
    backbone_shape_counts = (len(source_features), len(package_features))
    backbone_hashes = {
        f"level_{index}": {
            "source": tensor_sha256(source_value),
            "package": tensor_sha256(package_value),
            "bitwise": bitwise,
        }
        for index, (source_value, package_value, bitwise) in enumerate(
            zip(source_features, package_features, backbone_levels, strict=False)
        )
    }
    del images, source_features, package_features
    if next(state.model.parameters()).is_cuda:
        torch.cuda.empty_cache()

    from test_radm_300_lockstep import _package_batch

    package_batch = _package_batch(batch, state.model, state.effective)
    module = RADMTrainingModule(
        config=package.radm_config,
        model=package,
        effective=state.effective,
    )
    source_capture, source_handles = _install_trace_hooks(state.model)
    package_capture, package_handles = _install_trace_hooks(package)
    source_prepare: dict[str, Any] = {}
    source_loss_surfaces: dict[str, torch.Tensor] = {}
    original_prepare_targets = cast(Any, state.model).prepare_targets

    def capture_prepare_targets(targets: Any) -> Any:
        prepared = original_prepare_targets(targets)
        source_prepare["targets"] = _capture_cpu_tree(prepared[0])
        source_prepare["diffusion_input"] = (
            prepared[1].detach().to(device="cpu", copy=True)
        )
        return prepared

    try:
        state.model.train()
        module.train()
        step_rng = capture_rng_state()
        with patch.object(state.model, "prepare_targets", new=capture_prepare_targets):
            with torch.no_grad():
                source_output = state.model(batch)
        source_loss_surfaces = {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in source_output.items()
        }
        source_total = source_loss_surfaces["loss_ce"] * 0
        for value in source_loss_surfaces.values():
            source_total = source_total + value
        source_loss_surfaces["train_loss"] = source_total
        source_forward_rng = _rng_digest(capture_rng_state())
        restore_rng_state(step_rng)
        with torch.no_grad():
            module._compute_step_loss(package_batch, record_trace=True)
        package_forward_rng = _rng_digest(capture_rng_state())
    finally:
        for handle in (*source_handles, *package_handles):
            handle.remove()

    rng_bitwise = source_forward_rng == package_forward_rng
    package_diffusion_input = (
        module.latest_step_trace["diffusion_input"].detach().to(device="cpu", copy=True)
    )
    diffusion_input_bitwise = torch.equal(
        source_prepare["diffusion_input"], package_diffusion_input
    )
    stage0_boxes_bitwise = torch.equal(
        source_capture["block_input_boxes"], package_capture["block_input_boxes"]
    )
    source_logits = source_capture["head_logits"]
    source_boxes = source_capture["head_boxes"]
    package_logits = package_capture["head_logits"]
    package_boxes = package_capture["head_boxes"]
    source_targets = source_prepare["targets"]
    package_targets: list[RADMTarget] = []
    for index in range(len(batch)):
        valid = package_batch["mask"][index]
        normalized_boxes = (
            package_batch["boxes_xyxy"][index][valid]
            .detach()
            .to(device="cpu", copy=True)
        )
        scale = (
            package_batch["image_scales"][index].detach().to(device="cpu", copy=True)
        )
        package_targets.append(
            {
                "labels": package_batch["labels"][index][valid]
                .detach()
                .to(device="cpu", copy=True),
                "boxes": _xyxy_to_cxcywh(normalized_boxes),
                "boxes_xyxy": normalized_boxes * scale,
                "image_size_xyxy": scale,
                "image_size_xyxy_tgt": scale.expand(normalized_boxes.shape[0], -1),
            }
        )
    assignment_surfaces: list[dict[str, Any]] = []
    for head_index in range(source_logits.shape[0]):
        source_matches, _ = cast(Any, state.model).criterion.matcher(
            {
                "pred_logits": source_logits[head_index],
                "pred_boxes": source_boxes[head_index],
            },
            source_targets,
        )
        for batch_index in range(len(batch)):
            package_match = _dynamic_k_match(
                package_logits[head_index, batch_index],
                package_boxes[head_index, batch_index],
                package_targets[batch_index],
                alpha=state.effective.alpha,
                gamma=state.effective.gamma,
                ota_k=state.effective.ota_k,
                class_weight=state.effective.class_weight,
                l1_weight=state.effective.l1_weight,
                giou_weight=state.effective.giou_weight,
            )
            selected_bitwise = torch.equal(
                source_matches[batch_index][0], package_match[0]
            )
            matched_bitwise = torch.equal(
                source_matches[batch_index][1], package_match[1]
            )
            assignment_surfaces.append(
                {
                    "head": head_index,
                    "batch": batch_index,
                    "selected_bitwise": selected_bitwise,
                    "matched_bitwise": matched_bitwise,
                    "source_selected_sha256": tensor_sha256(
                        source_matches[batch_index][0]
                    ),
                    "package_selected_sha256": tensor_sha256(package_match[0]),
                    "source_matched_sha256": tensor_sha256(
                        source_matches[batch_index][1]
                    ),
                    "package_matched_sha256": tensor_sha256(package_match[1]),
                }
            )

    assignments_bitwise = all(
        item["selected_bitwise"] and item["matched_bitwise"]
        for item in assignment_surfaces
    )
    package_loss_surfaces = {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in module.latest_step_trace.items()
        if name.startswith("loss_") or name == "train_loss"
    }
    loss_surface_comparisons: dict[str, dict[str, Any]] = {}
    loss_surfaces_bitwise = set(source_loss_surfaces) == set(package_loss_surfaces)
    if loss_surfaces_bitwise:
        for name in sorted(source_loss_surfaces):
            source_value = source_loss_surfaces[name]
            package_value = package_loss_surfaces[name]
            difference = (source_value - package_value).abs()
            loss_surface_comparisons[name] = {
                "source": tensor_sha256(source_value),
                "package": tensor_sha256(package_value),
                "max_abs_diff": float(difference.max().item()),
                "bitwise": torch.equal(source_value, package_value),
            }
        loss_surfaces_bitwise = all(
            comparison["bitwise"] for comparison in loss_surface_comparisons.values()
        )
    guard_payload = {
        "vendor_revision": _VENDOR_REVISION,
        "seed": _SEED,
        "batch_image_ids": list(_REAL_BACKBONE_IMAGE_IDS),
        "per_surface": {
            "backbone": {"bitwise": backbone_bitwise, "levels": backbone_hashes},
            "rng_after_forward": {"bitwise": rng_bitwise},
            "diffusion_input": {"bitwise": diffusion_input_bitwise},
            "stage0_absolute_boxes": {"bitwise": stage0_boxes_bitwise},
            "assignments": {
                "bitwise": assignments_bitwise,
                "surfaces": assignment_surfaces,
            },
            "loss_surfaces": {
                "bitwise": loss_surfaces_bitwise,
                "surfaces": loss_surface_comparisons,
            },
        },
        "state_sha256": _mapped_state_digest(state.model, package, key_map),
        "rng": {"source": source_forward_rng, "package": package_forward_rng},
    }
    _write_s1_guard_evidence(evidence_path, guard_payload)
    print(
        "S1 batch16 guard evidence "
        f"path={evidence_path} sha256={hashlib.sha256(evidence_path.read_bytes()).hexdigest()} "
        f"payload_sha256={json.loads(evidence_path.read_text(encoding='utf-8'))['sha256']}"
    )
    assert backbone_shape_counts == (5, 5)
    assert backbone_bitwise
    assert rng_bitwise
    assert diffusion_input_bitwise
    assert stage0_boxes_bitwise
    assert assignments_bitwise
    assert loss_surfaces_bitwise


def _build_s3_followup_fixtures(
    state: ReferenceTrainingState, root: Path
) -> list[tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    """Generate two additional source-mapped batches for the S3 stream."""
    return [
        _build_source_fixture(
            state,
            root,
            fixture_name="s3_fixture_1",
            annotations=[
                {
                    "bbox": [4, 28, 24, 48],
                    "bbox_mode": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                },
                {
                    "bbox": [36, 4, 56, 24],
                    "bbox_mode": 0,
                    "category_id": 2,
                    "iscrowd": 0,
                },
            ],
            feature_ranges=((-0.75, -0.25), (0.1, 0.6)),
        ),
        _build_source_fixture(
            state,
            root,
            fixture_name="s3_fixture_2",
            annotations=[
                {
                    "bbox": [12, 4, 28, 32],
                    "bbox_mode": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                },
                {
                    "bbox": [28, 28, 56, 48],
                    "bbox_mode": 0,
                    "category_id": 3,
                    "iscrowd": 0,
                },
            ],
            feature_ranges=((-0.4, 0.2), (0.35, 0.95)),
        ),
    ]


def _capture_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {key: _capture_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_capture_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_capture_cpu_tree(item) for item in value]
    return value


def _install_trace_hooks(model: torch.nn.Module) -> tuple[dict[str, Any], list[Any]]:
    captured: dict[str, Any] = {}
    dynamic_model = cast(Any, model)

    def tensor_summary(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, torch.Tensor):
            return None
        original_device = str(value.device)
        cpu_value = value.detach().to(device="cpu", copy=True)
        return {
            "shape": list(cpu_value.shape),
            "dtype": str(cpu_value.dtype),
            "device": original_device,
            "stride": list(cpu_value.stride()),
            "sha256": tensor_sha256(cpu_value),
        }

    def module_identity(module: torch.nn.Module) -> dict[str, Any]:
        return {
            "class": module.__class__.__qualname__,
            "approximate": getattr(module, "approximate", None),
        }

    time_mlp = dynamic_model.head.time_mlp
    captured["time_mlp_identity"] = {
        "activation": module_identity(time_mlp[2]),
    }
    captured["time_mlp_linear2_parameters"] = {
        "weight": tensor_summary(time_mlp[3].weight),
        "bias": tensor_summary(time_mlp[3].bias),
    }

    def trace_value(value: Any) -> Any:
        return _capture_cpu_tree(value)

    def first_output(value: Any) -> Any:
        if isinstance(value, tuple) and value:
            return value[0]
        return value

    def capture_head_inputs(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if hasattr(dynamic_model.head, "head_series"):
            text_index, mask_index, timestep_index = 3, 4, 5
            roi_scale = None
        else:
            text_index, mask_index, timestep_index = 2, 3, 4
            roi_scale = inputs[5]
        captured["head_inputs"] = {
            "text_features": trace_value(inputs[text_index]),
            "text_mask": trace_value(inputs[mask_index]),
            "timesteps": trace_value(inputs[timestep_index]),
            "roi_scale": trace_value(roi_scale),
        }

    def capture_time_embedding(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        captured["time_embedding"] = trace_value(output)

    def capture_time_mlp_invocation(
        _module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        frames = traceback.extract_stack()[:-1]
        stack = [
            f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.name}"
            for frame in frames
            if frame.name
            not in {
                "_call_impl",
                "_wrapped_call_impl",
                "capture_time_mlp_invocation",
            }
            and "/torch/" not in frame.filename
        ]
        captured.setdefault("time_mlp_invocations", []).append(
            {
                "invocation": len(captured.get("time_mlp_invocations", [])),
                "input": tensor_summary(inputs[0] if inputs else None),
                "output": tensor_summary(output),
                "autocast": {
                    "cuda": torch.is_autocast_enabled("cuda"),
                    "cpu": torch.is_autocast_enabled("cpu"),
                },
                "grad_enabled": torch.is_grad_enabled(),
                "stack": stack[-6:],
            }
        )

    def capture_box_pooler(
        _module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not isinstance(output, torch.Tensor):
            return
        if not inputs:
            return
        boxes = inputs[1] if len(inputs) > 1 else None
        if isinstance(boxes, list):
            batch_size = len(boxes)
            proposal_count = len(boxes[0]) if batch_size else 0
            output = output.reshape(batch_size, proposal_count, *output.shape[1:])
        if "stage0_roi_features" not in captured:
            captured["stage0_roi_features"] = trace_value(output)

    def capture_backbone(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        captured["backbone"] = {
            str(name): value.detach().to(device="cpu", copy=True)
            for name, value in output.items()
        }

    def capture_head(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        captured["head_logits"] = output[0].detach().to(device="cpu", copy=True)
        captured["head_boxes"] = output[1].detach().to(device="cpu", copy=True)

    def capture_block(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if hasattr(dynamic_model.head, "head_series"):
            roi_index, input_index, proposal_index = 0, 1, 5
        else:
            roi_index, input_index, proposal_index = 1, 2, 0
        if "stage0_roi_features" not in captured:
            captured["stage0_roi_features"] = trace_value(_inputs[roi_index])
        captured["block_input_boxes"] = (
            _inputs[input_index].detach().to(device="cpu", copy=True)
        )
        captured.setdefault(
            "stage0_initial_proposal_features", trace_value(_inputs[proposal_index])
        )
        captured["block_logits"] = output[0].detach().to(device="cpu", copy=True)
        captured["block_boxes"] = output[1].detach().to(device="cpu", copy=True)
        captured["block_features"] = output[2].detach().to(device="cpu", copy=True)
        captured["stage0_output_logits"] = (
            output[0].detach().to(device="cpu", copy=True)
        )
        captured["stage0_output_boxes_absolute"] = (
            output[1].detach().to(device="cpu", copy=True)
        )
        captured["stage0_box_renewal"] = output[1].detach().to(device="cpu", copy=True)

    def capture_self_attn_input(
        _module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        if inputs:
            captured["stage0_self_attn_input"] = trace_value(inputs[0])
            captured["stage0_initial_proposal_features"] = trace_value(inputs[0])

    def capture_module_output(
        name: str,
        *,
        module: torch.nn.Module,
        output_index: int | None = None,
    ) -> Any:
        def capture(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            value = output[output_index] if output_index is not None else output
            captured[name] = trace_value(first_output(value))

        return module.register_forward_hook(capture)

    first_block = (
        dynamic_model.head.head_series[0]
        if hasattr(dynamic_model.head, "head_series")
        else dynamic_model.head.blocks[0]
    )
    handles = [
        dynamic_model.head.register_forward_pre_hook(capture_head_inputs),
        dynamic_model.head.time_mlp.register_forward_hook(capture_time_embedding),
        dynamic_model.head.time_mlp.register_forward_hook(capture_time_mlp_invocation),
        dynamic_model.backbone.register_forward_hook(capture_backbone),
        dynamic_model.head.register_forward_hook(capture_head),
        first_block.register_forward_hook(capture_block),
    ]
    if hasattr(dynamic_model.head, "box_pooler"):
        handles.append(
            dynamic_model.head.box_pooler.register_forward_hook(capture_box_pooler)
        )
    handles.extend(
        [
            first_block.self_attn.register_forward_pre_hook(capture_self_attn_input),
            capture_module_output(
                "time_sinusoidal", module=dynamic_model.head.time_mlp[0]
            ),
            capture_module_output(
                "time_linear1", module=dynamic_model.head.time_mlp[1]
            ),
            capture_module_output(
                "time_activation", module=dynamic_model.head.time_mlp[2]
            ),
            capture_module_output(
                "time_linear2", module=dynamic_model.head.time_mlp[3]
            ),
            capture_module_output(
                "stage0_self_attn_output", module=first_block.self_attn
            ),
            capture_module_output(
                "stage0_inst_interact_output", module=first_block.inst_interact
            ),
            capture_module_output(
                "stage0_vis_text_att_output", module=first_block.vis_text_att
            ),
            capture_module_output(
                "stage0_time_scale_shift", module=first_block.block_time_mlp
            ),
            capture_module_output(
                "stage0_cls_tower_output", module=first_block.cls_module[-1]
            ),
            capture_module_output(
                "stage0_reg_tower_output", module=first_block.reg_module[-1]
            ),
            capture_module_output(
                "stage0_bboxes_delta", module=first_block.bboxes_delta
            ),
        ]
    )
    return captured, handles


def _run_reference_step(
    state: ReferenceTrainingState, sample: dict[str, Any]
) -> dict[str, Any]:
    """Execute the original pre-optimizer forward and weighted criterion."""
    model = cast(Any, state.model)
    model.train()
    images, image_scales = model.preprocess_image([sample])
    features = model.backbone(images.tensor)
    feature_list = [features[name] for name in state.config.MODEL.ROI_HEADS.IN_FEATURES]
    text_features = sample["text_fea"]["feats"].to(model.device).unsqueeze(0)
    text_mask = sample["text_mask"].to(model.device).unsqueeze(0)
    instances = [sample["instances"].to(model.device)]
    targets, diffusion_input, noise, timesteps = model.prepare_targets(instances)
    timesteps = timesteps.squeeze(-1)
    normalized_diffusion_input = diffusion_input.to(torch.float32)
    absolute_diffusion_input = diffusion_input * image_scales[:, None, :]
    outputs_class, outputs_coord = model.head(
        feature_list,
        absolute_diffusion_input,
        normalized_diffusion_input,
        text_features,
        text_mask,
        timesteps,
        None,
    )
    output: dict[str, Any] = {
        "pred_logits": outputs_class[-1],
        "pred_boxes": outputs_coord[-1],
        "aux_outputs": [
            {"pred_logits": logits, "pred_boxes": boxes}
            for logits, boxes in zip(
                outputs_class[:-1], outputs_coord[:-1], strict=True
            )
        ],
    }
    losses = model.criterion(output, targets)
    for name in tuple(losses):
        if name in model.criterion.weight_dict:
            losses[name] *= model.criterion.weight_dict[name]
    total = outputs_class[-1].sum() * 0
    for value in losses.values():
        total = total + value
    assignments = []
    for logits, boxes in zip(outputs_class, outputs_coord, strict=True):
        matched, _ = model.criterion.matcher(
            {"pred_logits": logits, "pred_boxes": boxes}, targets
        )
        assignments.append((matched[0][0], matched[0][1]))
    return {
        "prepared_image": images.tensor.detach(),
        "image_scales": image_scales.detach(),
        "targets_boxes": targets[0]["boxes"].detach(),
        "targets_boxes_xyxy": targets[0]["boxes_xyxy"].detach(),
        "labels": targets[0]["labels"].detach(),
        "text_features": text_features.detach(),
        "text_mask": text_mask.detach(),
        "timesteps": timesteps.detach(),
        "noise": noise.detach(),
        "diffusion_input": diffusion_input.detach(),
        "outputs_class": outputs_class.detach(),
        "outputs_coord": outputs_coord.detach(),
        "losses": {name: value.detach() for name, value in losses.items()},
        "total": total.detach(),
        "total_graph": total,
        "assignments": assignments,
    }


def _run_paired_forward(
    case: _ParityCase,
    *,
    capture_traces: bool,
    sample: dict[str, Any] | None = None,
    package_batch: dict[str, torch.Tensor] | None = None,
) -> _PairedForward:
    """Run source and package forwards from the same captured RNG state."""
    source_sample = case.sample if sample is None else sample
    package_inputs = case.package_batch if package_batch is None else package_batch
    source_capture: dict[str, Any] = {}
    package_capture: dict[str, Any] = {}
    handles: list[Any] = []
    if capture_traces:
        source_capture, source_handles = _install_trace_hooks(case.state.model)
        package_capture, package_handles = _install_trace_hooks(case.package)
        handles.extend((*source_handles, *package_handles))
    try:
        case.state.model.train()
        case.module.train()
        step_rng = capture_rng_state()
        rng_before = _rng_digest(step_rng)
        source = _run_reference_step(case.state, source_sample)
        source_forward_rng = _rng_digest(capture_rng_state())
        restore_rng_state(step_rng)
        package_loss = case.module.training_step(package_inputs, 0)
        package_forward_rng = _rng_digest(capture_rng_state())
        package_trace = _capture_cpu_tree(case.module.latest_step_trace)
    finally:
        for handle in handles:
            handle.remove()
    return _PairedForward(
        source=source,
        package_loss=package_loss,
        package_trace=package_trace,
        source_capture=source_capture,
        package_capture=package_capture,
        rng_before=rng_before,
        source_forward_rng=source_forward_rng,
        package_forward_rng=package_forward_rng,
    )


def _package_targets(batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    from radm.training.lightning_module import _xyxy_to_cxcywh

    valid = batch["mask"][0]
    boxes = batch["boxes_xyxy"][0][valid]
    scale = batch["image_scales"][0]
    return [
        {
            "labels": batch["labels"][0][valid],
            "boxes": _xyxy_to_cxcywh(boxes),
            "boxes_xyxy": boxes * scale,
            "image_size_xyxy": scale,
            "image_size_xyxy_tgt": scale.expand(boxes.shape[0], -1),
        }
    ]


def _rng_digest(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state.python).encode())
    digest.update(repr(state.numpy[0]).encode())
    digest.update(state.numpy[1].tobytes())
    digest.update(state.torch_cpu.numpy().tobytes())
    for cuda_state in state.torch_cuda:
        digest.update(cuda_state.numpy().tobytes())
    return digest.hexdigest()


def _mapped_state_digest(
    reference_model: torch.nn.Module,
    package_model: torch.nn.Module,
    key_map: dict[str, str],
) -> str:
    """Hash the reviewed state map while requiring exact tensor equality."""
    reference_state = reference_model.state_dict()
    package_state = package_model.state_dict()
    digest = hashlib.sha256()
    for package_key in sorted(key_map):
        reference_key = key_map[package_key]
        reference_tensor = reference_state[reference_key]
        package_tensor = package_state[package_key]
        assert reference_tensor.dtype == package_tensor.dtype, package_key
        assert reference_tensor.shape == package_tensor.shape, package_key
        assert torch.equal(reference_tensor, package_tensor), package_key
        digest.update(package_key.encode())
        digest.update(tensor_sha256(package_tensor).encode())
    return digest.hexdigest()


def _named_optimizer_parameters(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, torch.nn.Parameter]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result: dict[str, torch.nn.Parameter] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names[id(parameter)]
            assert name not in result, name
            result[name] = parameter
    return result


def _snapshot_gradients(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in _named_optimizer_parameters(model, optimizer).items()
    }


def _snapshot_parameters(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in _named_optimizer_parameters(model, optimizer).items()
    }


def _snapshot_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, Any]]:
    parameters = _named_optimizer_parameters(model, optimizer)
    return {
        name: {
            state_name: (
                value.detach().clone() if isinstance(value, torch.Tensor) else value
            )
            for state_name, value in optimizer.state.get(parameter, {}).items()
        }
        for name, parameter in parameters.items()
    }


def _copy_optimizer_state_for_sync(
    source_optimizer: torch.optim.Optimizer,
    package_optimizer: torch.optim.Optimizer,
) -> None:
    """Copy optimizer state without sharing tensor storage between paths."""
    package_optimizer.load_state_dict(copy.deepcopy(source_optimizer.state_dict()))
    source_parameters = tuple(
        parameter
        for group in source_optimizer.param_groups
        for parameter in group["params"]
    )
    package_parameters = tuple(
        parameter
        for group in package_optimizer.param_groups
        for parameter in group["params"]
    )
    assert len(source_parameters) == len(package_parameters)
    for source_parameter, package_parameter in zip(
        source_parameters, package_parameters, strict=True
    ):
        source_state = source_optimizer.state.get(source_parameter, {})
        package_state = package_optimizer.state.get(package_parameter, {})
        assert source_state.keys() == package_state.keys()
        for state_name, source_value in source_state.items():
            package_value = package_state[state_name]
            if isinstance(source_value, torch.Tensor):
                assert isinstance(package_value, torch.Tensor)
                assert source_value.data_ptr() != package_value.data_ptr(), state_name


def _digest_named_tensors(values: Mapping[str, torch.Tensor]) -> str:
    """Hash named tensor values for compact optimizer/checkpoint evidence."""
    payload = {name: tensor_sha256(value) for name, value in sorted(values.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _digest_optimizer_snapshot(
    values: Mapping[str, Mapping[str, Any]],
    name_map: Mapping[str, str] | None = None,
) -> str:
    """Hash named optimizer state snapshots, including scalar state values."""
    digest = hashlib.sha256()
    names = sorted(values, key=lambda value: (name_map or {}).get(value, value))
    for name in names:
        canonical_name = (name_map or {}).get(name, name)
        for state_name in sorted(values[name]):
            digest.update(f"{canonical_name}.{state_name}".encode())
            value = values[name][state_name]
            if isinstance(value, torch.Tensor):
                digest.update(tensor_sha256(value).encode())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def _digest_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> str:
    """Hash scheduler state values for the S3 trajectory record."""
    return hashlib.sha256(
        json.dumps(scheduler.state_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()


def _digest_mapped_model_state(
    reference_model: torch.nn.Module,
    package_model: torch.nn.Module,
    key_map: Mapping[str, str],
    *,
    side: str,
) -> str:
    """Hash model state with reviewed source names on either side."""
    reference_state = reference_model.state_dict()
    package_state = package_model.state_dict()
    digest = hashlib.sha256()
    for package_key in sorted(key_map):
        reference_key = key_map[package_key]
        digest.update(reference_key.encode())
        value = (
            reference_state[reference_key]
            if side == "source"
            else package_state[package_key]
        )
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def _s3_state_hashes(case: _ParityCase) -> dict[str, dict[str, str]]:
    """Return source/package model, optimizer, and scheduler state hashes."""
    return {
        "model": {
            "source": _digest_mapped_model_state(
                case.state.model, case.package, case.key_map, side="source"
            ),
            "package": _digest_mapped_model_state(
                case.state.model, case.package, case.key_map, side="package"
            ),
        },
        "optimizer": {
            "source": _digest_optimizer_snapshot(
                _snapshot_optimizer_state(case.state.model, case.state.optimizer)
            ),
            "package": _digest_optimizer_snapshot(
                _snapshot_optimizer_state(case.package, case.package_optimizer),
                {
                    package_name: reference_name
                    for package_name, reference_name in case.key_map.items()
                },
            ),
        },
        "scheduler": {
            "source": _digest_scheduler_state(case.state.scheduler),
            "package": _digest_scheduler_state(case.package_scheduler),
        },
    }


def _synchronize_s3_state(case: _ParityCase) -> dict[str, Any]:
    """Synchronize package state from source before the next S3 batch."""
    before = _s3_state_hashes(case)
    copy_reviewed_state_dict(
        case.state.model,
        case.package,
        case.key_map,
        allowlist=case.state.reviewed_state_allowlist,
    )
    _copy_optimizer_state_for_sync(case.state.optimizer, case.package_optimizer)
    case.package_scheduler.load_state_dict(
        copy.deepcopy(case.state.scheduler.state_dict())
    )
    after = _s3_state_hashes(case)
    assert _mapped_state_digest(case.state.model, case.package, case.key_map)
    source_optimizer_state = _snapshot_optimizer_state(
        case.state.model, case.state.optimizer
    )
    package_optimizer_state = _snapshot_optimizer_state(
        case.package, case.package_optimizer
    )
    _compare_mapped_optimizer_state(
        source_optimizer_state,
        package_optimizer_state,
        case.key_map,
        tolerance=TensorTolerance(atol=0.0, rtol=0.0),
        stage="S3 state sync",
    )
    assert case.state.scheduler.last_epoch == case.package_scheduler.last_epoch
    assert case.state.scheduler.get_last_lr() == pytest.approx(
        case.package_scheduler.get_last_lr(), abs=0.0, rel=0.0
    )
    assert after["model"]["source"] == after["model"]["package"]
    assert after["optimizer"]["source"] == after["optimizer"]["package"]
    return {
        "before": before,
        "after": after,
        "zero_diff": True,
        "optimizer_tensor_storage_independent": True,
    }


def _compare_mapped_tensors(
    reference: Mapping[str, torch.Tensor | None],
    package: Mapping[str, torch.Tensor | None],
    key_map: dict[str, str],
    *,
    label: str,
    tolerance: TensorTolerance,
    stage: str = "S2",
    enforce: bool = True,
    divergences: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, float]]:
    reverse_map = {
        reference_key: package_key for package_key, reference_key in key_map.items()
    }
    reference_tensors: dict[str, torch.Tensor] = {}
    package_tensors: dict[str, torch.Tensor] = {}
    for reference_key in sorted(reference):
        package_key = reverse_map[reference_key]
        reference_value = reference[reference_key]
        package_value = package[package_key]
        if (reference_value is None) != (package_value is None):
            message = f"{label}.{reference_key}.grad_presence"
            if enforce:
                pytest.fail(f"{stage} first divergence: {message}")
            if divergences is not None:
                divergences.append(
                    {
                        "stage": stage,
                        "surface": message,
                        "max_abs_diff": float("inf"),
                        "max_rel_diff": float("inf"),
                    }
                )
            continue
        if reference_value is None or package_value is None:
            continue
        reference_tensors[f"{label}.{reference_key}"] = reference_value
        package_tensors[f"{label}.{reference_key}"] = package_value
    if not reference_tensors:
        return {}
    return _assert_trace(
        reference_tensors,
        package_tensors,
        float_names=set(reference_tensors),
        tolerance=tolerance,
        stage=stage,
        enforce=enforce,
        divergences=divergences,
    )


def _compare_mapped_optimizer_state(
    reference: dict[str, dict[str, Any]],
    package: dict[str, dict[str, Any]],
    key_map: dict[str, str],
    *,
    tolerance: TensorTolerance,
    stage: str = "S2",
    enforce: bool = True,
    divergences: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, float]]:
    reverse_map = {
        reference_key: package_key for package_key, reference_key in key_map.items()
    }
    reference_tensors: dict[str, torch.Tensor] = {}
    package_tensors: dict[str, torch.Tensor] = {}
    for reference_key in sorted(reference):
        package_key = reverse_map[reference_key]
        reference_values = reference[reference_key]
        package_values = package[package_key]
        if reference_values.keys() != package_values.keys():
            message = f"optimizer_state.{reference_key}.keys"
            if enforce:
                pytest.fail(f"{stage} first divergence: {message}")
            if divergences is not None:
                divergences.append(
                    {
                        "stage": stage,
                        "surface": message,
                        "max_abs_diff": float("inf"),
                        "max_rel_diff": float("inf"),
                    }
                )
            continue
        for state_name in sorted(reference_values):
            reference_value = reference_values[state_name]
            package_value = package_values[state_name]
            name = f"optimizer_state.{reference_key}.{state_name}"
            if isinstance(reference_value, torch.Tensor):
                if not isinstance(package_value, torch.Tensor):
                    message = f"{name}.type"
                    if enforce:
                        pytest.fail(f"{stage} first divergence: {message}")
                    if divergences is not None:
                        divergences.append(
                            {
                                "stage": stage,
                                "surface": message,
                                "max_abs_diff": float("inf"),
                                "max_rel_diff": float("inf"),
                            }
                        )
                    continue
                reference_tensors[name] = reference_value
                package_tensors[name] = package_value
            elif reference_value != package_value:
                if enforce:
                    pytest.fail(f"{stage} first divergence: {name}")
                if divergences is not None:
                    divergences.append(
                        {
                            "stage": stage,
                            "surface": name,
                            "max_abs_diff": float("inf"),
                            "max_rel_diff": float("inf"),
                        }
                    )
    if not reference_tensors:
        return {}
    return _assert_trace(
        reference_tensors,
        package_tensors,
        float_names=set(reference_tensors),
        tolerance=tolerance,
        stage=stage,
        enforce=enforce,
        divergences=divergences,
    )


def _gradient_norm(gradients: dict[str, torch.Tensor | None]) -> float:
    values = [value.flatten() for value in gradients.values() if value is not None]
    if not values:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(values)))


def _run_paired_optimizer_step(
    case: _ParityCase,
    *,
    sample: dict[str, Any] | None = None,
    package_batch: dict[str, torch.Tensor] | None = None,
) -> _PairedOptimizerStep:
    """Run one matched source/package optimizer and step-cadence update."""
    paired = _run_paired_forward(
        case,
        capture_traces=False,
        sample=sample,
        package_batch=package_batch,
    )
    state = case.state
    state.optimizer.zero_grad()
    paired.source["total_graph"].backward()
    source_preclip = _snapshot_gradients(state.model, state.optimizer)
    source_preclip_norm = _gradient_norm(source_preclip)
    state.optimizer.step()
    source_postclip = _snapshot_gradients(state.model, state.optimizer)
    source_postclip_norm = _gradient_norm(source_postclip)
    source_optimizer_state = _snapshot_optimizer_state(state.model, state.optimizer)
    source_parameters_after = _snapshot_parameters(state.model, state.optimizer)
    state.scheduler.step()
    source_after_rng = _rng_digest(capture_rng_state())

    case.package_optimizer.zero_grad()
    paired.package_loss.backward()
    package_preclip = _snapshot_gradients(case.package, case.package_optimizer)
    package_preclip_norm = _gradient_norm(package_preclip)
    case.package_optimizer.step()
    package_postclip = _snapshot_gradients(case.package, case.package_optimizer)
    package_postclip_norm = _gradient_norm(package_postclip)
    package_optimizer_state = _snapshot_optimizer_state(
        case.package, case.package_optimizer
    )
    package_parameters_after = _snapshot_parameters(
        case.package, case.package_optimizer
    )
    case.package_scheduler.step()
    package_after_rng = _rng_digest(capture_rng_state())
    return _PairedOptimizerStep(
        paired=paired,
        source_preclip=source_preclip,
        package_preclip=package_preclip,
        source_postclip=source_postclip,
        package_postclip=package_postclip,
        source_optimizer_state=source_optimizer_state,
        package_optimizer_state=package_optimizer_state,
        source_parameters_after=source_parameters_after,
        package_parameters_after=package_parameters_after,
        source_preclip_norm=source_preclip_norm,
        package_preclip_norm=package_preclip_norm,
        source_postclip_norm=source_postclip_norm,
        package_postclip_norm=package_postclip_norm,
        source_after_rng=source_after_rng,
        package_after_rng=package_after_rng,
    )


def _assert_optimizer_mapping(
    case: _ParityCase,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    """Assert the source/package optimizer parameter order and group defaults."""
    source_parameters = _named_optimizer_parameters(
        case.state.model, case.state.optimizer
    )
    package_parameters = _named_optimizer_parameters(
        case.package, case.package_optimizer
    )
    assert set(source_parameters) == {case.key_map[name] for name in package_parameters}
    assert list(source_parameters) == [
        case.key_map[name] for name in package_parameters
    ]
    assert len(case.state.optimizer.param_groups) == len(
        case.package_optimizer.param_groups
    )
    source_group_by_name = {
        name: (group["lr"], group["weight_decay"])
        for group in case.state.optimizer.param_groups
        for name in source_parameters
        if id(source_parameters[name]) in {id(value) for value in group["params"]}
    }
    package_group_by_source_name = {
        case.key_map[name]: (group["lr"], group["weight_decay"])
        for group in case.package_optimizer.param_groups
        for name in package_parameters
        if id(package_parameters[name]) in {id(value) for value in group["params"]}
    }
    assert source_group_by_name == package_group_by_source_name
    return source_parameters, package_parameters


def _mapped_learning_rates(
    case: _ParityCase,
    source_parameters: Mapping[str, torch.nn.Parameter],
    package_parameters: Mapping[str, torch.nn.Parameter],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Return source-named learning-rate vectors for the paired optimizers."""
    source_lrs_by_name = {
        name: group["lr"]
        for group in case.state.optimizer.param_groups
        for name in source_parameters
        if id(source_parameters[name]) in {id(value) for value in group["params"]}
    }
    package_lrs_by_source_name = {
        case.key_map[name]: group["lr"]
        for group in case.package_optimizer.param_groups
        for name in package_parameters
        if id(package_parameters[name]) in {id(value) for value in group["params"]}
    }
    lr_names = sorted(source_lrs_by_name)
    return (
        lr_names,
        torch.tensor(
            [source_lrs_by_name[name] for name in lr_names], dtype=torch.float64
        ),
        torch.tensor(
            [package_lrs_by_source_name[name] for name in lr_names], dtype=torch.float64
        ),
    )


def _compare_paired_optimizer_step(
    case: _ParityCase,
    step: _PairedOptimizerStep,
    source_parameters: Mapping[str, torch.nn.Parameter],
    package_parameters: Mapping[str, torch.nn.Parameter],
    *,
    stage: str,
    enforce: bool = True,
    divergences: list[dict[str, object]] | None = None,
) -> tuple[
    dict[str, dict[str, dict[str, float]]], list[str], torch.Tensor, torch.Tensor
]:
    """Compare the common S2/S3 backward, update, and scheduler surfaces."""
    paired = step.paired
    source_losses = {**paired.source["losses"], "train_loss": paired.source["total"]}
    package_losses = {
        name: value
        for name, value in paired.package_trace.items()
        if name.startswith("loss_") or name == "train_loss"
    }
    errors: dict[str, dict[str, dict[str, float]]] = {
        "loss": _assert_trace(
            source_losses,
            package_losses,
            float_names=set(source_losses),
            tolerance=_S2_FLOAT_TOLERANCE,
            stage=stage,
            enforce=enforce,
            divergences=divergences,
        ),
        "pre_clip_gradients": _compare_mapped_tensors(
            step.source_preclip,
            step.package_preclip,
            case.key_map,
            label="pre_clip_gradients",
            tolerance=_S2_FLOAT_TOLERANCE,
            stage=stage,
            enforce=enforce,
            divergences=divergences,
        ),
        "post_clip_gradients": _compare_mapped_tensors(
            step.source_postclip,
            step.package_postclip,
            case.key_map,
            label="post_clip_gradients",
            tolerance=_S2_FLOAT_TOLERANCE,
            stage=stage,
            enforce=enforce,
            divergences=divergences,
        ),
        "parameters_after_step": _compare_mapped_tensors(
            step.source_parameters_after,
            step.package_parameters_after,
            case.key_map,
            label="parameters_after_step",
            tolerance=_S2_FLOAT_TOLERANCE,
            stage=stage,
            enforce=enforce,
            divergences=divergences,
        ),
        "optimizer_state": _compare_mapped_optimizer_state(
            step.source_optimizer_state,
            step.package_optimizer_state,
            case.key_map,
            tolerance=_S2_FLOAT_TOLERANCE,
            stage=stage,
            enforce=enforce,
            divergences=divergences,
        ),
    }
    errors["gradient_norm"] = _assert_trace(
        {
            "pre_clip_gradient_norm": torch.tensor(step.source_preclip_norm),
            "post_clip_gradient_norm": torch.tensor(step.source_postclip_norm),
        },
        {
            "pre_clip_gradient_norm": torch.tensor(step.package_preclip_norm),
            "post_clip_gradient_norm": torch.tensor(step.package_postclip_norm),
        },
        float_names={"pre_clip_gradient_norm", "post_clip_gradient_norm"},
        tolerance=_S2_FLOAT_TOLERANCE,
        stage=stage,
        enforce=enforce,
        divergences=divergences,
    )
    lr_names, source_lrs, package_lrs = _mapped_learning_rates(
        case, source_parameters, package_parameters
    )
    errors["learning_rates"] = _assert_trace(
        {"learning_rates": source_lrs},
        {"learning_rates": package_lrs},
        float_names={"learning_rates"},
        tolerance=_S2_FLOAT_TOLERANCE,
        stage=stage,
        enforce=enforce,
        divergences=divergences,
    )
    if case.state.scheduler.last_epoch != case.package_scheduler.last_epoch:
        message = "scheduler.last_epoch"
        if enforce:
            pytest.fail(f"{stage} first divergence: {message}")
        if divergences is not None:
            divergences.append(
                {
                    "stage": stage,
                    "surface": message,
                    "max_abs_diff": float(
                        abs(
                            case.state.scheduler.last_epoch
                            - case.package_scheduler.last_epoch
                        )
                    ),
                    "max_rel_diff": float("inf"),
                }
            )
    errors["scheduler"] = _assert_trace(
        {
            "last_epoch": torch.tensor([case.state.scheduler.last_epoch]),
            "learning_rates": source_lrs,
        },
        {
            "last_epoch": torch.tensor([case.package_scheduler.last_epoch]),
            "learning_rates": package_lrs,
        },
        float_names={"learning_rates"},
        tolerance=_S2_FLOAT_TOLERANCE,
        stage=stage,
        enforce=enforce,
        divergences=divergences,
    )
    contract_checks = (
        ("forward_rng", paired.source_forward_rng == paired.package_forward_rng),
        ("after_step_rng", step.source_after_rng == step.package_after_rng),
        (
            "source_preclip_norm",
            step.source_preclip_norm >= case.state.effective.gradient_clip_norm,
        ),
        (
            "package_preclip_norm",
            step.package_preclip_norm >= case.state.effective.gradient_clip_norm,
        ),
        (
            "source_postclip_norm",
            step.source_postclip_norm <= case.state.effective.gradient_clip_norm + 1e-5,
        ),
        (
            "package_postclip_norm",
            step.package_postclip_norm
            <= case.state.effective.gradient_clip_norm + 1e-5,
        ),
    )
    for name, passed in contract_checks:
        if not passed:
            if enforce:
                pytest.fail(f"{stage} first divergence: {name}")
            if divergences is not None:
                divergences.append(
                    {
                        "stage": stage,
                        "surface": name,
                        "max_abs_diff": float("inf"),
                        "max_rel_diff": float("inf"),
                    }
                )
    return errors, lr_names, source_lrs, package_lrs


def _assert_trace(
    reference_tensors: dict[str, torch.Tensor],
    package_tensors: dict[str, torch.Tensor],
    *,
    float_names: set[str],
    tolerance: TensorTolerance = _FLOAT_TOLERANCE,
    stage: str = "S1",
    enforce: bool = True,
    divergences: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, float]]:
    tolerances = {name: tolerance for name in reference_tensors if name in float_names}
    reference = build_step_trace("original", reference_tensors)
    package = build_step_trace("package", package_tensors)
    report = compare_step_trace(reference, package, tolerances)
    for name in reference_tensors:
        if name not in package_tensors:
            continue
        if package_tensors[name].dtype != reference_tensors[name].dtype:
            if enforce:
                pytest.fail(f"{stage} first divergence: {name}.dtype")
            if divergences is not None:
                divergences.append(
                    {
                        "stage": stage,
                        "surface": f"{name}.dtype",
                        "max_abs_diff": float("inf"),
                        "max_rel_diff": float("inf"),
                    }
                )
    if not report.passed:
        first = next((item for item in report.comparisons if not item.passed), None)
        surface = first.name if first else report.missing
        max_abs = first.max_abs_diff if first else float("inf")
        max_rel = first.max_rel_diff if first else float("inf")
        if enforce:
            pytest.fail(
                f"{stage} first divergence: {surface}; "
                f"max_abs={max_abs}; max_rel={max_rel}; missing={report.missing}"
            )
        if divergences is not None:
            divergences.append(
                {
                    "stage": stage,
                    "surface": str(surface),
                    "max_abs_diff": float(max_abs),
                    "max_rel_diff": float(max_rel),
                }
            )
    return {
        item.name: {
            "max_abs_diff": item.max_abs_diff,
            "max_rel_diff": item.max_rel_diff,
            "atol": tolerances.get(item.name, TensorTolerance()).atol,
            "rtol": tolerances.get(item.name, TensorTolerance()).rtol,
        }
        for item in report.comparisons
    }


@pytest.mark.vendor_parity
def test_s1_radm_fixed_batch_pre_optimizer_parity() -> None:
    """Compare one source-generated batch before backward or optimizer state."""
    apply_determinism(DeterminismConfig(seed=_SEED))
    with tempfile.TemporaryDirectory() as temporary:
        state = _require_reference_state(Path(temporary))
        case = _build_parity_case(state, Path(temporary))
        paired = _run_paired_forward(case, capture_traces=True)
        source = paired.source
        package_batch = case.package_batch
        fixture_tensors = case.fixture_tensors
        package = case.package
        key_map = case.key_map
        source_capture = paired.source_capture
        package_capture = paired.package_capture
        state_before = case.state_before
        state_after = _mapped_state_digest(state.model, package, key_map)
        source_before = paired.rng_before
        source_after = paired.source_forward_rng
        assert state_before == state_after
        assert paired.source_forward_rng == paired.package_forward_rng
        source_tensors: dict[str, torch.Tensor] = {
            "prepared_image": source["prepared_image"],
            "image_scales": source["image_scales"],
            "targets_boxes": source["targets_boxes"],
            "targets_boxes_xyxy": source["targets_boxes_xyxy"],
            "labels": source["labels"],
            "text_features": source["text_features"],
            "text_mask": source["text_mask"],
            "timesteps": source["timesteps"],
            "noise": source["noise"],
            "diffusion_input": source["diffusion_input"],
        }
        package_trace = paired.package_trace
        package_tensors = {
            "prepared_image": package_batch["images"],
            "image_scales": package_batch["image_scales"],
            "targets_boxes": _package_targets(package_batch)[0]["boxes"],
            "targets_boxes_xyxy": _package_targets(package_batch)[0]["boxes_xyxy"],
            "labels": package_batch["labels"][0][package_batch["mask"][0]],
            "text_features": package_batch["text_features"],
            "text_mask": package_batch["text_mask"],
            "timesteps": package_trace["timestep"],
            "noise": package_trace["noise"],
            "diffusion_input": package_trace["diffusion_input"],
        }
        input_errors = _assert_trace(
            source_tensors,
            package_tensors,
            float_names={
                "prepared_image",
                "image_scales",
                "targets_boxes",
                "targets_boxes_xyxy",
                "text_features",
                "noise",
                "diffusion_input",
            },
        )
        torch.testing.assert_close(
            source["diffusion_input"],
            package_trace["diffusion_input"],
            rtol=0.0,
            atol=0.0,
        )
        consumed_features = tuple(state.config.MODEL.ROI_HEADS.IN_FEATURES)
        source_backbone = {
            name: source_capture["backbone"][name] for name in consumed_features
        }
        package_backbone = {
            name: package_capture["backbone"][name] for name in consumed_features
        }
        assert set(source_capture["backbone"]) == set((*consumed_features, "p6"))
        assert set(package_capture["backbone"]) == set(consumed_features)
        backbone_errors = _assert_trace(
            source_backbone,
            package_backbone,
            float_names=set(source_backbone),
        )
        from detectron2.structures import Boxes
        from detectron2.modeling.poolers import assign_boxes_to_levels

        source_absolute_input = (
            source["diffusion_input"] * source["image_scales"][:, None, :]
        )
        source_levels = assign_boxes_to_levels(
            [Boxes(source_absolute_input[0])], 2, 5, 224, 4
        )
        package_levels = package.head._assign_pooler_levels(
            source_absolute_input
        ).flatten()
        assert torch.equal(source_levels, package_levels), (
            source_levels.tolist(),
            package_levels.tolist(),
        )
        source_model = cast(Any, state.model)
        source_pooled = source_model.head.box_pooler(
            [source_capture["backbone"][name] for name in consumed_features],
            [Boxes(source_absolute_input[0])],
        )
        package_pooled = package.head._roi_features(
            package_capture["backbone"], source_absolute_input
        )
        roi_errors = _assert_trace(
            {"roi_features": source_pooled},
            {"roi_features": package_pooled.reshape_as(source_pooled)},
            float_names={"roi_features"},
        )
        source_logits = source_capture["head_logits"]
        package_logits = package_capture["head_logits"]
        source_boxes = source_capture["head_boxes"]
        package_boxes = package_capture["head_boxes"]
        _assert_trace(
            {
                "block_features": source_capture["block_features"],
                "block_boxes_absolute": source_capture["block_boxes"],
                "block_logits": source_capture["block_logits"],
            },
            {
                "block_features": package_capture["block_features"],
                "block_boxes_absolute": package_capture["block_boxes"],
                "block_logits": package_capture["block_logits"],
            },
            float_names={
                "block_logits",
                "block_boxes_absolute",
                "block_features",
            },
        )
        model_errors = _assert_trace(
            {
                "head_logits": source_logits,
                "head_boxes_absolute": source_boxes,
            },
            {
                "head_logits": package_logits,
                "head_boxes_absolute": package_boxes,
            },
            float_names={"head_logits", "head_boxes_absolute"},
        )

        source_losses = {**source["losses"], "train_loss": source["total"]}
        package_losses = {
            name: value
            for name, value in package_trace.items()
            if name.startswith("loss_") or name == "train_loss"
        }
        loss_errors = _assert_trace(
            source_losses,
            package_losses,
            float_names=set(source_losses),
        )

        package_targets = _package_targets(package_batch)
        assignment_errors: dict[str, dict[str, float]] = {}
        for index, ((source_indices, source_matched), (logits, boxes)) in enumerate(
            zip(source["assignments"], zip(package_logits, package_boxes), strict=True)
        ):
            package_indices = _dynamic_k_match(
                logits[0],
                boxes[0],
                cast(RADMTarget, package_targets[0]),
                alpha=state.effective.alpha,
                gamma=state.effective.gamma,
                ota_k=state.effective.ota_k,
                class_weight=state.effective.class_weight,
                l1_weight=state.effective.l1_weight,
                giou_weight=state.effective.giou_weight,
            )
            assignment_errors.update(
                _assert_trace(
                    {
                        f"assignment_{index}_selected": source_indices,
                        f"assignment_{index}_matched": source_matched,
                    },
                    {
                        f"assignment_{index}_selected": package_indices[0],
                        f"assignment_{index}_matched": package_indices[1],
                    },
                    float_names=set(),
                )
            )
            assert torch.equal(source_indices, package_indices[0])
            assert torch.equal(source_matched, package_indices[1])

        fixture_hash_payload = {
            name: tensor_sha256(value) for name, value in fixture_tensors.items()
        }
        errors = {
            "inputs": input_errors,
            "backbone": backbone_errors,
            "roi": roi_errors,
            "model": model_errors,
            "loss": loss_errors,
            "assignments": assignment_errors,
        }
        comparisons = [
            comparison for group in errors.values() for comparison in group.values()
        ]
        max_abs_diff = max(
            float(comparison["max_abs_diff"]) for comparison in comparisons
        )
        max_rel_diff = max(
            float(comparison["max_rel_diff"]) for comparison in comparisons
        )
        evidence = {
            "vendor_revision": _VENDOR_REVISION,
            "config_sha256": hashlib.sha256(
                Path(
                    "models/radm/configs/training/effective_radm_config.yaml"
                ).read_bytes()
            ).hexdigest(),
            "seed": _SEED,
            "fixture": {
                "image": "64x64 RGB arange uint8",
                "labels": [0, 3],
                "text_features": "two deterministic 768-D rows plus source padding",
                "hashes": fixture_hash_payload,
            },
            "tolerance": {"atol": _FLOAT_TOLERANCE.atol, "rtol": _FLOAT_TOLERANCE.rtol},
            "state": {"before": state_before, "after": state_after},
            "rng": {"before": source_before, "after": source_after},
            "errors": errors,
            "max_error": {"max_abs_diff": max_abs_diff, "max_rel_diff": max_rel_diff},
            "first_divergence": None,
            "stages_pending": ["S2", "S3", "S4", "S5"],
        }
        evidence_path = os.environ.get(
            "RADM_S1_EVIDENCE_PATH", ".cache/radm/s1/fixed_batch_trace.json"
        )
        path = Path(evidence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "S1 evidence "
            f"path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()} "
            f"vendor={_VENDOR_REVISION} executed=1 skipped=0 "
            f"first_divergence=none max_abs={max_abs_diff} max_rel={max_rel_diff}"
        )


@pytest.mark.vendor_parity
def test_s2_radm_one_optimizer_step_parity() -> None:
    """Compare one source-shaped backward, optimizer, and scheduler step."""
    apply_determinism(DeterminismConfig(seed=_SEED))
    with tempfile.TemporaryDirectory() as temporary:
        state = _require_reference_state(Path(temporary))
        case = _build_parity_case(state, Path(temporary))
        package_scheduler = case.package_scheduler
        fixture_tensors = case.fixture_tensors

        source_parameters, package_parameters = _assert_optimizer_mapping(case)

        step = _run_paired_optimizer_step(case)
        paired = step.paired
        source_before = paired.rng_before
        source_forward_after = paired.source_forward_rng
        package_forward_after = paired.package_forward_rng
        source_optimizer_state = step.source_optimizer_state
        package_optimizer_state = step.package_optimizer_state
        source_preclip_norm = step.source_preclip_norm
        package_preclip_norm = step.package_preclip_norm
        source_postclip_norm = step.source_postclip_norm
        package_postclip_norm = step.package_postclip_norm

        assert source_forward_after == package_forward_after
        assert step.source_after_rng == step.package_after_rng

        errors, lr_names, source_lrs, package_lrs = _compare_paired_optimizer_step(
            case,
            step,
            source_parameters,
            package_parameters,
            stage="S2",
        )
        loss_errors = errors["loss"]
        preclip_errors = errors["pre_clip_gradients"]
        postclip_errors = errors["post_clip_gradients"]
        parameter_errors = errors["parameters_after_step"]
        optimizer_errors = errors["optimizer_state"]
        norm_errors = errors["gradient_norm"]
        lr_errors = errors["learning_rates"]
        scheduler_errors = errors["scheduler"]
        assert state.scheduler.last_epoch == 1
        assert package_scheduler.last_epoch == 1

        fixture_hash_payload = {
            name: tensor_sha256(value) for name, value in fixture_tensors.items()
        }
        errors = {
            "loss": loss_errors,
            "pre_clip_gradients": preclip_errors,
            "post_clip_gradients": postclip_errors,
            "parameters_after_step": parameter_errors,
            "optimizer_state": optimizer_errors,
            "gradient_norm": norm_errors,
            "learning_rates": lr_errors,
            "scheduler": scheduler_errors,
        }
        comparisons = [
            comparison for group in errors.values() for comparison in group.values()
        ]
        max_abs_diff = max(
            float(comparison["max_abs_diff"]) for comparison in comparisons
        )
        max_rel_diff = max(
            float(comparison["max_rel_diff"]) for comparison in comparisons
        )
        evidence = {
            "vendor_revision": _VENDOR_REVISION,
            "config_sha256": hashlib.sha256(
                Path(
                    "models/radm/configs/training/effective_radm_config.yaml"
                ).read_bytes()
            ).hexdigest(),
            "seed": _SEED,
            "fixture": {"source_generated": True, "hashes": fixture_hash_payload},
            "step_order": [
                "forward",
                "zero_grad",
                "backward",
                "optimizer_step_with_full_model_clip",
                "scheduler_step",
            ],
            "tolerance": {
                "atol": _S2_FLOAT_TOLERANCE.atol,
                "rtol": _S2_FLOAT_TOLERANCE.rtol,
            },
            "rng": {
                "before": source_before,
                "after_forward": source_forward_after,
                "after": step.source_after_rng,
            },
            "gradient_norm": {
                "source_pre_clip": source_preclip_norm,
                "package_pre_clip": package_preclip_norm,
                "source_post_clip": source_postclip_norm,
                "package_post_clip": package_postclip_norm,
            },
            "optimizer": {
                "source_state_sha256": _digest_named_tensors(
                    {
                        f"{name}.{state_name}": value
                        for name, values in source_optimizer_state.items()
                        for state_name, value in values.items()
                        if isinstance(value, torch.Tensor)
                    }
                ),
                "package_state_sha256": _digest_named_tensors(
                    {
                        f"{name}.{state_name}": value
                        for name, values in package_optimizer_state.items()
                        for state_name, value in values.items()
                        if isinstance(value, torch.Tensor)
                    }
                ),
                "scheduler_last_epoch": state.scheduler.last_epoch,
                "source_class": type(state.scheduler).__name__,
                "package_class": type(package_scheduler).__name__,
                "source_state_keys": sorted(state.scheduler.state_dict()),
                "package_state_keys": sorted(package_scheduler.state_dict()),
                "learning_rates": dict(zip(lr_names, source_lrs.tolist(), strict=True)),
            },
            "errors": errors,
            "max_error": {"max_abs_diff": max_abs_diff, "max_rel_diff": max_rel_diff},
            "first_divergence": None,
            "executed": 1,
            "skipped": 0,
            "stages_pending": ["S3", "S4", "S5"],
        }
        evidence_path = os.environ.get(
            "RADM_S2_EVIDENCE_PATH", ".cache/radm/s2/one_step_trace.json"
        )
        path = Path(evidence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "S2 evidence "
            f"path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()} "
            f"vendor={_VENDOR_REVISION} executed=1 skipped=0 "
            f"first_divergence=none max_abs={max_abs_diff} max_rel={max_rel_diff}"
        )


@pytest.mark.vendor_parity
def test_s3_radm_deterministic_multi_batch_parity() -> None:
    """Compare repeated updates under the proven deterministic CUDA setting."""
    previous_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        # The supported V100 same-op ROI control found nonzero default backward
        # deltas but exact gradients with this setting; both paired paths must
        # observe the same global algorithm policy.
        assert torch.are_deterministic_algorithms_enabled()
        _run_s3_radm_deterministic_multi_batch_parity()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_algorithms)


def test_s3_optimizer_state_sync_deepcopies_tensor_storage() -> None:
    """Require synchronized optimizer tensors to remain independent after load."""
    source_parameter = torch.nn.Parameter(torch.ones(2))
    package_parameter = torch.nn.Parameter(torch.ones(2))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=1e-3)
    package_optimizer = torch.optim.AdamW([package_parameter], lr=1e-3)
    source_parameter.grad = torch.ones_like(source_parameter)
    source_optimizer.step()

    _copy_optimizer_state_for_sync(source_optimizer, package_optimizer)

    source_state = source_optimizer.state[source_parameter]
    package_state = package_optimizer.state[package_parameter]
    for state_name, source_value in source_state.items():
        package_value = package_state[state_name]
        if isinstance(source_value, torch.Tensor):
            assert isinstance(package_value, torch.Tensor)
            assert source_value.data_ptr() != package_value.data_ptr()
    source_state["exp_avg"].add_(1.0)
    assert not torch.equal(source_state["exp_avg"], package_state["exp_avg"])


def _run_s3_radm_mode(mode: str, run_root: Path) -> dict[str, Any]:
    """Record one natural or synchronized source-shaped S3 trajectory."""
    if mode not in {"natural", "synchronized"}:
        raise ValueError(f"unsupported S3 evidence mode: {mode}")
    enforce = mode == "synchronized"
    apply_determinism(DeterminismConfig(seed=_SEED))
    with tempfile.TemporaryDirectory() as temporary:
        state = _require_reference_state(Path(temporary))
        case = _build_parity_case(
            state,
            Path(temporary),
            fixture_name="s3_fixture_0",
            annotations=[
                {
                    "bbox": [8, 8, 24, 24],
                    "bbox_mode": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                },
                {
                    "bbox": [32, 32, 16, 16],
                    "bbox_mode": 0,
                    "category_id": 3,
                    "iscrowd": 0,
                },
            ],
        )
        fixtures = [
            (case.sample, case.package_batch, case.fixture_tensors),
            *_build_s3_followup_fixtures(state, Path(temporary)),
        ]
        assert len(fixtures) == 3

        source_parameters, package_parameters = _assert_optimizer_mapping(case)
        assert case.state_before == _mapped_state_digest(
            state.model, case.package, case.key_map
        )
        assert state.effective.gradient_accumulation_steps == 1
        assert state.effective.ema_enabled is False
        logged: list[tuple[str, float]] = []

        def record_log(name: str, value: torch.Tensor, **_: Any) -> None:
            logged.append((name, float(value.detach().cpu())))

        step_results: list[dict[str, Any]] = []
        state_sync_records: list[dict[str, Any]] = []
        divergences: list[dict[str, object]] = []
        with patch.object(case.module, "log", side_effect=record_log):
            for step_index, (sample, package_batch, _) in enumerate(fixtures):
                log_start = len(logged)
                state_hashes_before = _s3_state_hashes(case)
                step = _run_paired_optimizer_step(
                    case,
                    sample=sample,
                    package_batch=package_batch,
                )
                paired = step.paired
                step_stage = f"S3 step {step_index + 1}"
                errors, lr_names, source_lrs, package_lrs = (
                    _compare_paired_optimizer_step(
                        case,
                        step,
                        source_parameters,
                        package_parameters,
                        stage=step_stage,
                        enforce=enforce,
                        divergences=divergences,
                    )
                )

                step_logs = logged[log_start:]
                assert [name for name, _ in step_logs] == ["train_loss"]
                assert step_logs[0][1] == pytest.approx(
                    float(paired.source["total"]), abs=5e-5, rel=2e-4
                )
                checkpoint_fields = {
                    "model": case.package.state_dict(),
                    "optimizer": case.package_optimizer.state_dict(),
                    "scheduler": case.package_scheduler.state_dict(),
                    "global_step": step_index + 1,
                }
                assert set(checkpoint_fields) == {
                    "model",
                    "optimizer",
                    "scheduler",
                    "global_step",
                }
                step_comparisons = [
                    comparison
                    for group in errors.values()
                    for comparison in group.values()
                ]
                step_results.append(
                    {
                        "evidence_mode": mode,
                        "step": step_index + 1,
                        "rng": {
                            "before": paired.rng_before,
                            "after_forward": paired.source_forward_rng,
                            "after": step.source_after_rng,
                        },
                        "logging": {
                            "source_total_loss": float(paired.source["total"]),
                            "package_total_loss": float(
                                paired.package_loss.detach().cpu()
                            ),
                            "package_metric": "train_loss",
                            "calls": len(step_logs),
                        },
                        "gradient_norm": {
                            "source_pre_clip": step.source_preclip_norm,
                            "package_pre_clip": step.package_preclip_norm,
                            "source_post_clip": step.source_postclip_norm,
                            "package_post_clip": step.package_postclip_norm,
                        },
                        "learning_rates": {
                            "names": lr_names,
                            "source": source_lrs.tolist(),
                            "package": package_lrs.tolist(),
                        },
                        "scheduler_last_epoch": state.scheduler.last_epoch,
                        "scheduler_state": {
                            "source": case.state.scheduler.state_dict(),
                            "package": case.package_scheduler.state_dict(),
                        },
                        "state_hashes_before": state_hashes_before,
                        "state_hashes_after_optimizer_step": _s3_state_hashes(case),
                        "errors": errors,
                        "max_error": {
                            "max_abs_diff": max(
                                float(item["max_abs_diff"]) for item in step_comparisons
                            ),
                            "max_rel_diff": max(
                                float(item["max_rel_diff"]) for item in step_comparisons
                            ),
                        },
                    }
                )
                if enforce:
                    state_sync_records.append(_synchronize_s3_state(case))
                    step_results[-1]["state_sync"] = state_sync_records[-1]

        fixture_hash_payload = {
            f"batch_{index}.{name}": tensor_sha256(value)
            for index, (_, _, tensors) in enumerate(fixtures)
            for name, value in tensors.items()
        }
        all_comparisons = [
            comparison
            for result in step_results
            for group in result["errors"].values()
            for comparison in group.values()
        ]
        run_root.mkdir(parents=True, exist_ok=True)
        evidence = {
            "status": "PASS" if enforce else "RECORDED",
            "evidence_mode": mode,
            "run_root": run_root.as_posix(),
            "vendor_revision": _VENDOR_REVISION,
            "config_sha256": hashlib.sha256(
                Path(
                    "models/radm/configs/training/effective_radm_config.yaml"
                ).read_bytes()
            ).hexdigest(),
            "seed": _SEED,
            "fixture": {
                "source_generated": True,
                "batch_count": len(fixtures),
                "text_features": "deterministic 768-D rows generated for each source mapper sample",
                "hashes": fixture_hash_payload,
            },
            "steps": step_results,
            "trajectory": step_results,
            "state_synchronized_lockstep": state_sync_records,
            "step_order": [
                "forward",
                "zero_grad",
                "backward",
                "optimizer_step_with_full_model_clip",
                "scheduler_step",
                "checkpoint_state_capture",
            ],
            "logging": {
                "source_metric": "total_loss",
                "package_metric": "train_loss",
                "mapping_explicit": True,
                "calls": len(logged),
            },
            "accumulation_steps": state.effective.gradient_accumulation_steps,
            "checkpoint_fields": ["model", "optimizer", "scheduler", "global_step"],
            "tolerance": {
                "atol": _S2_FLOAT_TOLERANCE.atol,
                "rtol": _S2_FLOAT_TOLERANCE.rtol,
            },
            "max_error": {
                "max_abs_diff": max(
                    float(item["max_abs_diff"]) for item in all_comparisons
                ),
                "max_rel_diff": max(
                    float(item["max_rel_diff"]) for item in all_comparisons
                ),
            },
            "first_divergence": divergences[0] if divergences else None,
            "executed": len(fixtures),
            "batches": len(fixtures),
            "skipped": 0,
            "stages_pending": ["S4", "S5"],
        }
        path = run_root / "s3-trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"S3 {mode} evidence "
            f"path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()} "
            f"vendor={_VENDOR_REVISION} executed={len(fixtures)} "
            f"batches={len(fixtures)} skipped=0 "
            f"first_divergence={evidence['first_divergence']} "
            f"max_abs={evidence['max_error']['max_abs_diff']} "
            f"max_rel={evidence['max_error']['max_rel_diff']}"
        )
        return evidence


def _fresh_s3_run_root(base_root: Path) -> Path:
    """Allocate a non-overwriting numbered root for one S3 evidence run."""
    runs_root = base_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = runs_root / f"run-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no unused S3 run directory under {runs_root}")


def _natural_run_to_run_envelope(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize natural-run drift without turning it into a parity assertion."""
    first_steps = cast(list[dict[str, Any]], first["trajectory"])
    second_steps = cast(list[dict[str, Any]], second["trajectory"])
    assert len(first_steps) == len(second_steps)
    metrics = {
        "source_total_loss": 0.0,
        "package_total_loss": 0.0,
        "source_pre_clip_gradient_norm": 0.0,
        "package_pre_clip_gradient_norm": 0.0,
        "source_post_clip_gradient_norm": 0.0,
        "package_post_clip_gradient_norm": 0.0,
    }
    first_hash_divergence: int | None = None
    for first_step, second_step in zip(first_steps, second_steps, strict=True):
        step = int(first_step["step"])
        assert step == int(second_step["step"])
        for loss_name in ("source_total_loss", "package_total_loss"):
            first_loss = float(first_step["logging"][loss_name])
            second_loss = float(second_step["logging"][loss_name])
            metrics[loss_name] = max(metrics[loss_name], abs(first_loss - second_loss))
        first_norms = first_step["gradient_norm"]
        second_norms = second_step["gradient_norm"]
        for key, metric in (
            ("source_pre_clip", "source_pre_clip_gradient_norm"),
            ("package_pre_clip", "package_pre_clip_gradient_norm"),
            ("source_post_clip", "source_post_clip_gradient_norm"),
            ("package_post_clip", "package_post_clip_gradient_norm"),
        ):
            metrics[metric] = max(
                metrics[metric], abs(float(first_norms[key]) - float(second_norms[key]))
            )
        first_model_hash = first_step["state_hashes_after_optimizer_step"]["model"][
            "package"
        ]
        second_model_hash = second_step["state_hashes_after_optimizer_step"]["model"][
            "package"
        ]
        if first_model_hash != second_model_hash and first_hash_divergence is None:
            first_hash_divergence = step
    return {
        "run_count": 2,
        "step_count": len(first_steps),
        "max_abs_diff": metrics,
        "first_package_state_hash_divergence_step": first_hash_divergence,
        "package_state_hashes_equal": first_hash_divergence is None,
    }


def _run_s3_radm_deterministic_multi_batch_parity() -> None:
    """Run natural duplicates and one enforcing synchronized lockstep."""
    evidence_path = Path(
        os.environ.get("RADM_S3_EVIDENCE_PATH", ".cache/radm/s3/multi_batch_trace.json")
    )
    run_base = evidence_path.parent
    natural_runs = [
        _run_s3_radm_mode("natural", _fresh_s3_run_root(run_base)),
        _run_s3_radm_mode("natural", _fresh_s3_run_root(run_base)),
    ]
    synchronized_run = _run_s3_radm_mode("synchronized", _fresh_s3_run_root(run_base))
    evidence = {
        "status": synchronized_run["status"],
        "evidence_layers": {
            "natural_trajectory": natural_runs,
            "state_synchronized_lockstep": synchronized_run,
        },
        "natural_run_to_run_envelope": _natural_run_to_run_envelope(
            natural_runs[0], natural_runs[1]
        ),
        "first_divergence": {
            "natural_run_1": natural_runs[0]["first_divergence"],
            "natural_run_2": natural_runs[1]["first_divergence"],
            "state_synchronized_lockstep": synchronized_run["first_divergence"],
        },
        "vendor_revision": _VENDOR_REVISION,
        "config_sha256": natural_runs[0]["config_sha256"],
        "seed": _SEED,
        "tolerance": natural_runs[0]["tolerance"],
        "runtime": {
            "device": os.environ.get("RADM_REFERENCE_DEVICE", "cpu"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "run_artifacts": {
            "natural": [run["run_root"] for run in natural_runs],
            "synchronized": synchronized_run["run_root"],
        },
        "stages_pending": ["S4", "S5"],
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "S3 aggregate evidence "
        f"path={evidence_path} sha256={hashlib.sha256(evidence_path.read_bytes()).hexdigest()} "
        f"executed=3 skipped=0 first_divergence={evidence['first_divergence']}"
    )
