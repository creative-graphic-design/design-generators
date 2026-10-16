"""Authoritative RADM loader-stream parity on the approved local data."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - dynamic Detectron2 mapper surface.

import numpy as np
import pytest
import torch

from radm.training.config import effective_radm_config
from radm.training.datamodule import RADMDataModule
from radm.training.dataset import RADMCOCODataset, load_text_features
from reference_adapter import (
    RADMReferenceAdapter,
    _legacy_pillow_compat,
    _vendor_import_root,
)


ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = ROOT / "vendor" / "radm"
EXPECTED_LABELS = ("Logo", "文字", "衬底", "符号元素", "强调突出子部分文字")
pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_evidence(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite S4 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _first_order_difference(
    source: list[int], package: list[int]
) -> dict[str, object] | None:
    for index, (source_id, package_id) in enumerate(zip(source, package)):
        if source_id != package_id:
            return {
                "index": index,
                "source_image_id": source_id,
                "package_image_id": package_id,
            }
    if len(source) != len(package):
        return {
            "index": min(len(source), len(package)),
            "lengths": [len(source), len(package)],
        }
    return None


def _feature_inventory(
    root: Path, split: str, image_names: set[str]
) -> dict[str, object]:
    feature_root = root / "text_features" / split
    stems = {Path(name).stem for name in image_names}
    present = {
        path.name.removesuffix("_feats.pth")
        for path in feature_root.iterdir()
        if path.is_file() and path.name.endswith("_feats.pth")
    }
    missing = sorted(stems - present)
    return {
        "image_count": len(stems),
        "feature_count": len(present),
        "missing_count": len(missing),
        "missing_stem_sha256": _sha256_json(missing),
        "missing_examples": missing[:3],
    }


def _runtime_metadata(device: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("PARITY_REQUIRE=1 S4 requires CUDA")
    logical_index = torch.cuda.current_device()
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": device,
        "current_device": logical_index,
        "device_name": torch.cuda.get_device_name(logical_index),
        "capability": list(torch.cuda.get_device_capability(logical_index)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _aligned_sample(
    *,
    source_record: dict[str, Any],
    package_index: int,
    mapper: Any,
    package_dataset: RADMCOCODataset,
    effective: Any,
) -> dict[str, object]:
    numpy_state = copy.deepcopy(np.random.get_state())
    python_state = copy.deepcopy(__import__("random").getstate())
    torch_state = torch.random.get_rng_state()
    source = mapper(source_record)
    np.random.set_state(numpy_state)
    __import__("random").setstate(python_state)
    torch.random.set_rng_state(torch_state)
    package = package_dataset[package_index]

    source_image = source["image"].detach().cpu().float()
    mean = torch.tensor(effective.pixel_mean).reshape(3, 1, 1)
    std = torch.tensor(effective.pixel_std).reshape(3, 1, 1)
    source_image = (source_image - mean) / std
    source_features = source["text_fea"]["feats"].detach().cpu()
    source_mask = source["text_mask"].detach().cpu().bool()
    package_image = package["image"].detach().cpu()
    package_boxes = package["boxes_xyxy"].detach().cpu()
    package_features = package["text_features"].detach().cpu()
    package_mask = package["text_mask"].detach().cpu().bool()
    source_height, source_width = source_image.shape[-2:]
    package_height, package_width = package_image.shape[-2:]
    package_image_scale = package["image_size_xyxy"].detach().cpu()
    package_original_scale = package["original_image_size_xyxy"].detach().cpu()
    package_boxes_absolute = package_boxes * package_image_scale
    boxes_in_resized_frame = bool(
        torch.all(package_boxes_absolute >= 0)
        and torch.all(package_boxes_absolute <= package_image_scale.reshape(1, 4))
    )
    instances = source.get("instances")
    if instances is None:
        source_boxes: torch.Tensor | None = None
        source_boxes_dtype: str | None = None
        source_labels: list[int] | None = None
    else:
        height, width = instances.image_size
        scale = torch.tensor((width, height, width, height), dtype=torch.float32)
        source_boxes_tensor = instances.gt_boxes.tensor.detach().cpu()
        source_boxes_dtype = str(source_boxes_tensor.dtype)
        source_boxes = source_boxes_tensor.float() / scale
        source_labels = instances.gt_classes.detach().cpu().tolist()
    return {
        "image_shape": [list(source_image.shape), list(package_image.shape)],
        "image_exact": torch.equal(source_image, package_image),
        "image_max_abs": float((source_image - package_image).abs().max()),
        "boxes_exact": (
            torch.equal(source_boxes, package_boxes)
            if source_boxes is not None and source_boxes.shape == package_boxes.shape
            else None
        ),
        "boxes_max_abs": (
            float((source_boxes - package_boxes).abs().max())
            if source_boxes is not None and source_boxes.shape == package_boxes.shape
            else None
        ),
        "source_boxes_xyxy": source_boxes.tolist()
        if source_boxes is not None
        else None,
        "package_boxes_xyxy": package_boxes.tolist(),
        "source_boxes_dtype": source_boxes_dtype,
        "package_boxes_dtype": str(package_boxes.dtype),
        "source_labels": source_labels,
        "package_labels": package["labels"].detach().cpu().tolist(),
        "labels_equal": torch.equal(
            instances.gt_classes.detach().cpu(), package["labels"]
        )
        if instances is not None
        else None,
        "text_features_max_abs": float(
            (source_features - package_features).abs().max()
        ),
        "text_mask_equal": torch.equal(source_mask, package_mask),
        "source_image_size": [int(source_height), int(source_width)],
        "package_image_size": [
            int(package_height),
            int(package_width),
        ],
        "output_box_frame": {
            "source": "normalized xyxy relative to the resized mapper image",
            "package": "normalized xyxy relative to image_size_xyxy",
            "package_image_size_xyxy": package_image_scale.tolist(),
            "package_original_image_size_xyxy": package_original_scale.tolist(),
            "package_boxes_within_resized_frame": boxes_in_resized_frame,
        },
    }


def _assert_aligned_train_mapper_oracle(sample: dict[str, object]) -> None:
    """Fail before the long stream walk if the vendor mapper fixture diverges."""
    assert sample["boxes_exact"] is True, json.dumps(
        sample, ensure_ascii=False, indent=2
    )
    assert sample["source_boxes_dtype"] == "torch.float32", json.dumps(
        sample, ensure_ascii=False, indent=2
    )
    assert sample["package_boxes_dtype"] == "torch.float32", json.dumps(
        sample, ensure_ascii=False, indent=2
    )


def _rng_state() -> tuple[object, object, torch.Tensor]:
    return (
        copy.deepcopy(np.random.get_state()),
        copy.deepcopy(random.getstate()),
        torch.random.get_rng_state(),
    )


def _restore_rng_state(state: tuple[object, object, torch.Tensor]) -> None:
    numpy_state, python_state, torch_state = state
    np.random.set_state(cast(tuple[Any, ...], numpy_state))
    random.setstate(cast(tuple[Any, ...], python_state))
    torch.random.set_rng_state(torch_state)


def _transform_metadata(transforms: Any) -> dict[str, object]:
    values = getattr(transforms, "transforms", [transforms])
    resize = next(
        (value for value in values if type(value).__name__ == "ResizeTransform"),
        None,
    )
    return {
        "flipped": any(type(value).__name__ == "HFlipTransform" for value in values),
        "resized_height": int(resize.new_h) if resize is not None else None,
        "resized_width": int(resize.new_w) if resize is not None else None,
        "min_size": (
            min(int(resize.new_h), int(resize.new_w)) if resize is not None else None
        ),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _full_training_stream(
    *,
    state: Any,
    package_effective: Any,
    data_root: Path,
    mapper_module: Any,
    train_net: Any,
    window_batches: int,
) -> dict[str, object]:
    source_loader = train_net.Trainer.build_train_loader(state.config)
    source_sampler = _find_sampler(source_loader)
    source_sampler_seed = int(source_sampler._seed)
    package_module = RADMDataModule(
        train_annotations=data_root / "annotations" / "train.json",
        train_image_root=data_root / "images" / "train",
        train_text_feature_root=data_root / "text_features" / "train",
        batch_size=package_effective.batch_size,
        num_workers=package_effective.num_workers,
        sampler_seed=source_sampler_seed,
        allow_missing_text_features=True,
        effective=package_effective,
    )
    package_loader = package_module.train_dataloader()
    source_iterator = iter(source_loader)
    package_iterator = iter(package_loader)
    source_transform_records: list[dict[str, object]] = []
    original_apply = mapper_module.T.apply_transform_gens

    def capture_transforms(*args: Any, **kwargs: Any) -> Any:
        result = original_apply(*args, **kwargs)
        source_transform_records.append(_transform_metadata(result[1]))
        return result

    mapper_module.T.apply_transform_gens = capture_transforms
    batches: list[dict[str, object]] = []
    first_mismatch: dict[str, object] | None = None
    try:
        for batch_index in range(window_batches):
            source_transform_records.clear()
            state_before_mapping = _rng_state()
            source_batch = next(source_iterator)
            source_metadata = source_transform_records[:]
            _restore_rng_state(state_before_mapping)
            package_batch = next(package_iterator)
            source_ids = [int(item["image_id"]) for item in source_batch]
            package_ids = package_batch["image_ids"].tolist()
            batch_report: dict[str, object] = {
                "batch": batch_index,
                "source_image_ids": source_ids,
                "package_image_ids": package_ids,
                "source_transforms": source_metadata,
                "package_flipped": package_batch["transform_flipped"].tolist(),
                "package_min_sizes": package_batch["transform_min_size"].tolist(),
                "source_target_sha256": [],
                "package_target_sha256": [],
                "max_target_abs": 0.0,
                "target_mismatches": [],
            }
            checks: list[bool] = [source_ids == package_ids]
            if len(source_metadata) != len(source_batch):
                checks.append(False)
            for item_index, source_item in enumerate(source_batch):
                source_image = source_item["image"].detach().cpu().float()
                source_instances = source_item["instances"]
                height, width = source_instances.image_size
                source_scale = torch.tensor(
                    (width, height, width, height), dtype=torch.float32
                )
                source_boxes = (
                    source_instances.gt_boxes.tensor.detach().cpu() / source_scale
                )
                package_height = int(package_batch["image_scales"][item_index, 1])
                package_width = int(package_batch["image_scales"][item_index, 0])
                package_boxes = (
                    package_batch["boxes_xyxy"][item_index][
                        package_batch["mask"][item_index]
                    ]
                    .detach()
                    .cpu()
                )
                package_image = (
                    package_batch["images"][
                        item_index, :, :package_height, :package_width
                    ]
                    .detach()
                    .cpu()
                )
                mean = torch.tensor(package_effective.pixel_mean).reshape(3, 1, 1)
                std = torch.tensor(package_effective.pixel_std).reshape(3, 1, 1)
                source_image = (source_image - mean) / std
                target_difference = (
                    float((source_boxes - package_boxes).abs().max())
                    if source_boxes.numel()
                    else 0.0
                )
                if target_difference > 1e-6:
                    source_boxes_absolute = (
                        source_instances.gt_boxes.tensor.detach().cpu()
                    )
                    package_boxes_absolute = package_boxes * source_scale
                    difference = (source_boxes_absolute - package_boxes_absolute).abs()
                    mismatch_indices = torch.nonzero(difference > 1e-6, as_tuple=False)
                    target_mismatches = cast(
                        list[dict[str, object]], batch_report["target_mismatches"]
                    )
                    target_mismatches.append(
                        {
                            "item_index": item_index,
                            "image_id": source_item["image_id"],
                            "source_target_dtype": str(source_boxes_absolute.dtype),
                            "package_target_dtype": str(package_boxes_absolute.dtype),
                            "differing_box_coordinates": mismatch_indices.tolist(),
                            "source_boxes_xyxy": source_boxes_absolute.tolist(),
                            "package_boxes_xyxy": package_boxes_absolute.tolist(),
                        }
                    )
                batch_report["source_target_sha256"].append(
                    _tensor_sha256(source_boxes)
                )
                batch_report["package_target_sha256"].append(
                    _tensor_sha256(package_boxes)
                )
                batch_report["max_target_abs"] = max(
                    float(batch_report["max_target_abs"]), target_difference
                )
                checks.extend(
                    [
                        source_item["image_id"] == package_ids[item_index],
                        list(source_image.shape) == [3, package_height, package_width],
                        torch.equal(source_image, package_image),
                        source_boxes.shape == package_boxes.shape,
                        target_difference <= 1e-6,
                        source_instances.gt_classes.detach().cpu().tolist()
                        == package_batch["labels"][item_index][
                            package_batch["mask"][item_index]
                        ].tolist(),
                        source_metadata[item_index]["flipped"]
                        == bool(package_batch["transform_flipped"][item_index]),
                        source_metadata[item_index]["min_size"]
                        == int(package_batch["transform_min_size"][item_index]),
                    ]
                )
            batch_report["pass"] = all(checks)
            batches.append(batch_report)
            if not all(checks) and first_mismatch is None:
                first_mismatch = batch_report
    finally:
        mapper_module.T.apply_transform_gens = original_apply
    return {
        "status": "PASS" if first_mismatch is None else "FAIL",
        "window_batches": window_batches,
        "window_images": window_batches * package_effective.batch_size,
        "source_sampler": {
            "class": type(source_sampler).__name__,
            "seed": source_sampler_seed,
            "aspect_ratio_grouping": True,
            "num_workers": int(state.config.DATALOADER.NUM_WORKERS),
        },
        "package_sampler": {
            "class": "RADMTrainingSampler",
            "seed": source_sampler_seed,
            "aspect_ratio_grouping": True,
            "num_workers": package_effective.num_workers,
            "drop_last": True,
        },
        "first_mismatch": first_mismatch,
        "batches": batches,
    }


def _find_sampler(loader: Any) -> Any:
    pending = [loader]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        sampler = getattr(value, "sampler", None)
        if sampler is not None and hasattr(sampler, "_seed"):
            return sampler
        for attribute in ("dataset", "_dataset"):
            child = getattr(value, attribute, None)
            if child is not None:
                pending.append(child)
    raise RuntimeError("could not locate the source training sampler")


def _run_s4(*, include_full_training_stream: bool = True) -> dict[str, object]:
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("PARITY_REQUIRE=1 is required for RADM S4")
    if os.environ.get("RADM_S4_ALLOW_MISSING") != "1":
        raise RuntimeError(
            "RADM_S4_ALLOW_MISSING=1 is required for the approved diagnostic fallback"
        )
    data_root = Path(
        os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl")
    ).resolve()
    device = os.environ.get("RADM_REFERENCE_DEVICE", "cuda:0")
    archive_sha256 = os.environ.get("RADM_S4_ARCHIVE_SHA256")
    if not archive_sha256:
        raise RuntimeError("RADM_S4_ARCHIVE_SHA256 must identify the approved archive")
    runtime = _runtime_metadata(device)
    required = [
        data_root / "annotations" / "train.json",
        data_root / "annotations" / "test.json",
        data_root / "images" / "train",
        data_root / "images" / "test",
        data_root / "text_features" / "train",
        data_root / "text_features" / "test",
    ]
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"S4 data preflight missing paths: {missing_paths}")

    adapter = RADMReferenceAdapter(
        vendor_root=VENDOR_ROOT,
        dataset_root=data_root,
        text_feature_root=data_root / "text_features",
        device=device,
    )
    state = adapter.build_initialized_state()
    package_effective = effective_radm_config()
    split_metadata: dict[str, object] = {}
    first_divergence: str | None = None
    aligned_samples: dict[str, dict[str, object]] = {}
    train_dataset: RADMCOCODataset | None = None
    with _vendor_import_root(VENDOR_ROOT), _legacy_pillow_compat():
        detectron2_data = importlib.import_module("detectron2.data")
        mapper_module = importlib.import_module("RADM.dataset_mapper")
        mapper_class = mapper_module.RADMDatasetMapper
        train_net = importlib.import_module("train_net")
        for split, dataset_name in (("train", "layout_train"), ("test", "layout_val")):
            annotation_payload = json.loads(
                (data_root / "annotations" / f"{split}.json").read_text()
            )
            package_dataset = RADMCOCODataset(
                annotation_path=data_root / "annotations" / f"{split}.json",
                image_root=data_root / "images" / split,
                text_feature_root=data_root / "text_features" / split,
                effective=package_effective,
                allow_missing_text_features=True,
                train=split == "train",
                read_text_features=split == "train",
            )
            if split == "train":
                train_dataset = package_dataset
            source_records = detectron2_data.DatasetCatalog.get(dataset_name)
            source_ids = [int(record["image_id"]) for record in source_records]
            package_ids = [int(record["id"]) for record in package_dataset.images]
            package_by_id = {
                image_id: index for index, image_id in enumerate(package_ids)
            }
            image_names = {
                str(record["file_name"]) for record in annotation_payload["images"]
            }
            source_paths_present = all(
                Path(record["file_name"]).is_file() for record in source_records
            )
            mapper = mapper_class(state.config, is_train=split == "train")
            split_report: dict[str, object] = {
                "source_count": len(source_ids),
                "package_count": len(package_ids),
                "source_order_sha256": _sha256_json(source_ids),
                "package_order_sha256": _sha256_json(package_ids),
                "first_order_difference": _first_order_difference(
                    source_ids, package_ids
                ),
                "source_image_paths_present": source_paths_present,
                "feature_inventory": _feature_inventory(data_root, split, image_names),
                "category_names": list(EXPECTED_LABELS),
                "annotation_category_ids": sorted(
                    {
                        int(row["category_id"])
                        for row in annotation_payload["annotations"]
                    }
                ),
            }
            split_metadata[split] = split_report
            if (
                first_divergence is None
                and split_report["first_order_difference"] is not None
            ):
                first_divergence = f"{split}.order"
            first_record = source_records[0]
            aligned_samples[split] = _aligned_sample(
                source_record=first_record,
                package_index=package_by_id[int(first_record["image_id"])],
                mapper=mapper,
                package_dataset=package_dataset,
                effective=package_effective,
            )
            if split == "train":
                feature_inventory = _feature_inventory(data_root, split, image_names)
                missing_stem = next(
                    (
                        stem
                        for stem in cast(
                            list[str], feature_inventory["missing_examples"]
                        )
                        if stem
                    ),
                    None,
                )
                if missing_stem is not None:
                    missing_name = next(
                        name for name in image_names if Path(name).stem == missing_stem
                    )
                    source_features, source_mask = mapper.load_text(
                        str(data_root / "images" / split / missing_name)
                    )
                    package_features, package_mask = load_text_features(
                        missing_name,
                        root=data_root / "text_features" / split,
                        effective=package_effective,
                        allow_missing=True,
                    )
                    split_report["fallback_equal"] = bool(
                        torch.equal(source_features["feats"], package_features)
                        and torch.equal(source_mask, package_mask)
                    )

        if train_dataset is None:
            raise RuntimeError("training dataset was not initialized")
        train_mapper_sample = aligned_samples["train"]
        _assert_aligned_train_mapper_oracle(train_mapper_sample)
        if include_full_training_stream:
            full_training_stream = _full_training_stream(
                state=state,
                package_effective=package_effective,
                data_root=data_root,
                mapper_module=mapper_module,
                train_net=train_net,
                window_batches=max(
                    1, len(train_dataset) // package_effective.batch_size
                ),
            )
        else:
            full_training_stream = {
                "status": "NOT_RUN",
                "reason": "fast aligned-mapper oracle only",
            }

    stream_status = (
        full_training_stream.get("status")
        if isinstance(full_training_stream, dict)
        else None
    )
    status = (
        "PASS"
        if first_divergence is None and stream_status in {"PASS", "NOT_RUN"}
        else "FAIL"
    )
    return {
        "status": status,
        "stage": "S4",
        "first_divergence": first_divergence,
        "data_root": str(data_root),
        "archive_sha256": archive_sha256,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "effective_config_sha256": hashlib.sha256(
            (
                ROOT / "models/radm/configs/training/effective_radm_config.yaml"
            ).read_bytes()
        ).hexdigest(),
        "runtime": runtime,
        "split_metadata": split_metadata,
        "aligned_train_sample": aligned_samples.get("train"),
        "aligned_test_sample": aligned_samples.get("test"),
        "full_training_stream": full_training_stream,
        "fallback_policy": "diagnostic allow_missing=True; package default unchanged",
        "command": " ".join(sys.argv),
        "package_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def test_s4_aligned_train_mapper_fixture_matches_vendor() -> None:
    """Check vendor/package target dtype and coordinates before the stream walk."""
    report = _run_s4(include_full_training_stream=False)
    aligned_train_sample = report["aligned_train_sample"]
    assert isinstance(aligned_train_sample, dict)
    _assert_aligned_train_mapper_oracle(cast(dict[str, object], aligned_train_sample))


def test_s4_radm_loader_stream_parity() -> None:
    """Compare source and package authoritative train/test loader metadata."""
    evidence_path = Path(
        os.environ.get(
            "RADM_S4_EVIDENCE_PATH",
            ".cache/radm/s4/loader-stream.json",
        )
    )
    report = _run_s4()
    _write_evidence(evidence_path, report)
    assert report["status"] == "PASS", json.dumps(report, ensure_ascii=False, indent=2)
    aligned_train_sample = report["aligned_train_sample"]
    assert isinstance(aligned_train_sample, dict)
    aligned_train_sample = cast(dict[str, object], aligned_train_sample)
    assert aligned_train_sample["labels_equal"] is True, json.dumps(
        aligned_train_sample, ensure_ascii=False, indent=2
    )
    assert aligned_train_sample["image_exact"] is True, json.dumps(
        aligned_train_sample, ensure_ascii=False, indent=2
    )
    assert aligned_train_sample["boxes_exact"] is True, json.dumps(
        aligned_train_sample, ensure_ascii=False, indent=2
    )
    aligned_test_sample = report["aligned_test_sample"]
    assert isinstance(aligned_test_sample, dict)
    aligned_test_sample = cast(dict[str, object], aligned_test_sample)
    assert aligned_test_sample["image_exact"] is True, json.dumps(
        aligned_test_sample, ensure_ascii=False, indent=2
    )
    assert aligned_test_sample["text_features_max_abs"] == 0.0, json.dumps(
        aligned_test_sample, ensure_ascii=False, indent=2
    )
    assert aligned_test_sample["text_mask_equal"] is True, json.dumps(
        aligned_test_sample, ensure_ascii=False, indent=2
    )
    output_box_frame = aligned_test_sample["output_box_frame"]
    assert isinstance(output_box_frame, dict)
    output_box_frame = cast(dict[str, object], output_box_frame)
    assert output_box_frame["package_boxes_within_resized_frame"] is True, json.dumps(
        aligned_test_sample, ensure_ascii=False, indent=2
    )
    assert output_box_frame["package_image_size_xyxy"] == [800.0, 1189.0, 800.0, 1189.0]
    assert output_box_frame["package_original_image_size_xyxy"] == [
        350.0,
        520.0,
        350.0,
        520.0,
    ]
    full_training_stream = report["full_training_stream"]
    assert isinstance(full_training_stream, dict)
    full_training_stream = cast(dict[str, object], full_training_stream)
    assert full_training_stream["status"] == "PASS", json.dumps(
        full_training_stream, ensure_ascii=False, indent=2
    )
    assert full_training_stream["first_mismatch"] is None
