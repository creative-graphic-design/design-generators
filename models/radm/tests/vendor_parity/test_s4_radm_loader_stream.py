"""Authoritative RADM loader-stream parity on the approved local data."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - dynamic Detectron2 mapper surface.

import numpy as np
import pytest
import torch

from radm.training.config import effective_radm_config
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


def _first_order_difference(source: list[int], package: list[int]) -> dict[str, object] | None:
    for index, (source_id, package_id) in enumerate(zip(source, package)):
        if source_id != package_id:
            return {
                "index": index,
                "source_image_id": source_id,
                "package_image_id": package_id,
            }
    if len(source) != len(package):
        return {"index": min(len(source), len(package)), "lengths": [len(source), len(package)]}
    return None


def _feature_inventory(root: Path, split: str, image_names: set[str]) -> dict[str, object]:
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

    instances = source["instances"]
    height, width = instances.image_size
    scale = torch.tensor((width, height, width, height), dtype=torch.float32)
    source_boxes = instances.gt_boxes.tensor.detach().cpu().float() / scale
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
    return {
        "image_shape": [list(source_image.shape), list(package_image.shape)],
        "image_exact": torch.equal(source_image, package_image),
        "image_max_abs": float((source_image - package_image).abs().max()),
        "boxes_exact": torch.equal(source_boxes, package_boxes)
        if source_boxes.shape == package_boxes.shape
        else False,
        "boxes_max_abs": float((source_boxes - package_boxes).abs().max())
        if source_boxes.shape == package_boxes.shape
        else None,
        "source_labels": instances.gt_classes.detach().cpu().tolist(),
        "package_labels": package["labels"].detach().cpu().tolist(),
        "labels_equal": torch.equal(instances.gt_classes.detach().cpu(), package["labels"]),
        "text_features_max_abs": float((source_features - package_features).abs().max()),
        "text_mask_equal": torch.equal(source_mask, package_mask),
        "source_image_size": [int(height), int(width)],
        "package_image_size": [int(package["image"].shape[-2]), int(package["image"].shape[-1])],
    }


def _run_s4() -> dict[str, object]:
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("PARITY_REQUIRE=1 is required for RADM S4")
    if os.environ.get("RADM_S4_ALLOW_MISSING") != "1":
        raise RuntimeError("RADM_S4_ALLOW_MISSING=1 is required for the approved diagnostic fallback")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl")).resolve()
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
    aligned_sample: dict[str, object] | None = None
    with _vendor_import_root(VENDOR_ROOT), _legacy_pillow_compat():
        detectron2_data = importlib.import_module("detectron2.data")
        mapper_class = importlib.import_module("RADM.dataset_mapper").RADMDatasetMapper
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
            )
            source_records = detectron2_data.DatasetCatalog.get(dataset_name)
            source_ids = [int(record["image_id"]) for record in source_records]
            package_ids = [int(record["id"]) for record in package_dataset.images]
            package_by_id = {image_id: index for index, image_id in enumerate(package_ids)}
            image_names = {str(record["file_name"]) for record in annotation_payload["images"]}
            source_paths_present = all(
                Path(record["file_name"]).is_file() for record in source_records
            )
            mapper = mapper_class(state.config, is_train=split == "train")
            split_report: dict[str, object] = {
                "source_count": len(source_ids),
                "package_count": len(package_ids),
                "source_order_sha256": _sha256_json(source_ids),
                "package_order_sha256": _sha256_json(package_ids),
                "first_order_difference": _first_order_difference(source_ids, package_ids),
                "source_image_paths_present": source_paths_present,
                "feature_inventory": _feature_inventory(data_root, split, image_names),
                "category_names": list(EXPECTED_LABELS),
                "annotation_category_ids": sorted(
                    {int(row["category_id"]) for row in annotation_payload["annotations"]}
                ),
            }
            split_metadata[split] = split_report
            if first_divergence is None and split_report["first_order_difference"] is not None:
                first_divergence = f"{split}.order"
            if split == "train":
                first_record = source_records[0]
                aligned_sample = _aligned_sample(
                    source_record=first_record,
                    package_index=package_by_id[int(first_record["image_id"])],
                    mapper=mapper,
                    package_dataset=package_dataset,
                    effective=package_effective,
                )
                feature_inventory = _feature_inventory(data_root, split, image_names)
                missing_stem = next(
                    (
                        stem
                        for stem in cast(list[str], feature_inventory["missing_examples"])
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

    return {
        "status": "PASS" if first_divergence is None else "FAIL",
        "stage": "S4",
        "first_divergence": first_divergence,
        "data_root": str(data_root),
        "archive_sha256": archive_sha256,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "effective_config_sha256": hashlib.sha256(
            (ROOT / "models/radm/configs/training/effective_radm_config.yaml").read_bytes()
        ).hexdigest(),
        "runtime": runtime,
        "split_metadata": split_metadata,
        "aligned_train_sample": aligned_sample,
        "fallback_policy": "diagnostic allow_missing=True; package default unchanged",
        "command": " ".join(sys.argv),
        "package_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


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
