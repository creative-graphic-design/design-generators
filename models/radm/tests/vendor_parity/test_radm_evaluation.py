"""Numerical evaluation parity against the pinned evaluation stack."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - Detectron2 evaluator objects are dynamic.

import pytest
import torch

from radm.evaluation import (
    CocoMetricResults,
    CocoPrediction,
    evaluate_cgl_predictions,
)
from reference_adapter import (
    RADMReferenceAdapter,
    _legacy_pillow_compat,
    _vendor_import_root,
)


ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = ROOT / "vendor" / "radm"
pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def test_cgl_evaluator_matches_vendor_bbox_metrics() -> None:
    """Compare real COCO AP values on a fixed CGL test subset."""
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("PARITY_REQUIRE=1 is required for vendor evaluation parity")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    annotation_path = data_root / "annotations" / "test.json"
    if not annotation_path.is_file():
        pytest.fail(f"CGL test annotations are missing: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    subset = payload["images"][:8]
    subset_ids = [int(image["id"]) for image in subset]
    annotations_by_id: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in subset_ids
    }
    for annotation in payload["annotations"]:
        if int(annotation["image_id"]) in annotations_by_id:
            annotations_by_id[int(annotation["image_id"])].append(annotation)
    predictions: list[CocoPrediction] = []
    for image in subset:
        image_id = int(image["id"])
        for annotation in annotations_by_id[image_id]:
            predictions.append(
                {
                    "image_id": image_id,
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "score": 0.75,
                    "category_id": int(annotation["category_id"]),
                }
            )

    package = evaluate_cgl_predictions(
        annotation_path,
        predictions,
        image_ids=subset_ids,
    )

    with _vendor_import_root(VENDOR_ROOT), _legacy_pillow_compat():
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.evaluation import COCOEvaluator
        from detectron2.structures import Boxes, Instances

        adapter = RADMReferenceAdapter(
            vendor_root=VENDOR_ROOT,
            dataset_root=data_root,
            text_feature_root=data_root / "text_features",
            device="cpu",
        )
        state = adapter.build_initialized_state()
        records = DatasetCatalog.get("layout_val")
        record_by_id = {int(record["image_id"]): record for record in records}
        with tempfile.TemporaryDirectory() as output_dir:
            evaluator = COCOEvaluator(
                "layout_val",
                tasks=("bbox",),
                distributed=False,
                output_dir=output_dir,
                use_fast_impl=False,
            )
            evaluator.reset()
            grouped: dict[int, list[CocoPrediction]] = {
                image_id: [] for image_id in subset_ids
            }
            for prediction in predictions:
                image_id = cast(int, prediction["image_id"])
                grouped[image_id].append(prediction)
            inputs: list[dict[str, object]] = []
            outputs: list[dict[str, Instances]] = []
            for image_id in subset_ids:
                record = record_by_id[image_id]
                instances = Instances((int(record["height"]), int(record["width"])))
                rows = grouped[image_id]
                instances.pred_boxes = Boxes(
                    torch.tensor(
                        [
                            [
                                cast(list[float], row["bbox"])[0],
                                cast(list[float], row["bbox"])[1],
                                cast(list[float], row["bbox"])[0]
                                + cast(list[float], row["bbox"])[2],
                                cast(list[float], row["bbox"])[1]
                                + cast(list[float], row["bbox"])[3],
                            ]
                            for row in rows
                        ],
                        dtype=torch.float32,
                    )
                )
                instances.scores = torch.tensor(
                    [cast(float, row["score"]) for row in rows], dtype=torch.float32
                )
                instances.pred_classes = torch.tensor(
                    [cast(int, row["category_id"]) - 1 for row in rows],
                    dtype=torch.long,
                )
                inputs.append(record)
                outputs.append({"instances": instances})
            evaluator.process(inputs, outputs)
            source = evaluator.evaluate(img_ids=subset_ids)
        assert state.effective.num_classes == 4
        assert MetadataCatalog.get("layout_val").thing_classes

    package_results = cast(CocoMetricResults, package["results"])
    package_metrics = package_results["bbox"]
    source_metrics = source["bbox"]
    assert package_metrics.keys() == source_metrics.keys()
    for name in package_metrics:
        package_value = float(package_metrics[name])
        source_value = float(source_metrics[name])
        if math.isnan(source_value):
            assert math.isnan(package_value), name
        else:
            assert package_value == pytest.approx(source_value, abs=1e-12, rel=0.0), (
                name
            )
