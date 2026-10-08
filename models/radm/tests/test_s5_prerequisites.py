from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import cast

import numpy as np
import torch
import pytest
from PIL import Image
from torch.utils.data import SequentialSampler

from radm import RADMConfig
from radm.evaluation import (
    evaluate_cgl_predictions,
    evaluate_checkpoint,
    layout_predictions_to_coco,
)
from radm.training.config import effective_radm_config
from radm.training.datamodule import RADMDataModule
from radm.training.dataset import RADMCOCODataset


def _write_test_split(root: Path) -> tuple[Path, Path, Path]:
    annotations = root / "annotations.json"
    images = root / "images"
    features = root / "text_features"
    images.mkdir()
    features.mkdir()
    image_name = "sample.png"
    Image.new("RGB", (16, 12), color=(20, 30, 40)).save(images / image_name)
    torch.save(
        {"feats": [torch.zeros(1, 768)]},
        features / "sample_feats.pth",
    )
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": image_name, "width": 16, "height": 12}
                ],
                "annotations": [
                    {
                        "image_id": 7,
                        "bbox": [1, 2, 4, 3],
                        "category_id": 1,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return annotations, images, features


def _install_fake_coco_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCOCO:
        def __init__(self, annotation_path: str) -> None:
            self.annotation_path = annotation_path

        def loadRes(self, predictions: list[dict[str, object]]) -> object:
            return predictions

        def getCatIds(self) -> list[int]:
            return [1]

        def loadCats(self, category_ids: list[int]) -> list[dict[str, str]]:
            assert category_ids == [1]
            return [{"name": "box"}]

    class FakeCOCOeval:
        def __init__(self, coco: FakeCOCO, detections: object, task: str) -> None:
            assert isinstance(coco, FakeCOCO)
            assert detections is not None
            assert task == "bbox"
            self.params = types.SimpleNamespace(imgIds=[])
            self.stats = np.asarray([0.5, 0.6, 0.4, 0.3, 0.2, 0.1])
            self.eval = {"precision": np.ones((1, 1, 1, 1, 1))}

        def evaluate(self) -> None:
            return None

        def accumulate(self) -> None:
            return None

        def summarize(self) -> None:
            return None

    pycocotools = types.ModuleType("pycocotools")
    coco = types.ModuleType("pycocotools.coco")
    cocoeval = types.ModuleType("pycocotools.cocoeval")
    setattr(coco, "COCO", FakeCOCO)
    setattr(cocoeval, "COCOeval", FakeCOCOeval)
    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval)


def test_evaluate_cgl_predictions_writes_empty_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_coco_api(monkeypatch)
    report = evaluate_cgl_predictions(
        tmp_path / "annotations.json",
        [],
        output_dir=tmp_path / "empty-output",
        image_ids=[7],
    )

    assert report["prediction_count"] == 0
    assert report["image_ids"] == [7]
    assert (tmp_path / "empty-output" / "metrics.json").is_file()


def test_evaluate_cgl_predictions_serializes_fake_coco_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_coco_api(monkeypatch)
    predictions = [
        {
            "image_id": 7,
            "bbox": [1.0, 2.0, 4.0, 3.0],
            "score": 0.75,
            "category_id": 1,
        }
    ]

    report = evaluate_cgl_predictions(
        tmp_path / "annotations.json",
        predictions,
        output_dir=tmp_path / "metrics-output",
        image_ids=[7],
    )

    results = report["results"]
    assert isinstance(results, dict)
    assert results["bbox"]["AP"] == 50.0
    assert results["bbox"]["AP-box"] == 100.0
    assert report["results_path"]
    assert report["metrics_path"]


def test_layout_predictions_reject_misaligned_image_ids() -> None:
    with pytest.raises(ValueError, match="image_ids must align"):
        layout_predictions_to_coco(
            image_ids=[],
            boxes_xyxy=torch.zeros(1, 1, 4),
            labels=torch.zeros(1, 1, dtype=torch.long),
            mask=torch.ones(1, 1, dtype=torch.bool),
            scores=torch.ones(1, 1),
            image_scales=torch.ones(1, 4),
            original_image_scales=torch.ones(1, 4),
        )


def test_evaluate_checkpoint_rejects_missing_test_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTrainingModule:
        model = object()

        @classmethod
        def load_from_checkpoint(
            cls, *args: object, **kwargs: object
        ) -> "FakeTrainingModule":
            return cls()

        def to(self, device: str) -> "FakeTrainingModule":
            assert device == "cpu"
            return self

        def eval(self) -> "FakeTrainingModule":
            return self

    class EmptyDataModule:
        test_dataset = None

        def setup(self, stage: str) -> None:
            assert stage == "test"

        def test_dataloader(self) -> None:
            return None

    import radm.training.lightning_module as lightning_module

    monkeypatch.setattr(lightning_module, "RADMTrainingModule", FakeTrainingModule)
    with pytest.raises(ValueError, match="test annotations"):
        evaluate_checkpoint(
            tmp_path / "checkpoint.ckpt",
            config=RADMConfig(),
            effective=effective_radm_config(),
            data_module=cast(RADMDataModule, EmptyDataModule()),
        )


def test_data_module_exposes_the_approved_test_stream(tmp_path: Path) -> None:
    annotations, images, features = _write_test_split(tmp_path)
    effective = effective_radm_config()
    module = RADMDataModule(
        train_annotations=annotations,
        train_image_root=images,
        train_text_feature_root=features,
        val_annotations=annotations,
        val_image_root=images,
        val_text_feature_root=features,
        test_annotations=annotations,
        test_image_root=images,
        test_text_feature_root=features,
        allow_missing_text_features=False,
        effective=effective,
    )

    module.setup("test")

    assert module.test_dataset is not None
    loader = module.test_dataloader()
    assert loader is not None
    assert isinstance(loader.sampler, SequentialSampler)
    batch = next(iter(loader))
    assert batch["image_scales"].shape == (1, 4)
    assert batch["image_scales"].tolist() == [[1067.0, 800.0, 1067.0, 800.0]]
    assert batch["original_image_scales"].tolist() == [[16.0, 12.0, 16.0, 12.0]]
    assert batch["labels"].shape == (1, 100)


def test_evaluation_dataset_uses_fixed_resize_without_flip(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    images = tmp_path / "images"
    features = tmp_path / "text_features"
    images.mkdir()
    features.mkdir()
    Image.new("RGB", (350, 520), color=(20, 30, 40)).save(images / "sample.png")
    torch.save({"feats": [torch.zeros(1, 768)]}, features / "sample_feats.pth")
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "sample.png", "width": 350, "height": 520}
                ],
                "annotations": [
                    {"image_id": 7, "bbox": [100, 100, 200, 100], "category_id": 1}
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = RADMCOCODataset(
        annotation_path=annotations,
        image_root=images,
        text_feature_root=features,
        effective=effective_radm_config(),
        allow_missing_text_features=False,
        train=False,
    )
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 1189, 800)
    assert sample["image_size_xyxy"].tolist() == [800.0, 1189.0, 800.0, 1189.0]
    assert sample["original_image_size_xyxy"].tolist() == [350.0, 520.0, 350.0, 520.0]
    torch.testing.assert_close(
        sample["boxes_xyxy"],
        torch.tensor(
            [
                [
                    torch.trunc(torch.tensor(100 * 800 / 350)) / 800,
                    torch.trunc(torch.tensor(100 * 1189 / 520)) / 1189,
                    torch.trunc(torch.tensor(300 * 800 / 350)) / 800,
                    torch.trunc(torch.tensor(200 * 1189 / 520)) / 1189,
                ]
            ]
        ),
    )


def test_evaluation_dataset_can_reproduce_source_missing_text_branch(
    tmp_path: Path,
) -> None:
    annotations, images, features = _write_test_split(tmp_path)
    dataset = RADMCOCODataset(
        annotation_path=annotations,
        image_root=images,
        text_feature_root=features,
        effective=effective_radm_config(),
        allow_missing_text_features=False,
        train=False,
        read_text_features=False,
    )

    sample = dataset[0]

    assert torch.count_nonzero(sample["text_features"]) == 0
    assert not bool(sample["text_mask"].any())


def test_layout_predictions_are_clipped_in_original_image_frame() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[-0.1, -0.2, 1.2, 1.3]]]),
        labels=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
        scores=torch.tensor([[0.75]]),
        image_scales=torch.tensor([[800.0, 1189.0, 800.0, 1189.0]]),
        original_image_scales=torch.tensor([[350.0, 520.0, 350.0, 520.0]]),
    )

    assert predictions == [
        {
            "image_id": 7,
            "bbox": [0.0, 0.0, 350.0, 520.0],
            "score": 0.75,
            "category_id": 1,
        }
    ]


def test_layout_predictions_drop_boxes_empty_after_vendor_clipping() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[0.25, 0.25, 0.25, 0.5], [0.25, 0.25, 0.5, 0.25]]]),
        labels=torch.tensor([[0, 1]]),
        mask=torch.tensor([[True, True]]),
        scores=torch.tensor([[0.75, 0.8]]),
        image_scales=torch.tensor([[800.0, 1189.0, 800.0, 1189.0]]),
        original_image_scales=torch.tensor([[350.0, 520.0, 350.0, 520.0]]),
    )

    assert predictions == []


def test_layout_predictions_use_coco_xywh_and_dataset_category_ids() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[0.125, 0.25, 0.375, 0.5]]]),
        labels=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
        scores=torch.tensor([[0.75]]),
        image_scales=torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
        original_image_scales=torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
    )

    assert predictions == [
        {
            "image_id": 7,
            "bbox": [2.0, 3.0, 4.0, 3.0],
            "score": 0.75,
            "category_id": 1,
        }
    ]


def test_layout_predictions_reproduces_vendor_model_scale_before_postprocess() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[0.1, 0.2, 0.4, 0.5]]]),
        labels=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
        scores=torch.tensor([[0.75]]),
        image_scales=torch.tensor([[800.0, 1189.0, 800.0, 1189.0]]),
        original_image_scales=torch.tensor([[350.0, 520.0, 350.0, 520.0]]),
        model_image_scales=torch.tensor([[1189.0, 800.0, 1189.0, 800.0]]),
    )

    assert predictions[0]["image_id"] == 7
    assert predictions[0]["category_id"] == 1
    assert predictions[0]["score"] == 0.75
    assert predictions[0]["bbox"] == pytest.approx(
        [52.01875, 69.97476871320437, 156.05625, 104.96215306980656]
    )
