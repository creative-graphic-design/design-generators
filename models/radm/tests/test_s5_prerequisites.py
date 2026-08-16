from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from radm.evaluation import layout_predictions_to_coco
from radm.training.config import effective_radm_config
from radm.training.datamodule import RADMDataModule


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
    assert torch.equal(batch["image_scales"][:, 0], batch["image_scales"][:, 2])
    assert torch.equal(batch["image_scales"][:, 1], batch["image_scales"][:, 3])
    assert bool(torch.all(batch["image_scales"] > 0))
    assert batch["labels"].shape == (1, 100)


def test_layout_predictions_use_coco_xywh_and_dataset_category_ids() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[0.125, 0.25, 0.375, 0.5]]]),
        labels=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
        scores=torch.tensor([[0.75]]),
        image_scales=torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
    )

    assert predictions == [
        {
            "image_id": 7,
            "bbox": [2.0, 3.0, 4.0, 3.0],
            "score": 0.75,
            "category_id": 1,
        }
    ]
