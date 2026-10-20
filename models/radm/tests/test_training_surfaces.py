from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from PIL import Image
from torch import nn

from radm import RADMConfig
from radm.evaluation import evaluate_checkpoint, layout_predictions_to_coco
from radm.modeling_radm import RADMDenoiser, RADMDenoiserOutput
from radm.training.config import effective_radm_config
from radm.training.datamodule import (
    RADMDataModule,
    RADMTrainingSampler,
)
from radm.training.optim import build_radm_optimizer, build_radm_scheduler
from radm.training.topology import (
    assert_effective_runtime_state,
    assert_forward_parity,
    assert_optimizer_scheduler_parity,
    assert_radm_topology_parity,
    build_reviewed_state_key_map,
    compare_state_dict_topology,
    copy_reviewed_state_dict,
)


def _write_split(root: Path) -> tuple[Path, Path, Path]:
    annotations = root / "annotations.json"
    images = root / "images"
    features = root / "features"
    images.mkdir()
    features.mkdir()
    Image.new("RGB", (16, 12), color=(20, 30, 40)).save(images / "sample.png")
    torch.save({"feats": [torch.zeros(1, 768)]}, features / "sample_feats.pth")
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "sample.png", "width": 16, "height": 12}
                ],
                "annotations": [
                    {"image_id": 7, "bbox": [1, 2, 4, 3], "category_id": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    return annotations, images, features


def test_training_datamodule_builds_all_loaders(tmp_path: Path) -> None:
    annotations, images, features = _write_split(tmp_path)
    effective = replace(effective_radm_config(), max_iter=1)
    module = RADMDataModule(
        train_annotations=annotations,
        train_image_root=images,
        train_text_feature_root=features,
        val_annotations=annotations,
        val_image_root=images,
        val_text_feature_root=features,
        batch_size=1,
        effective=effective,
    )

    module.setup()
    assert module.train_dataset is not None
    assert module.val_dataset is not None
    assert module.test_dataset is not None
    train_loader = module.train_dataloader()
    assert len(train_loader) == 1
    assert next(iter(train_loader))["image_ids"].tolist() == [7]
    assert module.val_dataloader() is not None
    assert module.test_dataloader() is not None


def test_training_sampler_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RADMTrainingSampler(0, seed=1)


def test_layout_predictions_support_absolute_boxes_and_custom_categories() -> None:
    predictions = layout_predictions_to_coco(
        image_ids=[7],
        boxes_xyxy=torch.tensor([[[1.0, 2.0, 5.0, 7.0], [0.0, 0.0, 1.0, 1.0]]]),
        labels=torch.tensor([[3, 0]]),
        mask=torch.tensor([[True, False]]),
        scores=torch.tensor([[0.75, 0.1]]),
        image_scales=torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
        original_image_scales=torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
        boxes_are_absolute=True,
        category_id_map={3: 42},
    )

    assert predictions == [
        {
            "image_id": 7,
            "bbox": [1.0, 2.0, 4.0, 5.0],
            "score": 0.75,
            "category_id": 42,
        }
    ]


def _install_fake_coco(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCOCO:
        def __init__(self, annotation_path: str) -> None:
            self.annotation_path = annotation_path

        def loadRes(self, predictions: object) -> object:
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
            self.params = SimpleNamespace(imgIds=[])
            self.stats = torch.tensor([0.5, 0.6, 0.4, 0.3, 0.2, 0.1]).numpy()
            self.eval = {"precision": torch.ones(1, 1, 1, 1, 1).numpy()}

        def evaluate(self) -> None:
            return None

        def accumulate(self) -> None:
            return None

        def summarize(self) -> None:
            return None

    import sys
    import types

    pycocotools = types.ModuleType("pycocotools")
    coco = types.ModuleType("pycocotools.coco")
    cocoeval = types.ModuleType("pycocotools.cocoeval")
    setattr(coco, "COCO", FakeCOCO)
    setattr(cocoeval, "COCOeval", FakeCOCOeval)
    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval)


def test_evaluate_checkpoint_runs_one_test_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_coco(monkeypatch)
    import radm.training.lightning_module as lightning_module

    config = RADMConfig(
        num_classes=1,
        num_proposals=1,
        hidden_dim=8,
        text_feature_dim=4,
        backbone_depth=18,
        inference_steps=1,
        num_train_timesteps=4,
    )
    effective = replace(effective_radm_config(), sample_step=1)

    class FakeModel:
        def __call__(self, **kwargs: object) -> RADMDenoiserOutput:
            boxes = kwargs["boxes_xyxy"]
            assert isinstance(boxes, torch.Tensor)
            batch_size = boxes.shape[0]
            absolute = torch.tensor([[[1.0, 2.0, 5.0, 7.0]]], dtype=boxes.dtype).expand(
                batch_size, -1, -1
            )
            logits = torch.full((batch_size, 1, 1), 10.0, dtype=boxes.dtype)
            return RADMDenoiserOutput(
                logits=logits,
                boxes_xyxy=boxes,
                pred_original_sample=boxes,
                pred_noise=boxes,
                auxiliary_boxes_absolute_xyxy=absolute.unsqueeze(0),
            )

    class FakeModule:
        model = FakeModel()

        @classmethod
        def load_from_checkpoint(cls, *args: object, **kwargs: object) -> "FakeModule":
            return cls()

        def to(self, device: str | torch.device) -> "FakeModule":
            assert str(device) == "cpu"
            return self

        def eval(self) -> "FakeModule":
            return self

    annotations, _, _ = _write_split(tmp_path)
    batch = {
        "images": torch.zeros(1, 3, 12, 16),
        "image_scales": torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
        "original_image_scales": torch.tensor([[16.0, 12.0, 16.0, 12.0]]),
        "text_features": torch.zeros(1, 20, 768),
        "text_mask": torch.ones(1, 20, 1, dtype=torch.bool),
    }
    data_module = SimpleNamespace(
        test_dataset=SimpleNamespace(
            annotation_path=annotations,
            images=[{"id": 7}],
        ),
        setup=lambda stage: None,
        test_dataloader=lambda: [batch],
    )
    monkeypatch.setattr(lightning_module, "RADMTrainingModule", FakeModule)

    report = evaluate_checkpoint(
        tmp_path / "model.ckpt",
        config=config,
        effective=effective,
        data_module=cast(RADMDataModule, data_module),
        output_dir=tmp_path / "output",
    )

    assert report["prediction_count"] == 1
    assert (tmp_path / "output" / "metrics.json").is_file()


def test_topology_helpers_cover_reviewed_mapping_and_runtime_state() -> None:
    from radm.configuration_radm import RADM_DIFFUSION_STATE_KEYS

    keys = [
        next(iter(RADM_DIFFUSION_STATE_KEYS)),
        "head.time_mlp.1.weight",
        "head.blocks.0.self_attn.in_proj_weight",
        "backbone.body.fpn.inner_blocks.0.0.weight",
        "backbone.body.fpn.inner_blocks.3.0.bias",
        "backbone.body.fpn.layer_blocks.0.0.weight",
        "backbone.body.fpn.layer_blocks.3.0.bias",
        "backbone.body.body.conv1.weight",
        "backbone.body.body.bn1.weight",
        "backbone.body.body.layer1.0.downsample.0.weight",
        "backbone.body.body.layer2.0.downsample.1.weight",
        "backbone.body.body.layer3.0.bn1.weight",
        "backbone.body.body.layer4.0.bn2.weight",
        "backbone.body.body.layer4.0.bn3.weight",
    ]

    package_state = {key: torch.zeros(1) for key in keys}
    package_model = SimpleNamespace(state_dict=lambda: package_state)
    mapping = build_reviewed_state_key_map(
        nn.Identity(), cast(RADMDenoiser, package_model)
    )
    assert mapping["head.blocks.0.self_attn.in_proj_weight"] == (
        "head.head_series.0.self_attn.in_proj_weight"
    )
    with pytest.raises(AssertionError, match="Unreviewed"):
        build_reviewed_state_key_map(
            nn.Identity(),
            cast(
                RADMDenoiser,
                SimpleNamespace(state_dict=lambda: {"unknown": torch.zeros(1)}),
            ),
        )

    reference = {
        value: tensor.clone()
        for key, value in mapping.items()
        for tensor in [package_state[key]]
    }
    compare_state_dict_topology(reference, package_state, mapping)
    with pytest.raises(AssertionError, match="Unexpected"):
        compare_state_dict_topology(
            {**reference, "extra": torch.zeros(1)}, package_state, mapping
        )

    source = nn.Linear(1, 1, bias=False)
    target = nn.Linear(1, 1, bias=False)
    target.weight.data.zero_()
    copy_reviewed_state_dict(source, cast(RADMDenoiser, target), {"weight": "weight"})
    assert torch.equal(source.weight, target.weight)
    assert_radm_topology_parity(
        source, cast(RADMDenoiser, target), {"weight": "weight"}
    )
    assert_forward_parity(
        {
            "auxiliary_logits": torch.zeros(1),
            "auxiliary_boxes_xyxy": torch.zeros(1),
            "pred_original_sample": torch.zeros(1),
            "pred_noise": torch.zeros(1),
        },
        RADMDenoiserOutput(
            logits=torch.zeros(1),
            boxes_xyxy=torch.zeros(1),
            pred_original_sample=torch.zeros(1),
            pred_noise=torch.zeros(1),
            auxiliary_logits=torch.zeros(1),
            auxiliary_boxes_xyxy=torch.zeros(1),
        ),
    )

    effective = effective_radm_config()
    runtime_summary = {
        "optimizer": effective.optimizer,
        "scheduler_interval": effective.scheduler_interval,
        "box_renewal": effective.box_renewal,
        "use_ensemble": effective.use_ensemble,
        "ema_enabled": effective.ema_enabled,
        "amp_enabled": effective.amp_enabled,
        "ddp_enabled": effective.ddp_enabled,
        "simple_trainer": effective.simple_trainer,
        "transform_names": effective.transform_names,
        "crop_transform_names": effective.crop_transform_names,
        "min_size_train": effective.min_size_train,
        "max_size_train": effective.max_size_train,
        "min_size_train_sampling": effective.min_size_train_sampling,
        "text_feature_dim": effective.text_feature_dim,
        "max_text_num": effective.max_text_num,
        "text_mask_semantics": "true_valid_false_padding",
        "missing_text_fallback": "zero_features_all_padding",
    }
    config = SimpleNamespace(
        num_train_timesteps=effective.num_train_timesteps,
        snr_scale=effective.snr_scale,
        sample_step=effective.sample_step,
        with_vtram=effective.with_vtram,
        with_gram=effective.with_gram,
        deep_supervision=effective.deep_supervision,
        backbone_freeze_at=effective.backbone_freeze_at,
    )
    buffers = {key: torch.zeros(1) for key in RADM_DIFFUSION_STATE_KEYS}
    state = SimpleNamespace(
        effective=effective,
        model=SimpleNamespace(state_dict=lambda: buffers),
        runtime_summary=runtime_summary,
    )
    package = SimpleNamespace(config=config, state_dict=lambda: buffers)
    assert_effective_runtime_state(state, cast(RADMDenoiser, package))


def test_optimizer_scheduler_parity_and_training_module_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective = replace(effective_radm_config(), ota_k=1, max_iter=1)
    first = nn.Linear(1, 1, bias=False)
    second = nn.Linear(1, 1, bias=False)
    first_optimizer = build_radm_optimizer(first, effective)
    second_optimizer = build_radm_optimizer(second, effective)
    first_scheduler = build_radm_scheduler(first_optimizer, effective)
    second_scheduler = build_radm_scheduler(second_optimizer, effective)
    assert_optimizer_scheduler_parity(
        first_optimizer,
        second_optimizer,
        first_scheduler,
        second_scheduler,
        effective,
    )

    lightning_module = pytest.importorskip("radm.training.lightning_module")

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(num_classes=2)

        def prepare_diffusion_concat(
            self, boxes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return boxes, torch.zeros_like(boxes), torch.tensor([3])

        def forward(self, **kwargs: object) -> RADMDenoiserOutput:
            boxes = kwargs["boxes_xyxy"]
            scales = kwargs["image_scales"]
            assert isinstance(boxes, torch.Tensor)
            assert isinstance(scales, torch.Tensor)
            logits = self.scale.expand(boxes.shape[0], boxes.shape[1], 2)
            absolute = boxes * scales[:, None, :]
            return RADMDenoiserOutput(
                logits=logits,
                boxes_xyxy=boxes,
                pred_original_sample=boxes,
                pred_noise=boxes,
                auxiliary_logits=logits.unsqueeze(0),
                auxiliary_boxes_xyxy=boxes.unsqueeze(0),
                auxiliary_boxes_absolute_xyxy=absolute.unsqueeze(0),
            )

    module = lightning_module.RADMTrainingModule(model=TinyModel(), effective=effective)
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = {
        "boxes_xyxy": torch.tensor([[[0.1, 0.1, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0]]]),
        "mask": torch.tensor([[True, False]]),
        "labels": torch.tensor([[0, 0]]),
        "image_scales": torch.tensor([[100.0, 100.0, 100.0, 100.0]]),
        "forward_image_scales": torch.tensor([[100.0, 100.0, 100.0, 100.0]]),
        "images": torch.zeros(1, 3, 8, 8),
        "text_features": torch.zeros(1, 1, 4),
        "text_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }
    assert module.forward(
        boxes_xyxy=batch["boxes_xyxy"],
        timesteps=torch.tensor([3]),
        text_features=batch["text_features"],
        text_mask=batch["text_mask"],
        images=batch["images"],
        image_scales=batch["image_scales"],
    ).logits.shape == (1, 2, 2)
    assert module.training_step(batch, 0).ndim == 0
    assert module.validation_step(batch, 0).ndim == 0
    assert module.latest_step_trace
    configured = module.configure_optimizers()
    assert configured["lr_scheduler"]["interval"] == "step"


def test_training_module_rejects_missing_model_and_ema() -> None:
    lightning_module = pytest.importorskip("radm.training.lightning_module")
    effective = effective_radm_config()
    with pytest.raises(ValueError, match="requires a model"):
        lightning_module.RADMTrainingModule(effective=effective)
    with pytest.raises(NotImplementedError, match="EMA"):
        lightning_module.RADMTrainingModule(
            model=nn.Identity(),
            effective=replace(effective, ema_enabled=True),
        )
