import json
import sys
import types
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn


def _install_optional_training_stubs() -> None:
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        datasets = types.ModuleType("datasets")

        class Dataset:
            @classmethod
            def from_file(cls, path: str) -> str:
                return path

        setattr(datasets, "DatasetDict", dict)
        setattr(datasets, "Dataset", Dataset)
        setattr(datasets, "concatenate_datasets", lambda datasets: datasets[0])
        sys.modules["datasets"] = datasets

    try:
        import lightning.pytorch  # noqa: F401
    except ModuleNotFoundError:
        lightning_stub = types.ModuleType("lightning")
        pytorch_stub = types.ModuleType("lightning.pytorch")

        class LightningDataModule:
            def __init__(self) -> None:
                pass

        class LightningModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._trainer = None

            def save_hyperparameters(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

            def optimizers(self) -> list[torch.optim.Optimizer]:
                raise RuntimeError("test stub requires an optimizer override")

            def lr_schedulers(self) -> list[object]:
                raise RuntimeError("test stub requires a scheduler override")

            def manual_backward(self, loss: torch.Tensor) -> None:
                loss.backward()

            def configure_gradient_clipping(
                self,
                optimizer: torch.optim.Optimizer,
                gradient_clip_val: float | None = None,
                gradient_clip_algorithm: str | None = None,
            ) -> None:
                del optimizer, gradient_clip_val, gradient_clip_algorithm

        setattr(pytorch_stub, "LightningDataModule", LightningDataModule)
        setattr(pytorch_stub, "LightningModule", LightningModule)
        utilities = types.ModuleType("lightning.pytorch.utilities")
        types_module = types.ModuleType("lightning.pytorch.utilities.types")
        setattr(types_module, "OptimizerLRScheduler", object)
        sys.modules.update(
            {
                "lightning": lightning_stub,
                "lightning.pytorch": pytorch_stub,
                "lightning.pytorch.utilities": utilities,
                "lightning.pytorch.utilities.types": types_module,
            }
        )


_install_optional_training_stubs()

from ds_gan import DSGANConfig, DSGANModel  # noqa: E402
from ds_gan.modeling_ds_gan import DSGANModelOutput  # noqa: E402
from ds_gan.training import (  # noqa: E402
    DSGANDataModule,
    DSGANDataset,
    DSGANSetCriterion,
    DSGANTrainingModule,
    HungarianMatcher,
    build_bridge_manifest,
    explicit_setup_seed,
    vendor_random_initial_layout,
)
from ds_gan.training import datamodule as datamodule_module  # noqa: E402
from ds_gan.training import dataset as dataset_module  # noqa: E402
from ds_gan.training.discriminator import DSGANDiscriminator  # noqa: E402
from ds_gan.training.discriminator import (  # noqa: E402
    _LayoutArgmax,
    _cxcywh_to_xyxy,
    _first_connection,
    _layout_order,
)


def _config() -> DSGANConfig:
    return DSGANConfig(
        backbone="resnet18",
        max_elem=2,
        hidden_size=16,
        num_layers=1,
        image_size=(32, 32),
        backbone_feature_size=4,
    )


def _row() -> dict[str, object]:
    image = Image.new("RGB", (8, 8), (80, 100, 120))
    saliency = Image.new("L", (8, 8), 160)
    return {
        "inpainted_poster": image,
        "canvas": image,
        "pfpn_saliency_map": saliency,
        "basnet_saliency_map": Image.new("L", (8, 8), 80),
        "annotations": {
            "cls_elem": [0, 1, 3],
            "box_elem": [[0, 0, 4, 4], [1, 1, 3, 5], [0, 0, 1, 1]],
        },
    }


class _TinyGenerator(nn.Module):
    def __init__(self, max_elem: int) -> None:
        super().__init__()
        self.max_elem = max_elem
        self.resnet_fpn = nn.Linear(1, 4)
        self.head = nn.Linear(4, max_elem * 4)
        self.box_head = nn.Linear(4, max_elem * 4)

    def forward(
        self, pixel_values: torch.Tensor, layout: torch.Tensor
    ) -> DSGANModelOutput:
        del layout
        features = self.resnet_fpn(pixel_values.mean(dim=(1, 2, 3)).unsqueeze(-1))
        class_probs = torch.softmax(
            self.head(features).reshape(-1, self.max_elem, 4), dim=-1
        )
        bbox = torch.sigmoid(self.box_head(features).reshape(-1, self.max_elem, 4))
        return DSGANModelOutput(class_probs=class_probs, bbox=bbox)


class _TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet_fpn = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)

    def forward(self, pixel_values: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        image_score = self.resnet_fpn(pixel_values.mean(dim=(1, 2, 3)).unsqueeze(-1))
        layout_score = self.head(layout.mean(dim=(1, 2, 3)).unsqueeze(-1))
        return torch.tanh(image_score + layout_score)


class _BadGenerator(nn.Module):
    def forward(
        self, *, pixel_values: torch.Tensor, layout: torch.Tensor
    ) -> torch.Tensor:
        del layout
        return pixel_values


class _NoBoxGenerator(nn.Module):
    def forward(
        self, *, pixel_values: torch.Tensor, layout: torch.Tensor
    ) -> DSGANModelOutput:
        del layout
        return DSGANModelOutput(
            class_probs=torch.full(
                (pixel_values.shape[0], 2, 4),
                0.25,
                dtype=pixel_values.dtype,
            )
        )


def _training_batch() -> dict[str, torch.Tensor]:
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25]]])
    classes = torch.nn.functional.one_hot(labels, num_classes=4).float()
    return {
        "pixel_values": torch.ones(1, 4, 32, 32),
        "labels": labels,
        "boxes": boxes,
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "layout": torch.stack((classes, boxes), dim=2),
    }


def test_training_module_runs_g_then_d_and_scheduler(monkeypatch: pytest.MonkeyPatch):
    module = DSGANTrainingModule(
        config=_config(),
        generator=cast(DSGANModel, _TinyGenerator(2)),
        discriminator=cast(DSGANDiscriminator, _TinyDiscriminator()),
        scheduler_generator_milestones=(0, 1),
        scheduler_discriminator_milestones=(0, 1),
    )
    optimizers, schedulers = cast(
        tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.MultiStepLR]],
        module.configure_optimizers(),
    )
    batch = _training_batch()
    initial_layout = torch.zeros(1, 2, 2, 4)

    loss = module.step_with_optimizers(
        batch,
        optimizers[0],
        optimizers[1],
        initial_layout=initial_layout,
        epoch=2,
    )
    assert loss.ndim == 0
    assert module._step_index == 1
    assert set(module.latest_step_trace) == {
        "initial_layout",
        "class_probs",
        "bbox",
        "discriminator_generated",
        "discriminator_fake",
        "discriminator_real",
        "loss_g_adv",
        "loss_reconstruction",
        "loss_g",
        "loss_d_fake",
        "loss_d_real",
        "loss_d",
    }

    monkeypatch.setattr(module, "optimizers", lambda: optimizers)
    module.training_step(batch, 0)
    monkeypatch.setattr(module, "lr_schedulers", lambda: schedulers)
    module.on_train_epoch_end()
    assert module._step_index == 2
    assert module._parameter_groups(module.generator)[0]

    monkeypatch.setattr(module, "generator", _BadGenerator())
    with pytest.raises(TypeError, match="must return"):
        module.forward(batch["pixel_values"], initial_layout)

    monkeypatch.setattr(module, "optimizers", lambda: [])
    with pytest.raises(RuntimeError, match="two Lightning optimizers"):
        module.training_step(batch, 0)

    monkeypatch.setattr(module, "generator", _NoBoxGenerator())
    with pytest.raises(RuntimeError, match="bounding boxes"):
        module.step_with_optimizers(
            batch, optimizers[0], optimizers[1], initial_layout=initial_layout
        )
    monkeypatch.setattr(module, "lr_schedulers", lambda: object())
    module.on_train_epoch_end()


def test_training_module_accepts_explicit_discriminator_config():
    module = DSGANTrainingModule(
        config=_config(),
        discriminator_config=_config(),
        generator=cast(DSGANModel, _TinyGenerator(2)),
        discriminator=cast(DSGANDiscriminator, _TinyDiscriminator()),
    )
    assert module.ds_gan_config.max_elem == 2


def test_rng_helpers_are_seeded_and_explicit(monkeypatch: pytest.MonkeyPatch):
    explicit_setup_seed(12, deterministic=False)
    first = torch.rand(3)
    explicit_setup_seed(12, deterministic=False)
    assert torch.equal(first, torch.rand(3))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    explicit_setup_seed(12)
    layout = vendor_random_initial_layout(
        2,
        3,
        numpy_rng=np.random.RandomState(0),
        torch_generator=torch.Generator().manual_seed(0),
    )
    assert layout.shape == (2, 3, 2, 4)
    assert torch.equal(layout[:, :, 0].sum(-1), torch.ones(2, 3))


def test_dataset_bridge_and_manifest(tmp_path):
    train = DSGANDataset([_row()], split="train", max_elem=2)
    test = DSGANDataset([_row()], split="test", max_elem=2)
    assert len(train) == 1
    assert train[0]["layout"].shape == (2, 2, 4)
    assert test[0]["pixel_values"].shape == (4, 350, 240)

    with pytest.raises(KeyError, match="lacks required"):
        dataset_module.bridge_example({}, split="train")
    manifest = build_bridge_manifest(
        source_revision="revision",
        train_count=2,
        test_count=1,
        source_hashes={"train.arrow": "abc"},
        available_fields=["canvas"],
    )
    assert manifest["missing_source_fields"]

    payload = tmp_path / "payload"
    payload.write_bytes(b"ds-gan")
    assert dataset_module.hash_file(payload) == (
        "0d67f2794b6c885e7cdb54160e2de45004a166f5ef095646a509a981cfe910f1"
    )


def test_cached_manifest_and_missing_cache_errors(tmp_path):
    cache = tmp_path / "cache"
    dataset_dir = (
        cache
        / "creative-graphic-design___pku-poster_layout"
        / "default"
        / "0.0.0"
        / "revision"
    )
    dataset_dir.mkdir(parents=True)
    metadata = {
        "download_checksums": {"source@revision/dataset": "hash"},
        "splits": {"train": {"num_examples": 2}, "test": {"num_examples": 1}},
        "features": {
            field: {"dtype": "string"}
            for field in dataset_module.SOURCE_FIELDS.values()
        },
    }
    (dataset_dir / "dataset_info.json").write_text(json.dumps(metadata))
    (dataset_dir / "pku-poster_layout-train-0.arrow").write_bytes(b"train")
    (dataset_dir / "pku-poster_layout-test.arrow").write_bytes(b"test")
    manifest = dataset_module.manifest_from_cached_dataset(cache)
    source_manifest = cast(dict[str, object], manifest["source"])
    assert source_manifest["revision"] == "revision"
    assert source_manifest["split_counts"] == {"train": 2, "test": 1}

    loaded = dataset_module.load_cached_dataset(cache)
    assert loaded["train"].endswith("train-0.arrow")
    assert dataset_module.load_cached_test(cache).endswith("test.arrow")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        dataset_module.load_cached_dataset(empty)
    with pytest.raises(FileNotFoundError):
        dataset_module.load_cached_test(empty)
    incomplete = tmp_path / "incomplete"
    incomplete_dataset = (
        incomplete
        / "creative-graphic-design___pku-poster_layout"
        / "default"
        / "0.0.0"
        / "revision"
    )
    incomplete_dataset.mkdir(parents=True)
    (incomplete_dataset / "dataset_info.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Arrow files"):
        dataset_module.load_cached_dataset(incomplete)
    with pytest.raises(FileNotFoundError, match="TEST Arrow"):
        dataset_module.load_cached_test(incomplete)


def test_datamodule_builds_seeded_train_and_test_loaders(monkeypatch):
    source = {"train": [_row()], "test": [_row()]}
    monkeypatch.setattr(datamodule_module, "load_cached_dataset", lambda _: source)
    data = DSGANDataModule(batch_size=1, test_batch_size=1, max_elem=2)
    assert data.train_dataloader().batch_size == 1
    assert data.test_dataloader().batch_size == 1
    assert DSGANDataModule().test_dataloader().batch_size == 4
    with pytest.raises(ValueError, match="unsupported DS-GAN dataset"):
        DSGANDataModule(dataset_name="other")


def test_losses_match_and_discriminator_orders_layouts():
    matcher = HungarianMatcher()
    with pytest.raises(ValueError, match="at least one"):
        HungarianMatcher(0, 0, 0)
    outputs = {
        "pred_logits": torch.randn(2, 2, 4),
        "pred_boxes": torch.rand(2, 2, 4),
    }
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
        {"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)},
    ]
    assert len(matcher(outputs, targets)) == 2
    criterion = DSGANSetCriterion()
    losses = criterion(
        outputs["pred_logits"][:1],
        outputs["pred_boxes"][:1],
        targets[:1],
    )
    assert set(losses) == {"loss_ce", "loss_bbox", "loss_giou"}

    classes = torch.eye(4)[torch.tensor([[2, 1, 3, 0]])]
    boxes = torch.full((1, 4, 4), 0.5)
    boxes[:, :, 2:] = 0.4
    layout = torch.stack((classes, boxes), dim=2).detach().requires_grad_()
    ordered = _LayoutArgmax.apply(layout)
    ordered.sum().backward()
    assert layout.grad is not None
    assert _first_connection(2) == 2
    assert _first_connection([2, 3]) == 2
    assert _cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.4, 0.2]])).shape == (1, 4)
    assert len(_layout_order(torch.tensor([2, 1, 3, 0]), boxes[0])) == 4
    assert len(_layout_order(torch.tensor([0, 0, 3, 3]), boxes[0])) == 2


def test_discriminator_forward_on_tiny_cpu_model():
    discriminator = DSGANDiscriminator(_config()).eval()
    output = discriminator(torch.zeros(1, 4, 32, 32), torch.zeros(1, 2, 2, 4))
    assert output.shape == (1, 1)
