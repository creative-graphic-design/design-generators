import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from layoutganpp.training.datamodule import LayoutGANPPDataModule
from layoutganpp.training.dataset import (
    LayoutGANPPDataset,
    LayoutRow,
    _magazine_rows,
    _publaynet_rows,
    _rico_elements,
    _rico_rows,
    _sort_row,
    _split_indices,
    collate_layoutganpp,
    load_rows,
    synthetic_rows,
)
from layoutganpp.training.lightning_module import LayoutGANPPTrainingModule
from layoutganpp.training.modeling import LayoutGANPPDiscriminator, TransformerWithToken
from layoutganpp.training.step import (
    _generator_boxes,
    _resolve_latent_noise,
    gan_forward_trace,
    run_gan_iteration,
)


def _write_dataset_fixture(root):
    magazine = root / "magazine"
    annotations = magazine / "layoutdata" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "magazine.xml").write_text(
        """
        <annotation>
          <filename>magazine.png</filename>
          <size><width>100</width><height>200</height></size>
          <layout><element label="text" polygon_x="10 30" polygon_y="20 60"/></layout>
        </annotation>
        """
    )
    for index in range(9):
        (annotations / f"magazine-{index}.xml").write_text(
            """
            <annotation>
              <filename>magazine-extra.png</filename>
              <size><width>100</width><height>200</height></size>
              <layout><element label="text" polygon_x="10 30" polygon_y="20 60"/></layout>
            </annotation>
            """
        )

    rico = root / "rico" / "semantic_annotations"
    rico.mkdir(parents=True)
    (rico / "valid.json").write_text(
        json.dumps(
            {
                "bounds": [0, 0, 100, 200],
                "children": [
                    {
                        "componentLabel": "Text",
                        "bounds": [10, 20, 40, 70],
                        "children": [
                            {"componentLabel": "Icon", "bounds": [1, 2, 5, 8]}
                        ],
                    }
                ],
            }
        )
    )
    (rico / "landscape.json").write_text(
        json.dumps({"bounds": [0, 0, 200, 100], "children": []})
    )
    (rico / "offset.json").write_text(
        json.dumps({"bounds": [1, 0, 100, 200], "children": []})
    )

    publaynet = root / "publaynet"
    publaynet.mkdir(parents=True)
    payload = {
        "categories": [{"id": 1, "name": "text"}],
        "images": [
            {"id": 1, "file_name": "portrait.png", "width": 100, "height": 200},
            {"id": 2, "file_name": "landscape.png", "width": 200, "height": 100},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"image_id": 1, "category_id": 1, "bbox": [-1, 20, 30, 40]},
        ],
    }
    (publaynet / "train.json").write_text(json.dumps(payload))
    (publaynet / "val.json").write_text(json.dumps(payload))
    return magazine, root / "rico", root


def test_dataset_parsers_splits_and_collation(tmp_path):
    magazine, rico, publaynet = _write_dataset_fixture(tmp_path)
    magazine_rows = _magazine_rows(magazine)
    assert magazine_rows[0].bbox.shape == (1, 4)
    assert magazine_rows[0].labels.tolist() == [0]
    assert _rico_rows(rico)[0].labels.tolist() == [2, 3]
    train_rows, test_rows = _publaynet_rows(publaynet)
    assert len(train_rows) == len(test_rows) == 1
    assert _rico_elements({"children": [{"children": []}]})

    unsorted = LayoutRow(
        "row",
        torch.tensor([[0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]),
        torch.tensor([1, 0]),
        100.0,
        200.0,
    )
    assert _sort_row(unsorted).labels.tolist() == [0, 1]
    assert [
        len(_split_indices(10, "magazine", split)) for split in ("train", "val", "test")
    ] == [8, 1, 1]
    assert len(_split_indices(10, "publaynet", "test")) == 0
    with pytest.raises(ValueError, match="Unsupported split"):
        _split_indices(10, "magazine", "bad")

    assert len(load_rows("magazine", magazine, "train")) == 8
    assert len(load_rows("rico", rico, "test")) == 1
    assert len(load_rows("publaynet", publaynet, "test")) == 1
    assert len(load_rows("publaynet", publaynet, "val")) == 1
    with pytest.raises(ValueError, match="Unknown layoutganpp dataset_name"):
        load_rows("unknown", tmp_path, "train")

    rows = synthetic_rows("magazine", 3, 10)
    assert len(rows) == 3
    assert len(synthetic_rows("rico13", 1, 10)[0].labels) == 2
    assert len(synthetic_rows("publaynet", 1, 10)[0].labels) == 2
    dataset = LayoutGANPPDataset(
        dataset_name="magazine", split="train", data_root=None, synthetic_size=2, seed=7
    )
    assert len(dataset) == 2
    assert dataset[0].name == "synthetic-0"
    batch = collate_layoutganpp([rows[0], rows[1]])
    assert isinstance(batch["bbox"], torch.Tensor)
    assert batch["bbox"].shape[0] == 2
    with pytest.raises(ValueError, match="empty"):
        collate_layoutganpp([])


def test_datamodule_builds_all_loaders():
    data_module = LayoutGANPPDataModule(
        dataset_name="magazine",
        batch_size=2,
        num_workers=0,
        synthetic_size=4,
        seed=13,
        seed_mode="deterministic",
        shuffle_train=False,
    )
    data_module.setup("fit")
    assert len(data_module.train_dataset) == 4
    assert len(data_module.val_dataset) == 2
    assert len(data_module.test_dataset) == 2
    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))
    test_batch = next(iter(data_module.test_dataloader()))
    assert isinstance(train_batch["bbox"], torch.Tensor)
    assert isinstance(val_batch["bbox"], torch.Tensor)
    assert isinstance(test_batch["bbox"], torch.Tensor)
    assert train_batch["bbox"].shape[0] == 2
    assert val_batch["bbox"].shape[0] == 2
    assert test_batch["bbox"].shape[0] == 2


class _ToyGenerator(nn.Module):
    latent_size = 4

    def __init__(self, output_object=True):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.output_object = output_object

    def forward(self, latents, labels, padding_mask):
        del labels, padding_mask
        output = self.linear(latents)
        return SimpleNamespace(bbox=output) if self.output_object else output


class _ConfigGenerator(_ToyGenerator):
    def __init__(self):
        super().__init__()
        self.__dict__.pop("latent_size", None)
        self.config = SimpleNamespace(latent_size=4)


class _ToyDiscriminator(nn.Module):
    def __init__(self, labels=5):
        super().__init__()
        self.score = nn.Linear(4, 1)
        self.classifier = nn.Linear(4, labels)
        self.reconstructor = nn.Linear(4, 4)

    def forward(self, bbox, labels, padding_mask, reconst=False):
        del labels
        score = self.score(bbox).squeeze(-1).masked_fill(padding_mask, 0).mean(dim=1)
        if not reconst:
            return score
        valid = ~padding_mask
        return (
            score,
            self.classifier(bbox[valid]),
            torch.sigmoid(self.reconstructor(bbox[valid])),
        )


class _ToyOptimizer:
    def __init__(self, parameters: Iterable[torch.Tensor], lr: float = 1e-3):
        self.parameters = list(parameters)
        self.lr = lr

    @torch.no_grad()
    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter -= self.lr * parameter.grad


def test_training_models_and_step_trace():
    rows = synthetic_rows("magazine", 2, 3)
    batch = collate_layoutganpp(rows)
    latent = torch.randn(2, 3, 4)
    generator = _ToyGenerator()
    discriminator = _ToyDiscriminator()
    labels = cast(torch.Tensor, batch["labels"])
    mask = cast(torch.Tensor, batch["mask"])
    bbox = cast(torch.Tensor, batch["bbox"])
    assert (
        _generator_boxes(
            _ToyGenerator(output_object=False), latent, labels, ~mask
        ).shape[-1]
        == 4
    )
    assert _resolve_latent_noise(generator, labels, bbox, latent) is latent
    assert _resolve_latent_noise(generator, labels, bbox, None).shape == latent.shape
    assert (
        _resolve_latent_noise(_ConfigGenerator(), labels, bbox, None).shape
        == latent.shape
    )

    trace = gan_forward_trace(generator, discriminator, batch, latent_noise=latent)
    assert trace["update_order"].tolist() == [0, 1]
    assert trace["latent_noise"].shape == latent.shape
    assert trace["condition_labels"].shape == labels.shape
    trace_with_grad = gan_forward_trace(
        generator, discriminator, batch, latent_noise=latent, detach=False
    )
    assert trace_with_grad["generator_loss"].requires_grad
    optimizer_g = _ToyOptimizer(generator.parameters())
    optimizer_d = _ToyOptimizer(discriminator.parameters())
    result = run_gan_iteration(
        generator, discriminator, batch, optimizer_g, optimizer_d, latent_noise=latent
    )
    assert result["generator_gradient_norm"].numel() == 1
    callback_calls = []
    result_with_callback = run_gan_iteration(
        generator,
        discriminator,
        batch,
        optimizer_g,
        optimizer_d,
        backward=lambda loss: (
            callback_calls.append(float(loss.detach())),
            loss.backward(),
        )[1],
        latent_noise=latent,
    )
    assert result_with_callback["discriminator_gradient_norm"].numel() == 1
    assert len(callback_calls) == 2


def test_discriminator_and_lightning_module_paths(monkeypatch: pytest.MonkeyPatch):
    hidden = torch.randn(2, 2, 4)
    padding = torch.tensor([[False, True], [False, False]])
    transformer = TransformerWithToken(
        d_model=4, nhead=1, dim_feedforward=2, num_layers=1
    )
    assert transformer(hidden, padding).shape == (3, 2, 4)
    discriminator = LayoutGANPPDiscriminator(
        num_labels=5, d_model=4, nhead=1, num_layers=1, max_elements=3
    )
    bbox = torch.rand(2, 2, 4)
    labels = torch.tensor([[0, 1], [2, 3]])
    assert discriminator(bbox, labels, padding).shape == (2,)
    score, class_logits, reconstructed = discriminator(
        bbox, labels, padding, reconst=True
    )
    assert score.shape == (2,)
    assert class_logits.shape[0] == 3
    assert reconstructed.shape == (3, 4)

    module = LayoutGANPPTrainingModule(
        dataset_name="magazine",
        latent_size=4,
        generator_d_model=4,
        generator_nhead=1,
        generator_num_layers=1,
        discriminator_d_model=4,
        discriminator_nhead=1,
        discriminator_num_layers=1,
        learning_rate=1e-3,
        discriminator_max_elements=3,
    )
    monkeypatch.setattr(
        module,
        "optimizer_factory",
        lambda parameters, lr: _ToyOptimizer(parameters, lr),
    )
    optimizers = cast(list[torch.optim.Optimizer], module.configure_optimizers())

    def fake_optimizers(
        *, use_pl_optimizer: bool = False
    ) -> list[torch.optim.Optimizer]:
        del use_pl_optimizer
        return optimizers

    monkeypatch.setattr(
        module,
        "optimizers",
        fake_optimizers,
    )
    monkeypatch.setattr(module, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    output = module.training_step(
        collate_layoutganpp(synthetic_rows("magazine", 2, 8)), 0
    )
    assert output.ndim == 0
    assert "generator_loss" in module.latest_step_trace
