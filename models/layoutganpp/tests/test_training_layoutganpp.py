import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from layoutganpp.training.datamodule import LayoutGANPPDataModule
from layoutganpp.training.callbacks import LayoutGANPPVendorSchedule
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
from layoutganpp.training.step import gan_forward_trace, run_gan_iteration
from layoutganpp.training.validation import compute_maximum_iou


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
    assert len(_split_indices(10, "publaynet", "test")) == 1
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


def test_publaynet_validation_preserves_vendor_element_order(tmp_path):
    root = tmp_path / "publaynet"
    root.mkdir()
    train = {
        "categories": [{"id": 1, "name": "text"}, {"id": 2, "name": "title"}],
        "images": [
            {
                "id": index,
                "file_name": f"train-{index}.png",
                "width": 100,
                "height": 200,
            }
            for index in range(20)
        ],
        "annotations": [
            annotation
            for index in range(20)
            for annotation in (
                {
                    "image_id": index,
                    "category_id": 2,
                    "bbox": [70, 10, 20, 20],
                },
                {
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                },
            )
        ],
    }
    validation = {
        "categories": train["categories"],
        "images": [{"id": 100, "file_name": "test.png", "width": 100, "height": 200}],
        "annotations": [{"image_id": 100, "category_id": 1, "bbox": [10, 10, 20, 20]}],
    }
    (root / "train.json").write_text(json.dumps(train))
    (root / "val.json").write_text(json.dumps(validation))

    assert load_rows("publaynet", tmp_path, "val")[0].labels.tolist() == [1, 0]
    assert load_rows("publaynet", tmp_path, "train")[0].labels.tolist() == [0, 1]


def test_datamodule_builds_all_loaders():
    data_module = LayoutGANPPDataModule(
        dataset_name="magazine",
        batch_size=2,
        num_workers=0,
        synthetic_size=4,
        seed=13,
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


def test_full_run_checkpoint_configs_select_best_maximum_iou(tmp_path):
    configs = Path(__file__).parents[1] / "configs" / "training"
    checkpoint_args = None
    for dataset in ("magazine", "rico13", "publaynet"):
        config = yaml.safe_load((configs / f"layoutganpp_{dataset}.yaml").read_text())
        checkpoint = next(
            callback
            for callback in config["trainer"]["callbacks"]
            if callback["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
        )
        assert checkpoint["init_args"]["monitor"] == "val_maximum_iou"
        assert checkpoint["init_args"]["mode"] == "max"
        assert checkpoint["init_args"]["save_top_k"] == 1
        checkpoint_args = checkpoint["init_args"]

    assert checkpoint_args is not None
    checkpoint_args["dirpath"] = str(tmp_path)
    checkpoint_callback = ModelCheckpoint(**checkpoint_args)

    class ScoreModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.scores = (0.1, 0.8, 0.4)

        def training_step(self, batch, batch_idx):
            del batch, batch_idx
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            del batch, batch_idx
            self.log(
                "val_maximum_iou",
                self.scores[self.current_epoch],
                on_step=False,
                on_epoch=True,
            )

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    batch = [torch.ones(1)]
    trainer.fit(ScoreModule(), train_dataloaders=batch, val_dataloaders=batch)

    assert checkpoint_callback.best_model_score == pytest.approx(0.8)
    assert "epoch=000001" in Path(checkpoint_callback.best_model_path).name


def test_production_entrypoint_writes_best_checkpoint_for_bounded_full_run(tmp_path):
    repository_root = Path(__file__).resolve().parents[3]
    config_path = (
        Path(__file__).parents[1] / "configs" / "training" / "layoutganpp_magazine.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    checkpoint_dir = tmp_path / "checkpoints"
    config["trainer"].update(
        {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "max_steps": 1,
            "logger": False,
            "enable_progress_bar": False,
            "default_root_dir": str(tmp_path / "runs"),
        }
    )
    config["model"]["init_args"].update(
        {
            "generator_d_model": 4,
            "generator_nhead": 1,
            "generator_num_layers": 1,
            "discriminator_d_model": 4,
            "discriminator_nhead": 1,
            "discriminator_num_layers": 1,
        }
    )
    config["data"]["init_args"].update(
        {
            "data_root": None,
            "batch_size": 2,
            "num_workers": 0,
            "synthetic_size": 2,
            "shuffle_train": False,
        }
    )
    checkpoint_config = next(
        callback
        for callback in config["trainer"]["callbacks"]
        if callback["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
    )
    checkpoint_config["init_args"]["dirpath"] = str(checkpoint_dir)
    test_config_path = tmp_path / "layoutganpp_magazine.yaml"
    test_config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    traingen = shutil.which("traingen")
    assert traingen is not None
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [traingen, "fit", "--config", str(test_config_path)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoints) == 1
    checkpoint_path = checkpoints[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    optimizer_steps = [
        {int(state["step"].item()) for state in optimizer_state["state"].values()}
        for optimizer_state in checkpoint["optimizer_states"]
    ]
    assert optimizer_steps == [{1}, {1}]
    generator_updates = next(iter(optimizer_steps[0]))
    assert checkpoint["global_step"] == generator_updates

    checkpoint_callback = next(
        state
        for state in checkpoint["callbacks"].values()
        if state.get("best_model_path")
    )
    assert Path(checkpoint_callback["best_model_path"]) == checkpoint_path
    best_score = float(checkpoint_callback["best_model_score"])
    assert f"iou={best_score:.6f}" in checkpoint_path.name


def test_vendor_schedule_matches_iteration_and_validation_cadence():
    datamodule = LayoutGANPPDataModule(
        dataset_name="magazine",
        batch_size=2,
        num_workers=0,
        synthetic_size=8,
    )
    trainer = SimpleNamespace(
        datamodule=datamodule,
        fit_loop=SimpleNamespace(
            max_epochs=-1,
            epoch_loop=SimpleNamespace(max_steps=-1),
        ),
        current_epoch=0,
        check_val_every_n_epoch=1,
    )
    callback = LayoutGANPPVendorSchedule(iterations=200_000)

    callback.on_fit_start(cast(Trainer, trainer), cast(LightningModule, None))
    assert trainer.fit_loop.max_epochs == 50_000
    assert trainer.fit_loop.epoch_loop.max_steps == -1
    callback.on_validation_end(cast(Trainer, trainer), cast(LightningModule, None))
    assert trainer.check_val_every_n_epoch == 2_501

    trainer.current_epoch = 2_500
    callback.on_validation_end(cast(Trainer, trainer), cast(LightningModule, None))
    assert trainer.check_val_every_n_epoch == 5_001


def test_vendor_schedule_rejects_nonpositive_iterations_before_fit():
    callback = LayoutGANPPVendorSchedule(iterations=0)

    with pytest.raises(ValueError, match="iterations must be positive"):
        callback.on_fit_start(cast(Trainer, None), cast(LightningModule, None))


def test_vendor_schedule_preserves_bounded_diagnostic_step_limit():
    datamodule = LayoutGANPPDataModule(
        dataset_name="magazine",
        batch_size=2,
        num_workers=0,
        synthetic_size=8,
    )
    trainer = SimpleNamespace(
        datamodule=datamodule,
        fit_loop=SimpleNamespace(
            max_epochs=-1,
            epoch_loop=SimpleNamespace(max_steps=1_500),
        ),
        current_epoch=0,
        check_val_every_n_epoch=1,
    )
    callback = LayoutGANPPVendorSchedule(iterations=200_000)

    callback.on_fit_start(cast(Trainer, trainer), cast(LightningModule, None))
    assert trainer.fit_loop.max_epochs == -1
    assert trainer.fit_loop.epoch_loop.max_steps == 1_500
    callback.on_validation_end(cast(Trainer, trainer), cast(LightningModule, None))
    assert trainer.check_val_every_n_epoch == 2_501


def test_training_validation_maximum_iou_perfect_match():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]]).numpy()
    labels = torch.tensor([1]).numpy()

    assert compute_maximum_iou([(boxes, labels)], [(boxes, labels)]) == pytest.approx(
        1.0
    )
    shifted = torch.tensor([[0.6, 0.5, 0.4, 0.4]]).numpy()
    assert compute_maximum_iou([(boxes, labels)], [(shifted, labels)]) == pytest.approx(
        0.6
    )


def test_training_models_and_step_trace():
    rows = synthetic_rows("magazine", 2, 3)
    batch = collate_layoutganpp(rows)
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
    generator = module.generator
    discriminator = module.discriminator
    labels = cast(torch.Tensor, batch["labels"])
    trace = gan_forward_trace(generator, discriminator, batch)
    assert trace["latent_noise"].shape[:2] == labels.shape
    assert trace["condition_labels"].shape == labels.shape
    trace_with_grad = gan_forward_trace(generator, discriminator, batch, detach=False)
    assert trace_with_grad["generator_loss"].requires_grad
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    result = run_gan_iteration(
        generator, discriminator, batch, optimizer_g, optimizer_d
    )
    assert result["generator_gradient_norm"].numel() == 1
    assert result["update_order"].tolist() == [0, 1]
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
