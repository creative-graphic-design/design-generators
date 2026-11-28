"""Flex-DM Lightning adapter tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_datamodule_creates_splits_and_lazy_loaders(monkeypatch, tmp_path) -> None:
    pytest.importorskip("lightning")
    from flex_dm.training import datamodule as datamodule_module

    class TinyDataset:
        def __init__(self, data_dir, dataset_name, split, sample_order=None) -> None:
            self.split = split
            self.sample_order = sample_order

        def __len__(self) -> int:
            return len(self.sample_order) if self.sample_order is not None else 2

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            value = index if self.sample_order is None else self.sample_order[index]
            return {"value": torch.tensor(value)}

        def collate_fn(self, records) -> dict[str, torch.Tensor]:
            return {"value": torch.stack([record["value"] for record in records])}

    monkeypatch.setattr(datamodule_module, "FlexDmTfrecordDataset", TinyDataset)
    data = datamodule_module.FlexDmDataModule(
        data_dir=str(tmp_path),
        dataset_name="crello",
        batch_size=2,
        shuffle_train=False,
    )
    data.setup()
    assert set(data.datasets) == {"train", "val", "test"}
    assert next(iter(data.train_dataloader()))["value"].tolist() == [0, 1]
    assert next(iter(data.val_dataloader()))["value"].tolist() == [0, 1]
    assert next(iter(data.test_dataloader()))["value"].tolist() == [0, 1]

    fit_data = datamodule_module.FlexDmDataModule(
        data_dir=str(tmp_path), dataset_name="crello", shuffle_train=False
    )
    fit_data.setup("fit")
    assert set(fit_data.datasets) == {"train", "val"}

    test_data = datamodule_module.FlexDmDataModule(
        data_dir=str(tmp_path), dataset_name="crello", shuffle_train=False
    )
    test_data.setup("test")
    assert set(test_data.datasets) == {"test"}

    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        lazy_data = datamodule_module.FlexDmDataModule(
            data_dir=str(tmp_path), dataset_name="crello", shuffle_train=False
        )
        assert next(iter(getattr(lazy_data, loader_name)()))["value"].numel() == 2


def test_datamodule_replays_recorded_train_and_validation_orders(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("lightning")
    from flex_dm.training import datamodule as datamodule_module

    class TinyDataset:
        def __init__(self, data_dir, dataset_name, split, sample_order=None) -> None:
            del data_dir, dataset_name
            self.split = split
            self.order = sample_order

        def __len__(self) -> int:
            return len(self.order) if self.order is not None else 2

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            values = (index, index + 1) if self.order is None else self.order
            return {"value": torch.tensor(values[index])}

        def collate_fn(self, records) -> dict[str, torch.Tensor]:
            return {"value": torch.stack([record["value"] for record in records])}

    monkeypatch.setattr(datamodule_module, "FlexDmTfrecordDataset", TinyDataset)
    train_order = tmp_path / "train.txt"
    val_order = tmp_path / "val.txt"
    train_order.write_text("1\n0\n1\n", encoding="utf-8")
    val_order.write_text("1\n0\n", encoding="utf-8")
    data = datamodule_module.FlexDmDataModule(
        data_dir=str(tmp_path),
        dataset_name="crello",
        batch_size=3,
        train_order_path=str(train_order),
        val_order_path=str(val_order),
    )
    data.setup("fit")

    assert next(iter(data.train_dataloader()))["value"].tolist() == [1, 0, 1]
    assert next(iter(data.val_dataloader()))["value"].tolist() == [1, 0]


def test_lightning_module_prepares_batches_logs_losses_and_configures_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("traingen")
    from flex_dm.training import lightning_module as training_module
    from traingen.optim import KerasAdam

    columns = {
        "length": {
            "type": "categorical",
            "input_dim": 4,
            "shape": (1,),
            "is_sequence": False,
            "primary_label": None,
        },
        "type": {
            "type": "categorical",
            "input_dim": 3,
            "shape": (1,),
            "is_sequence": True,
            "primary_label": 0,
        },
    }

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.config = SimpleNamespace(
                task_names=("elem",),
                masking_method="elem",
                input_columns=columns,
                attribute_groups={"type": ("type",)},
                input_dtype="set",
            )

        def forward(self, *, inputs, labels, masks, task_ids):
            del inputs, labels, masks, task_ids
            loss = self.weight.square()
            return SimpleNamespace(
                loss=loss,
                loss_components={"type": loss},
                l2_loss=loss * 0,
            )

    def load_model(cls, model_name_or_path: str) -> TinyModel:
        assert model_name_or_path == "tiny"
        return TinyModel()

    monkeypatch.setattr(
        training_module.FlexDmForMaskedDocumentModeling,
        "from_pretrained",
        classmethod(load_model),
    )
    module = training_module.FlexDmTrainingModule("tiny")
    logged: list[str] = []
    monkeypatch.setattr(
        module, "log", lambda name, value, **kwargs: logged.append(name)
    )
    batch = {
        "length": torch.tensor([[1], [1]]),
        "type": torch.tensor([[[1], [2]], [[2], [1]]]),
    }

    loss = module.training_step(batch, 0)
    assert loss.item() == 4.0
    assert module.latest_step_trace["task_ids"].tolist() == [0, 0]
    assert module.latest_step_trace["train_loss"].requires_grad is False
    assert logged == ["type", "l2", "train_loss"]

    logged.clear()
    assert module.validation_step(batch, 0).item() == 4.0
    assert logged == ["val_loss"]

    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, KerasAdam)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-7
    module.model.weight.grad = torch.tensor(3.0)
    module.configure_gradient_clipping(optimizer, gradient_clip_val=0.0)
    torch.testing.assert_close(module.model.weight.grad, torch.tensor(1.0))
    with pytest.raises(ValueError, match="per-parameter clip_norm"):
        module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)

    monkeypatch.setattr(
        module.model,
        "forward",
        lambda **kwargs: SimpleNamespace(loss=None, loss_components=None, l2_loss=None),
    )
    with pytest.raises(RuntimeError, match="requires task losses"):
        module.training_step(batch, 0)
    with pytest.raises(RuntimeError, match="requires a loss"):
        module.validation_step(batch, 0)
