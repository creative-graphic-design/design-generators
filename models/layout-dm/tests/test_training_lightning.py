from typing import cast

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.training.datamodule import LayoutDMDataModule
from layout_dm.training.lightning_module import LayoutDMTrainingModule
from layout_dm.training.parity import (
    compare_layout_dm_optimizer_step,
    compare_layout_dm_step,
    trace_layout_dm_step,
)
from traingen.lightning.cli import main
from traingen_parity.trace import build_step_trace


pytestmark = pytest.mark.training


def tiny_config() -> LayoutDMConfig:
    return LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=4,
        num_bin_bboxes=8,
        bbox_quantization="linear",
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        num_timesteps=4,
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    dm = LayoutDMDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=2,
        num_workers=0,
    )
    dm.setup("fit")
    return cast(dict[str, torch.Tensor], next(iter(dm.train_dataloader())))


class TwoBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def test_training_step_records_required_trace_points() -> None:
    module = LayoutDMTrainingModule(
        config=tiny_config(), scheduler=None, time_sampler="uniform"
    )
    loss = module.training_step(tiny_batch(), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for key in ["t", "pt", "xt", "log_model_prob", "kl", "kl_loss", "train_loss"]:
        assert key in module.latest_step_trace
    assert module.lt_history.shape == (4,)
    assert module.lt_count.shape == (4,)


def test_optimizer_scheduler_and_parity_helpers() -> None:
    module = LayoutDMTrainingModule(config=tiny_config())
    optimizers = module.configure_optimizers()
    assert isinstance(optimizers, dict)
    optimizer_config = cast(dict[str, object], optimizers)
    lr_scheduler = cast(dict[str, object], optimizer_config["lr_scheduler"])
    assert lr_scheduler["monitor"] == "val_loss"
    torch.manual_seed(3)
    trace = trace_layout_dm_step(module, tiny_batch())
    assert compare_layout_dm_step(trace, trace).passed
    assert compare_layout_dm_optimizer_step(
        {"x": torch.ones(1)}, {"x": torch.ones(1)}
    ).passed
    mismatch_reference = build_step_trace("reference", {"x": torch.ones(1)})
    mismatch_target = build_step_trace("target", {"x": torch.zeros(1)})
    assert not compare_layout_dm_step(mismatch_reference, mismatch_target).passed


def test_validation_epoch_loss_uses_vendor_step_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = LayoutDMTrainingModule(
        config=tiny_config(), scheduler=None, time_sampler="uniform"
    )
    batch = tiny_batch()
    small_batch = {"input_ids": batch["input_ids"][:1]}
    seen: list[float] = []
    original_validation_step = module.validation_step

    def wrapped_validation_step(
        batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = original_validation_step(batch, batch_idx)
        seen.append(float(loss.detach()))
        return loss

    monkeypatch.setattr(module, "validation_step", wrapped_validation_step)
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    torch.manual_seed(123)
    result = trainer.validate(
        module,
        dataloaders=DataLoader(TwoBatchDataset([batch, small_batch]), batch_size=None),
        verbose=False,
    )
    expected = sum(seen) / len(seen)
    weighted = (seen[0] * 2 + seen[1]) / 3
    assert result[0]["val_loss"] == pytest.approx(expected)
    assert result[0]["val_loss"] != pytest.approx(weighted)


def test_datamodule_synthetic_loaders_and_cli_help() -> None:
    dm = LayoutDMDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=3,
        num_workers=0,
    )
    dm.setup()
    assert len(dm.train_dataloader()) == 2
    assert next(iter(dm.val_dataloader()))["input_ids"].shape == (2, 20)
    with pytest.raises(SystemExit):
        main(["--help"])
