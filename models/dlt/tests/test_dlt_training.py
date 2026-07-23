from functools import partial

import h5py
import numpy as np
import pytest
import torch

from dlt import DLTConfig
from dlt.training.config import DLTSeedMode

pytest.importorskip("lightning")

from dlt.training.datamodule import DLTDataModule
from dlt.training.dataset import SyntheticDLTDataset, collate_dlt_batch
from dlt.training.lightning_module import DLTTrainingModule
from dlt.training.parity import DLTSyntheticStepTraceAdapter
from dlt.training.seed import apply_seed_mode


def tiny_training_module() -> DLTTrainingModule:
    config = DLTConfig(
        dataset_name="publaynet",
        max_num_comp=4,
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
        num_cont_timesteps=4,
        num_discrete_steps=2,
    )
    return DLTTrainingModule(
        config=config,
        optimizer=partial(
            torch.optim.AdamW,
            lr=0.0001,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        ),
    )


@pytest.mark.training
def test_training_step_returns_loss_and_trace() -> None:
    module = tiny_training_module()
    batch = collate_dlt_batch(
        [SyntheticDLTDataset(length=1)[0], SyntheticDLTDataset(length=1, seed=1)[0]]
    )
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {
        "box",
        "box_cond",
        "cat",
        "noise",
        "t",
        "masked_l2",
        "masked_ce",
        "loss",
    } <= set(module.latest_step_trace)


@pytest.mark.training
def test_configure_optimizers_uses_injected_adamw_values() -> None:
    module = tiny_training_module()
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.0001
    assert optimizer.param_groups[0]["betas"] == (0.95, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 1e-6


@pytest.mark.training
def test_datamodule_seed_and_trace_adapter() -> None:
    apply_seed_mode(DLTSeedMode.default, 123)
    first = torch.rand(1)
    apply_seed_mode("default", 123)
    second = torch.rand(1)
    assert torch.equal(first, second)

    data = DLTDataModule(batch_size=2, length=4, max_num_comp=4, categories_num=7)
    train_batch = next(iter(data.train_dataloader()))
    val_batch = next(iter(data.val_dataloader()))
    assert train_batch["box"].shape == (2, 4, 4)
    assert val_batch["cat"].shape == (2, 4)

    trace = DLTSyntheticStepTraceAdapter().trace_training_step(
        tiny_training_module(), train_batch
    )
    assert set(DLTSyntheticStepTraceAdapter.trace_points) <= set(trace.tensors)


@pytest.mark.training
def test_datamodule_reads_h5_layouts(tmp_path) -> None:
    for name in ("publaynet_train.h5", "publaynet_val.h5"):
        with h5py.File(tmp_path / name, "w") as data:
            keep = data.create_group("0")
            keep.create_dataset(
                "bbox",
                data=np.asarray(
                    [[0.0, 0.0, 0.5, 0.25], [0.25, 0.25, 0.25, 0.25]],
                    dtype=np.float32,
                ),
            )
            keep.create_dataset("categories", data=np.asarray([1, 5], dtype=np.int64))
            keep.create_dataset("length", data=np.asarray(2, dtype=np.int64))
            drop = data.create_group("1")
            drop.create_dataset("bbox", data=np.zeros((1, 4), dtype=np.float32))
            drop.create_dataset("categories", data=np.asarray([1], dtype=np.int64))
            drop.create_dataset("length", data=np.asarray(1, dtype=np.int64))

    data = DLTDataModule(
        batch_size=1,
        data_path=str(tmp_path),
        max_num_comp=4,
        categories_num=7,
    )
    batch = next(iter(data.train_dataloader()))
    assert batch["box"].shape == (1, 4, 4)
    assert batch["cat"].shape == (1, 4)
    assert batch["mask"].sum() == 2
    assert torch.all(batch["box"][0, :2] >= -2.0)
    assert torch.all(batch["box"][0, :2] <= 2.0)
