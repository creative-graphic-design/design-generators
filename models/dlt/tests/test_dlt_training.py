from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import h5py
import numpy as np
import pytest
import torch

from dlt import DLTConfig
import dlt.training as training_namespace
from dlt.training.config import DLTSeedMode

pytest.importorskip("lightning")

from traingen.lightning.cli import lightning_cli_class
from lightning.pytorch import LightningModule, Trainer

from dlt.training.callbacks import (
    DLTReferenceEpochSamplingCallback,
    _reference_condition_sample,
    _sample_from_model,
    consume_reference_epoch_sampling_rng,
)
from dlt.training.datamodule import DLTDataModule
from dlt.training.dataset import H5DLTDataset, SyntheticDLTDataset, collate_dlt_batch
from dlt.training.lightning_module import (
    DLTTrainingModule,
    DLTWarmupCosineSchedulerFactory,
)
from dlt.training.parity import DLTSyntheticStepTraceAdapter
from dlt.training.seed import apply_seed_mode


CONFIG_DIR = Path("models/dlt/configs/training")
CONFIG_NAMES = (
    "dlt_magazine.yaml",
    "dlt_publaynet.yaml",
    "dlt_publaynet_deterministic.yaml",
    "dlt_rico13.yaml",
    "dlt_rico13_deterministic.yaml",
    "smoke.yaml",
)


class _SchedulerConfig(TypedDict):
    scheduler: torch.optim.lr_scheduler.LambdaLR
    interval: str


class _OptimizerSchedulerConfig(TypedDict):
    optimizer: torch.optim.AdamW
    lr_scheduler: _SchedulerConfig


class _TinyReferenceDataset:
    max_num_comp = 4

    def __len__(self) -> int:
        return 2

    def get_data_by_ix(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, list[int], str]:
        del index
        box = np.asarray(
            [[-1.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.25, 0.25]], dtype=np.float32
        )
        cat = np.asarray([1, 5], dtype=np.int64)
        return box, cat, [0, 1], "tiny"


class _TinySamplingModel(torch.nn.Module):
    def forward(self, sample, noisy_batch, *, timesteps):
        del sample, timesteps
        box = noisy_batch["box"] * 0.0
        logits = torch.zeros(
            noisy_batch["cat"].shape[0], noisy_batch["cat"].shape[1], 7
        )
        return box, logits


class _TinySchedulerOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.prev_sample = sample
        self.pred_original_sample = sample


class _TinyScheduler:
    def __init__(self, *, num_cont_steps: int = 1) -> None:
        self.num_cont_steps = num_cont_steps

    def step_jointly(self, cont_output, cat_output, timestep, sample):
        del cat_output, timestep
        cat = torch.zeros(sample.shape[0], sample.shape[1], dtype=torch.long)
        return _TinySchedulerOutput(cont_output), {"cat": cat}


class _TinySamplingModule(SimpleNamespace):
    @property
    def device(self) -> str:
        return "cpu"


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
def test_configure_optimizers_steps_scheduler_per_batch() -> None:
    module = tiny_training_module()
    module.lr_scheduler = partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda step: float(step + 1),
    )
    optimizer_config = cast(_OptimizerSchedulerConfig, module.configure_optimizers())
    assert isinstance(optimizer_config, dict)
    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
    scheduler_config = optimizer_config["lr_scheduler"]
    assert scheduler_config["interval"] == "step"
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.LambdaLR)


@pytest.mark.training
def test_warmup_cosine_scheduler_factory_matches_step_values() -> None:
    module = tiny_training_module()
    optimizer = module.optimizer(module.parameters())
    scheduler = DLTWarmupCosineSchedulerFactory(
        num_warmup_steps=100,
        num_training_steps=1000,
    )(optimizer)
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


@pytest.mark.training
def test_warmup_cosine_scheduler_factory_accepts_estimated_steps() -> None:
    module = tiny_training_module()
    optimizer = module.optimizer(module.parameters())
    scheduler = DLTWarmupCosineSchedulerFactory(num_warmup_steps=2)(
        optimizer,
        estimated_stepping_batches=10,
    )
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-5)


@pytest.mark.training
def test_warmup_cosine_scheduler_factory_requires_total_steps() -> None:
    module = tiny_training_module()
    optimizer = module.optimizer(module.parameters())
    scheduler_factory = DLTWarmupCosineSchedulerFactory(num_warmup_steps=2)
    with pytest.raises(ValueError, match="num_training_steps is required"):
        scheduler_factory(optimizer)


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

    h5_dataset = H5DLTDataset(tmp_path / "publaynet_train.h5", max_num_comp=4)
    box, cat, order, key = h5_dataset.get_data_by_ix(0)
    assert key == "0"
    assert sorted(order) == [0, 1]
    assert box.shape == (2, 4)
    assert cat.shape == (2,)


@pytest.mark.training
def test_reference_epoch_sampling_callback_consumes_rng(tmp_path) -> None:
    for name in ("publaynet_train.h5", "publaynet_val.h5"):
        with h5py.File(tmp_path / name, "w") as data:
            for index in range(2):
                group = data.create_group(str(index))
                group.create_dataset(
                    "bbox",
                    data=np.asarray(
                        [[0.0, 0.0, 0.5, 0.25], [0.25, 0.25, 0.25, 0.25]],
                        dtype=np.float32,
                    ),
                )
                group.create_dataset(
                    "categories", data=np.asarray([1, 5], dtype=np.int64)
                )
                group.create_dataset("length", data=np.asarray(2, dtype=np.int64))

    module = DLTTrainingModule(
        config=DLTConfig(
            dataset_name="publaynet",
            max_num_comp=4,
            categories_num=7,
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            cond_emb_size=12,
            cat_emb_size=8,
            num_cont_timesteps=2,
            num_discrete_steps=2,
        ),
        optimizer=partial(torch.optim.AdamW, lr=0.0001),
    )
    module.train()
    data = DLTDataModule(
        batch_size=1,
        data_path=str(tmp_path),
        max_num_comp=4,
        categories_num=7,
    )
    data.setup("fit")
    assert isinstance(data.val_dataset, H5DLTDataset)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    consume_reference_epoch_sampling_rng(module, data.val_dataset, num_samples=1)
    after = torch.random.get_rng_state()

    assert not torch.equal(before, after)
    assert module.model.training


@pytest.mark.training
def test_reference_epoch_sampling_callback_requires_validation_dataset() -> None:
    callback = DLTReferenceEpochSamplingCallback(num_samples=1)
    with pytest.raises(RuntimeError, match="prepared val_dataset"):
        callback.on_train_epoch_end(
            cast(Trainer, SimpleNamespace(datamodule=SimpleNamespace())),
            cast(LightningModule, _TinySamplingModule()),
        )


@pytest.mark.training
@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_training_config_resolves_namespace_class_paths(config_name: str) -> None:
    cli = lightning_cli_class()(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        args=[
            "--config",
            str(CONFIG_DIR / config_name),
            "--trainer.accelerator",
            "cpu",
            "--trainer.devices",
            "1",
            "--trainer.logger",
            "false",
            "--trainer.enable_checkpointing",
            "false",
            "--trainer.enable_model_summary",
            "false",
        ],
    )

    assert isinstance(cli.model, DLTTrainingModule)
    assert isinstance(cli.datamodule, DLTDataModule)
    assert isinstance(cli.model.lr_scheduler, DLTWarmupCosineSchedulerFactory)
    if "publaynet" in config_name:
        assert any(
            isinstance(callback, DLTReferenceEpochSamplingCallback)
            for callback in getattr(cli.trainer, "callbacks", ())
        )


@pytest.mark.training
def test_training_namespace_exports_when_lightning_is_installed() -> None:
    assert (
        training_namespace.DLTDataModule  # ty: ignore[possibly-missing-attribute]
        is DLTDataModule
    )
    assert (
        training_namespace.DLTReferenceEpochSamplingCallback  # ty: ignore[possibly-missing-attribute]
        is DLTReferenceEpochSamplingCallback
    )
    assert (
        training_namespace.DLTTrainingModule  # ty: ignore[possibly-missing-attribute]
        is DLTTrainingModule
    )
    assert (
        training_namespace.DLTWarmupCosineSchedulerFactory  # ty: ignore[possibly-missing-attribute]
        is DLTWarmupCosineSchedulerFactory
    )
    assert "__all__" not in training_namespace.__dict__


@pytest.mark.training
def test_reference_epoch_sampling_rejects_too_many_samples() -> None:
    module = _TinySamplingModule(
        model=_TinySamplingModel(),
        scheduler=_TinyScheduler(),
        dlt_config=SimpleNamespace(categories_num=7),
    )
    with pytest.raises(ValueError, match="num_samples cannot exceed"):
        consume_reference_epoch_sampling_rng(
            cast(LightningModule, module), _TinyReferenceDataset(), num_samples=3
        )


@pytest.mark.training
def test_reference_epoch_sampling_preserves_eval_mode() -> None:
    model = _TinySamplingModel()
    model.eval()
    module = _TinySamplingModule(
        model=model,
        scheduler=_TinyScheduler(),
        dlt_config=SimpleNamespace(categories_num=7),
    )

    consume_reference_epoch_sampling_rng(
        cast(LightningModule, module), _TinyReferenceDataset(), num_samples=1
    )

    assert not model.training


@pytest.mark.training
def test_reference_condition_sample_covers_mask_variants() -> None:
    dataset = _TinyReferenceDataset()
    samples = [
        _reference_condition_sample(
            dataset, 0, sample_index, device=torch.device("cpu")
        )
        for sample_index in range(5)
    ]

    assert torch.equal(samples[0]["mask_box"][0, :2, :2], torch.ones(2, 2))
    assert torch.equal(samples[1]["mask_box"][0, :2, 2:], torch.ones(2, 2))
    assert torch.equal(samples[2]["mask_box"][0, :2], torch.ones(2, 4))
    assert samples[3]["mask_box"].shape == (1, 4, 4)
    assert torch.equal(samples[4]["mask_cat"][0, :2], torch.ones(2))
    assert torch.equal(samples[4]["mask_box"][0, :2], torch.ones(2, 4))


@pytest.mark.training
def test_sample_from_model_rejects_zero_step_scheduler() -> None:
    sample = _reference_condition_sample(
        _TinyReferenceDataset(), 0, 4, device=torch.device("cpu")
    )

    with pytest.raises(RuntimeError, match="did not run"):
        _sample_from_model(
            sample,
            _TinySamplingModel(),
            _TinyScheduler(num_cont_steps=0),
            categories_num=7,
            device=torch.device("cpu"),
        )
