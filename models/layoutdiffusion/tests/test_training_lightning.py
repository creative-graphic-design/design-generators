import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")

from layoutdiffusion import LayoutDiffusionConfig
from layoutdiffusion.training.datamodule import LayoutDiffusionDataModule
from layoutdiffusion.training.lightning_module import LayoutDiffusionTrainingModule
from layoutdiffusion.training.parity import (
    compare_layoutdiffusion_optimizer_step,
    compare_layoutdiffusion_step,
    trace_layoutdiffusion_step,
)
from traingen.lightning.cli import main
from traingen_parity.trace import build_step_trace

pytestmark = pytest.mark.training


def tiny_config() -> LayoutDiffusionConfig:
    return LayoutDiffusionConfig(
        dataset_name="publaynet",
        seq_length=19,
        max_num_elements=3,
        diffusion_steps=10,
        num_channels=8,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=19,
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=2,
        num_workers=0,
    )
    dm.setup("fit")
    return cast(dict[str, torch.Tensor], next(iter(dm.train_dataloader())))


def _write_vendor_like_vocab(path: Path, labels: list[str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"START": 0, "END": 1, "UNK": 2, "PAD": 3, "|": 4}
    for label in labels:
        vocab[label] = len(vocab)
    for coord in range(128):
        vocab[str(coord)] = len(vocab)
    vocab_file = path / "vocab.json"
    vocab_file.write_text(json.dumps(vocab), encoding="utf-8")
    return vocab_file


class TwoBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def test_hf_source_rejects_injected_vendor_vocab(tmp_path: Path) -> None:
    vocab_file = _write_vendor_like_vocab(
        tmp_path / "publaynet", ["figure", "text", "title", "table", "list"]
    )

    with pytest.raises(
        ValueError,
        match="hf source with non-default id2label is not supported",
    ):
        LayoutDiffusionDataModule(
            dataset_name="publaynet",
            config=LayoutDiffusionConfig(dataset_name="publaynet", max_num_elements=1),
            dataset_source="hf",
            vocab_file=str(vocab_file),
            num_workers=0,
        )


def test_hf_source_rejects_premutated_vendor_id2label() -> None:
    config = LayoutDiffusionConfig(
        dataset_name="publaynet",
        max_num_elements=1,
        id2label={0: "figure", 1: "text", 2: "title", 3: "table", 4: "list"},
    )

    with pytest.raises(
        ValueError,
        match="hf source with non-default id2label is not supported",
    ):
        LayoutDiffusionDataModule(
            dataset_name="publaynet",
            config=config,
            dataset_source="hf",
            num_workers=0,
        )


def test_processed_source_rejects_cross_dataset_vocab_file(tmp_path: Path) -> None:
    vocab_file = _write_vendor_like_vocab(
        tmp_path / "rico25",
        [
            "Text",
            "Image",
            "Icon",
            "List_Item",
            "Text_Button",
            "Toolbar",
        ],
    )

    with pytest.raises(ValueError, match="vocab label count mismatch"):
        LayoutDiffusionDataModule(
            dataset_name="publaynet",
            config=LayoutDiffusionConfig(dataset_name="publaynet", max_num_elements=1),
            dataset_source="processed",
            processed_data_dir=str(tmp_path),
            vocab_file=str(vocab_file),
            num_workers=0,
        )


def test_training_step_records_required_trace_points() -> None:
    module = LayoutDiffusionTrainingModule(config=tiny_config(), scheduler=None)
    loss = module.training_step(tiny_batch(), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for key in [
        "t",
        "pt",
        "xt",
        "log_x_t",
        "log_x0_recon",
        "log_model_prob",
        "kl",
        "kl_loss",
        "lt_history",
        "lt_count",
        "aux_loss",
        "train_loss",
    ]:
        assert key in module.latest_step_trace
    lt_count = cast(torch.Tensor, module.lt_count)
    assert module.lt_history.shape == (10,)
    assert lt_count.shape == (10,)
    assert lt_count.sum() == 2


def test_optimizer_scheduler_ema_and_parity_helpers() -> None:
    module = LayoutDiffusionTrainingModule(config=tiny_config())
    assert module.auxiliary_loss_weight == pytest.approx(1e-3)
    optimizers = module.configure_optimizers()
    assert isinstance(optimizers, dict)
    optimizer_config = cast(dict[str, object], optimizers)
    lr_scheduler = cast(dict[str, object], optimizer_config["lr_scheduler"])
    assert lr_scheduler["interval"] == "step"
    assert module.ema_state_dict()
    torch.manual_seed(3)
    trace = trace_layoutdiffusion_step(module, tiny_batch())
    assert compare_layoutdiffusion_step(trace, trace).passed
    assert compare_layoutdiffusion_optimizer_step(
        {"x": torch.ones(1)}, {"x": torch.ones(1)}
    ).passed
    mismatch_reference = build_step_trace("reference", {"x": torch.ones(1)})
    mismatch_target = build_step_trace("target", {"x": torch.zeros(1)})
    assert not compare_layoutdiffusion_step(mismatch_reference, mismatch_target).passed


def test_trainer_fit_rejects_model_data_id2label_mismatch(tmp_path: Path) -> None:
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch import Trainer

    data_config = tiny_config()
    model_config = tiny_config()
    model_config.id2label = {0: "figure", 1: "text", 2: "title", 3: "table", 4: "list"}
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=data_config,
        batch_size=2,
        synthetic_size=2,
        num_workers=0,
    )
    module = LayoutDiffusionTrainingModule(
        config=model_config,
        time_sampler="uniform",
        seed_mode="deterministic",
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        plugins=[LightningEnvironment()],
    )

    with pytest.raises(ValueError, match="model/data id2label mismatch"):
        trainer.fit(module, datamodule=dm)


def test_trainer_fit_steps_scheduler_ema_and_checkpoint(tmp_path: Path) -> None:
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint

    seed_everything(23, workers=True)
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=8,
        num_workers=0,
    )
    module = LayoutDiffusionTrainingModule(
        config=tiny_config(),
        learning_rate=1e-4,
        scheduler="linear_anneal",
        lr_anneal_steps=4,
        time_sampler="uniform",
        seed_mode="deterministic",
    )
    initial_ema = module.ema_state_dict()
    expected_ema = {name: value.clone() for name, value in initial_ema.items()}
    ema_updates = 0
    original_update_ema = module.update_ema

    def traced_update_ema() -> None:
        nonlocal ema_updates
        ema_updates += 1
        with torch.no_grad():
            for name, parameter in module.model.named_parameters():
                if parameter.requires_grad:
                    expected_ema[name].mul_(module.ema_rate).add_(
                        parameter.detach().cpu(), alpha=1.0 - module.ema_rate
                    )
        original_update_ema()

    setattr(module, "update_ema", traced_update_ema)  # noqa: B010
    ckpt = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="step-{step}",
        every_n_train_steps=1,
        save_top_k=-1,
        save_last=True,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[ckpt],
        plugins=[LightningEnvironment()],
    )

    trainer.fit(module, datamodule=dm)

    assert trainer.global_step == 4
    scheduler = trainer.lr_scheduler_configs[0].scheduler
    assert scheduler.last_epoch == trainer.global_step
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.0)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)
    assert ema_updates == trainer.global_step
    actual_ema = module.ema_state_dict()
    assert any(
        not torch.equal(initial_ema[name], value) for name, value in actual_ema.items()
    )
    for name, expected_value in expected_ema.items():
        torch.testing.assert_close(actual_ema[name].cpu(), expected_value)
    assert ckpt.last_model_path
    checkpoint = torch.load(
        ckpt.last_model_path, map_location="cpu", weights_only=False
    )
    assert "layoutdiffusion_ema_state_dict" in checkpoint


def test_datamodule_synthetic_loaders_and_cli_help() -> None:
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=3,
        num_workers=0,
    )
    dm.setup()
    assert len(dm.train_dataloader()) == 2
    assert next(iter(dm.val_dataloader()))["input_ids"].shape == (2, 19)
    with pytest.raises(SystemExit):
        main(["--help"])


def test_train_dataloader_preconsume_skips_only_training_batches() -> None:
    rows = [
        {"input_ids": torch.full((19,), value), "attention_mask": torch.ones(19)}
        for value in range(6)
    ]
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=1,
        preconsume_train_batches=1,
        num_workers=0,
    )
    dm.train_dataset = cast(
        Dataset[dict[str, torch.Tensor | str]], TwoBatchDataset(rows)
    )
    dm.val_dataset = cast(Dataset[dict[str, torch.Tensor | str]], TwoBatchDataset(rows))
    dm.test_dataset = cast(
        Dataset[dict[str, torch.Tensor | str]], TwoBatchDataset(rows)
    )

    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    assert torch.equal(train_batch["input_ids"][:, 0], torch.tensor([2, 3]))
    assert torch.equal(val_batch["input_ids"][:, 0], torch.tensor([0, 1]))
    assert torch.equal(test_batch["input_ids"][:, 0], torch.tensor([0, 1]))


def test_preconsume_sampler_matches_skipped_iterator_order() -> None:
    rows = [
        {"input_ids": torch.full((19,), value), "attention_mask": torch.ones(19)}
        for value in range(8)
    ]
    dataset = TwoBatchDataset(rows)
    torch.manual_seed(17)
    expected_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    expected_iter = iter(expected_loader)
    next(expected_iter)
    expected = next(expected_iter)["input_ids"]

    torch.manual_seed(17)
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=tiny_config(),
        batch_size=2,
        synthetic_size=None,
        preconsume_train_batches=1,
        num_workers=0,
    )
    dm.train_dataset = cast(Dataset[dict[str, torch.Tensor | str]], dataset)
    actual = next(iter(dm.train_dataloader()))["input_ids"]

    assert torch.equal(actual, expected)


def test_processed_stream_rng_warmup_matches_original_embedding_init() -> None:
    rows = [
        {"input_ids": torch.full((19,), value), "attention_mask": torch.ones(19)}
        for value in range(8)
    ]
    dataset = TwoBatchDataset(rows)
    config = tiny_config()
    torch.manual_seed(17)
    random_embedding = torch.nn.Embedding(config.vocab_size - 1, 8)
    torch.nn.init.normal_(random_embedding.weight)
    del random_embedding
    expected_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    expected_iter = iter(expected_loader)
    next(expected_iter)
    expected = next(expected_iter)["input_ids"]

    torch.manual_seed(17)
    dm = LayoutDiffusionDataModule(
        dataset_name="publaynet",
        config=config,
        batch_size=2,
        synthetic_size=None,
        preconsume_train_batches=1,
        processed_stream_rng_warmup=True,
        num_workers=0,
    )
    dm.train_dataset = cast(Dataset[dict[str, torch.Tensor | str]], dataset)
    actual = next(iter(dm.train_dataloader()))["input_ids"]

    assert torch.equal(actual, expected)
