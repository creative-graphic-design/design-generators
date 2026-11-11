from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from traingen.checkpointing import load_weights_only_checkpoint

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from layoutformerpp.training.checkpoint import LayoutFormerPPModelCheckpoint  # noqa: E402
from layoutformerpp.training.lightning_module import (  # noqa: E402
    LayoutFormerPPDataModule,
    LayoutFormerPPTrainingModule,
)


def _identity() -> dict[str, object]:
    return {
        "config_sha256": "config",
        "source_commit": "source",
        "vendor_source_commit": None,
        "device": "cpu",
        "checkpoint_dirpath": "checkpoints",
        "campaign_root": "campaign",
    }


def _write_data_root(root: Path) -> None:
    split_root = root / "publaynet" / "pre_processed_20_5"
    split_root.mkdir(parents=True, exist_ok=True)
    if not (split_root / "train.pt").is_file():
        rows = [
            {
                "labels": torch.tensor([1]),
                "bboxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
                "name": f"row-{index}",
            }
            for index in range(64)
        ]
        torch.save(rows, split_root / "train.pt")
        torch.save(rows[:1], split_root / "val.pt")


def _run_package_track(
    root: Path,
    max_epochs: int,
    monkeypatch: pytest.MonkeyPatch,
    resume_path: Path | None = None,
) -> tuple[dict[str, torch.Tensor], list[Path]]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    from lightning.pytorch import Trainer, seed_everything

    data_root = root / "data"
    _write_data_root(data_root)
    seed_everything(11, workers=True)
    data = LayoutFormerPPDataModule(
        recipe_name="publaynet_label", data_root=str(data_root), seed=17
    )
    module = LayoutFormerPPTrainingModule(
        recipe_name="publaynet_label",
        config={
            "dataset": "publaynet",
            "task": "gen_t",
            "vocab_size": 139,
            "max_position_embeddings": 400,
            "d_model": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
    )
    checkpoint_dir = root / "checkpoints"
    callback = LayoutFormerPPModelCheckpoint(
        dirpath=str(checkpoint_dir), monitor="val_loss", mode="min", save_top_k=1
    )
    monkeypatch.setenv(
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY", json.dumps(_identity(), sort_keys=True)
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(
        module, datamodule=data, ckpt_path=str(resume_path) if resume_path else None
    )
    state = {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }
    return state, sorted(checkpoint_dir.glob("epoch-*.ckpt"))


def test_package_resume_matches_uninterrupted_epoch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uninterrupted, uninterrupted_checkpoints = _run_package_track(
        tmp_path / "uninterrupted", 2, monkeypatch
    )
    _, split_checkpoints = _run_package_track(tmp_path / "split", 1, monkeypatch)
    resumed, resumed_checkpoints = _run_package_track(
        tmp_path / "split", 2, monkeypatch, split_checkpoints[-1]
    )

    assert all(torch.equal(resumed[name], uninterrupted[name]) for name in resumed)
    assert len(resumed_checkpoints) == 2
    payload = load_weights_only_checkpoint(resumed_checkpoints[-1])
    assert set(payload) >= {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "epoch",
        "global_step",
        "loops",
        "layoutformerpp_identity",
        "layoutformerpp_rng",
        "layoutformerpp_datamodule",
    }
    assert uninterrupted_checkpoints[-1].is_file()
