from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything

from traingen.checkpointing import SafeValue, load_weights_only_checkpoint

from layoutformerpp.training import checkpoint as checkpoint_module
from layoutformerpp.training.checkpoint import (
    LayoutFormerPPModelCheckpoint,
    _validate_identity,
)
from layoutformerpp.training.lightning_module import (
    LayoutFormerPPDataModule,
    LayoutFormerPPTrainingModule,
)


Mutation = Literal[
    "on_load_checkpoint",
    "datamodule_load_state_dict",
    "restore_rng_state",
]


def _identity(checkpoint_dir: Path) -> dict[str, str | None]:
    return {
        "config_sha256": "config",
        "source_commit": "source",
        "vendor_source_commit": None,
        "device": "cpu",
        "checkpoint_dirpath": str(checkpoint_dir.resolve()),
        "campaign_root": "campaign",
    }


def _write_data_root(root: Path) -> None:
    split_root = root / "publaynet" / "pre_processed_20_5"
    split_root.mkdir(parents=True, exist_ok=True)
    if not (split_root / "train.pt").is_file():
        rows = [
            {
                "labels": torch.tensor([1, 2, 3]),
                "bboxes": torch.tensor(
                    [
                        [0.1, 0.1, 0.5, 0.5],
                        [0.2, 0.2, 0.3, 0.3],
                        [0.4, 0.4, 0.2, 0.2],
                    ]
                ),
                "name": f"row-{index}",
            }
            for index in range(256)
        ]
        torch.save(rows, split_root / "train.pt")
        torch.save(rows[:2], split_root / "val.pt")


def _disable_callback_load(
    callback: LayoutFormerPPModelCheckpoint,
    trainer: Trainer,
    pl_module: LightningModule,
    checkpoint: dict[str, SafeValue],
) -> None:
    del callback, trainer, pl_module, checkpoint


def _disable_datamodule_load(
    data: LayoutFormerPPDataModule,
    state_dict: dict[str, SafeValue],
) -> None:
    del data, state_dict


def _disable_rng_restore(state: dict[str, SafeValue]) -> None:
    del state


def _run_package_track(
    root: Path,
    max_epochs: int,
    monkeypatch: pytest.MonkeyPatch,
    resume_path: Path | None = None,
    mutation: Mutation | None = None,
) -> tuple[dict[str, torch.Tensor], list[Path]]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    data_root = root / "data"
    _write_data_root(data_root)
    seed_everything(11, workers=True)
    data = LayoutFormerPPDataModule(
        recipe_name="publaynet_relation", data_root=str(data_root), seed=17
    )
    module = LayoutFormerPPTrainingModule(
        recipe_name="publaynet_relation",
        config={
            "dataset": "publaynet",
            "task": "gen_r",
            "vocab_size": 178,
            "max_position_embeddings": 400,
            "d_model": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "dim_feedforward": 16,
            "dropout": 0.3,
        },
    )
    checkpoint_dir = root / "checkpoints"
    callback = LayoutFormerPPModelCheckpoint(
        dirpath=str(checkpoint_dir), monitor="val_loss", mode="min", save_top_k=1
    )
    manifest = root / "launch_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(_identity(checkpoint_dir)), encoding="utf-8")
    durable = root / "durable"
    durable.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        json.dumps(_identity(checkpoint_dir), sort_keys=True),
    )
    monkeypatch.setenv("LAYOUTFORMERPP_LAUNCH_MANIFEST", str(manifest))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE", "track")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )

    with pytest.MonkeyPatch.context() as mutation_patch:
        if mutation == "on_load_checkpoint":
            mutation_patch.setattr(
                LayoutFormerPPModelCheckpoint,
                "on_load_checkpoint",
                _disable_callback_load,
            )
        elif mutation == "datamodule_load_state_dict":
            mutation_patch.setattr(
                LayoutFormerPPDataModule,
                "load_state_dict",
                _disable_datamodule_load,
            )
        elif mutation == "restore_rng_state":
            mutation_patch.setattr(
                checkpoint_module, "restore_rng_state", _disable_rng_restore
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
        "LayoutFormerPPDataModule",
    }
    assert "layoutformerpp_datamodule" not in payload
    assert list((tmp_path / "split" / "durable" / "track").glob("epoch-*-best"))
    assert uninterrupted_checkpoints[-1].is_file()


@pytest.mark.parametrize(
    "mutation",
    ["on_load_checkpoint", "datamodule_load_state_dict", "restore_rng_state"],
)
def test_resume_equivalence_detects_disabled_restore_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Mutation,
) -> None:
    uninterrupted, _ = _run_package_track(tmp_path / "uninterrupted", 2, monkeypatch)
    _, split_checkpoints = _run_package_track(tmp_path / "split", 1, monkeypatch)
    resumed, _ = _run_package_track(
        tmp_path / "split",
        2,
        monkeypatch,
        split_checkpoints[-1],
        mutation,
    )

    assert any(not torch.equal(resumed[name], uninterrupted[name]) for name in resumed)


def test_checkpoint_identity_mismatch_and_missing_identity_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    wrong_dir = tmp_path / "wrong-checkpoints"
    monkeypatch.setenv(
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        json.dumps(_identity(wrong_dir), sort_keys=True),
    )
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        _validate_identity(checkpoint_dir)

    monkeypatch.delenv("LAYOUTFORMERPP_CHECKPOINT_IDENTITY")
    with pytest.raises(RuntimeError, match="IDENTITY is required"):
        _validate_identity(checkpoint_dir)


def test_durable_publish_requires_complete_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = LayoutFormerPPModelCheckpoint(dirpath=str(tmp_path / "checkpoints"))
    for name in (
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="publishing is required"):
        callback._publish(tmp_path / "checkpoint.ckpt", epoch=0)

    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="partially configured"):
        callback._publish(tmp_path / "checkpoint.ckpt", epoch=0)
