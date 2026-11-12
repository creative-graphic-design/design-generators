from __future__ import annotations

import json
import os
import random
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything

from traingen.checkpointing import (
    is_safe_mapping,
    load_weights_only_checkpoint,
    safe_value_type,
)

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


def _identity(
    checkpoint_dir: Path, campaign_root: Path | None = None
) -> dict[str, str | None]:
    root = (campaign_root or checkpoint_dir.parent).resolve()
    return {
        "config_sha256": "config",
        "source_commit": "source",
        "vendor_source_commit": None,
        "device": "cpu",
        "checkpoint_dirpath": str(checkpoint_dir.resolve()),
        "campaign_root": str(root),
    }


def _configure_durable_track(
    monkeypatch: pytest.MonkeyPatch,
    campaign_root: Path,
    checkpoint_dir: Path,
    track: str,
) -> None:
    campaign_root.mkdir(parents=True, exist_ok=True)
    durable_root = campaign_root / "durable"
    durable_root.mkdir(exist_ok=True)
    manifest = campaign_root / "launch_manifest.json"
    identity = _identity(checkpoint_dir, campaign_root)
    manifest_identity = {
        key: value for key, value in identity.items() if key != "campaign_root"
    }
    manifest.write_text(json.dumps(manifest_identity), encoding="utf-8")
    monkeypatch.setenv("CAMPAIGN_ROOT", str(campaign_root.resolve()))
    monkeypatch.setenv(
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY", json.dumps(identity, sort_keys=True)
    )
    monkeypatch.setenv("LAYOUTFORMERPP_LAUNCH_MANIFEST", str(manifest))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(durable_root))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE", track)


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
    checkpoint: dict[str, safe_value_type],
) -> None:
    del callback, trainer, pl_module, checkpoint


def _disable_datamodule_load(
    data: LayoutFormerPPDataModule,
    state_dict: dict[str, safe_value_type],
) -> None:
    del data, state_dict


def _disable_rng_restore(state: dict[str, safe_value_type]) -> None:
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
    _configure_durable_track(monkeypatch, root, checkpoint_dir, "track")
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


def test_plain_checkpoint_captures_and_restores_rng_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "CAMPAIGN_ROOT",
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    callback = LayoutFormerPPModelCheckpoint(dirpath=str(tmp_path / "checkpoints"))
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    checkpoint: dict[str, safe_value_type] = {}
    callback.on_save_checkpoint(trainer, LightningModule(), checkpoint)

    assert "layoutformerpp_identity" not in checkpoint
    assert is_safe_mapping(checkpoint.get("layoutformerpp_rng"))
    expected = (random.random(), np.random.rand(), torch.rand(2))
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)

    callback.on_load_checkpoint(trainer, LightningModule(), checkpoint)

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_plain_resume_refuses_checkpoint_with_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "CAMPAIGN_ROOT",
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    callback = LayoutFormerPPModelCheckpoint(dirpath=str(tmp_path / "checkpoints"))
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)

    with pytest.raises(
        RuntimeError, match="cannot be validated without durable settings"
    ):
        callback.on_load_checkpoint(
            trainer,
            LightningModule(),
            {"layoutformerpp_identity": {}},
        )


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
        "CAMPAIGN_ROOT",
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    callback._publish(tmp_path / "checkpoint.ckpt", epoch=0)

    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="configured together"):
        callback._publish(tmp_path / "checkpoint.ckpt", epoch=0)


def test_relative_checkpoint_dirpath_uses_each_seed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_dirpath = (
        Path(".cache")
        / "layoutformerpp"
        / "training-runs"
        / "publaynet_relation"
        / "checkpoints"
    )
    resolved_paths: list[Path] = []
    for seed in (1, 2):
        campaign_root = tmp_path / f"seed-{seed}"
        checkpoint_dir = campaign_root / relative_dirpath
        _configure_durable_track(
            monkeypatch, campaign_root, checkpoint_dir, f"seed-{seed}"
        )
        callback = LayoutFormerPPModelCheckpoint(dirpath=str(relative_dirpath))
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            default_root_dir=str(campaign_root),
            logger=False,
            enable_checkpointing=False,
        )

        callback.setup(trainer, LightningModule(), stage="fit")
        assert callback.dirpath is not None
        resolved_paths.append(Path(callback.dirpath))

    assert resolved_paths == [
        (tmp_path / f"seed-{seed}" / relative_dirpath).resolve() for seed in (1, 2)
    ]
    assert resolved_paths[0] != resolved_paths[1]


def test_partial_durable_setup_fails_before_first_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CountingModule(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.steps = 0

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            del batch_idx
            self.steps += 1
            return self.weight * batch

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    for name in (
        "CAMPAIGN_ROOT",
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(tmp_path))
    callback = LayoutFormerPPModelCheckpoint(dirpath=str(tmp_path / "checkpoints"))
    module = CountingModule()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=str(tmp_path),
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.raises(RuntimeError, match="configured together"):
        trainer.fit(module, train_dataloaders=[torch.ones(())])

    assert module.steps == 0


def test_on_load_checkpoint_rejects_empty_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    callback = LayoutFormerPPModelCheckpoint(dirpath=str(checkpoint_dir))
    identity = _identity(checkpoint_dir, tmp_path)
    identity["config_sha256"] = ""
    manifest = tmp_path / "launch_manifest.json"
    manifest_identity = {
        key: value for key, value in identity.items() if key != "campaign_root"
    }
    manifest.write_text(json.dumps(manifest_identity), encoding="utf-8")
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    monkeypatch.setenv("CAMPAIGN_ROOT", str(tmp_path.resolve()))
    monkeypatch.setenv("LAYOUTFORMERPP_CHECKPOINT_IDENTITY", json.dumps(identity))
    monkeypatch.setenv("LAYOUTFORMERPP_LAUNCH_MANIFEST", str(manifest))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_ROOT", str(durable_root))
    monkeypatch.setenv("LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE", "track")

    with pytest.raises(ValueError, match="empty value"):
        identity_state: dict[str, safe_value_type] = {
            key: value for key, value in _identity(checkpoint_dir, tmp_path).items()
        }
        trainer = Trainer(accelerator="cpu", devices=1, logger=False)
        callback.on_load_checkpoint(
            trainer,
            LightningModule(),
            {"layoutformerpp_identity": identity_state},
        )


def test_on_load_checkpoint_rejects_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    callback = LayoutFormerPPModelCheckpoint(dirpath=str(checkpoint_dir))
    _configure_durable_track(monkeypatch, tmp_path, checkpoint_dir, "track")
    identity_state: dict[str, safe_value_type] = {
        key: value
        for key, value in _identity(tmp_path / "other-checkpoints", tmp_path).items()
    }
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)

    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        callback.on_load_checkpoint(
            trainer,
            LightningModule(),
            {"layoutformerpp_identity": identity_state},
        )


def test_plain_traingen_fit_config_runs_without_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_root = tmp_path / "data"
    output_root = tmp_path / "seed-1"
    _write_data_root(data_root)
    for name in (
        "CAMPAIGN_ROOT",
        "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    command = [
        str(Path(sys.executable).with_name("traingen")),
        "fit",
        "--config",
        str(
            project_root
            / "models"
            / "layoutformerpp"
            / "configs"
            / "training"
            / "publaynet_relation.yaml"
        ),
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
        "--trainer.max_epochs=1",
        "--trainer.max_steps=1",
        "--trainer.limit_train_batches=1",
        "--trainer.limit_val_batches=1",
        "--trainer.num_sanity_val_steps=0",
        f"--trainer.default_root_dir={output_root}",
        "--trainer.enable_progress_bar=false",
        "--trainer.enable_model_summary=false",
        "--model.init_args.config.d_model=8",
        "--model.init_args.config.encoder_layers=1",
        "--model.init_args.config.decoder_layers=1",
        "--model.init_args.config.encoder_attention_heads=2",
        "--model.init_args.config.decoder_attention_heads=2",
        "--model.init_args.config.dim_feedforward=16",
        "--model.init_args.config.dropout=0.0",
    ]
    environment = {
        **os.environ,
        "CAMPAIGN_ROOT": str(output_root),
        "CUDA_VISIBLE_DEVICES": "",
        "LAYOUTFORMERPP_DATA_ROOT": str(data_root),
    }
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"plain traingen fit failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    checkpoints = list(output_root.rglob("*.ckpt"))
    assert checkpoints
    plain_checkpoint = load_weights_only_checkpoint(checkpoints[0])
    assert "layoutformerpp_identity" not in plain_checkpoint
    assert is_safe_mapping(plain_checkpoint.get("layoutformerpp_rng"))
