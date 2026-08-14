"""Bounded production LightningCLI wiring checks for the RADM CGL recipe."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch

from radm.training.config import effective_radm_config

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(f"checkpoint payload must be a mapping: {path}")
    return payload


def test_s3_radm_production_traingen_fit_wiring(
    tmp_path: Path,
) -> None:
    """Exercise the CGL fit boundary and its observable training artifacts."""
    project_root = Path(__file__).resolve().parents[4]
    config = project_root / "models/radm/configs/training/radm_cgl.yaml"
    output_root = tmp_path / "fit"
    logger_root = output_root / "logs"
    checkpoint_root = output_root / "checkpoints"
    callback_config = json.dumps(
        [
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": str(checkpoint_root),
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": False,
                },
            }
        ],
        separators=(",", ":"),
    )
    traingen = Path(sys.executable).with_name("traingen")
    if not traingen.is_file():
        raise AssertionError(f"compatible environment is missing {traingen}")

    command = [
        str(traingen),
        "fit",
        "--config",
        str(config),
        "--trainer.accelerator=gpu",
        "--trainer.devices=1",
        "--trainer.max_epochs=1",
        "--trainer.max_steps=2",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=1",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.check_val_every_n_epoch=1",
        "--trainer.log_every_n_steps=1",
        f"--trainer.default_root_dir={output_root}",
        f"--trainer.logger.init_args.save_dir={logger_root}",
        "--trainer.logger.init_args.name=csv",
        f"--trainer.callbacks={callback_config}",
        "--data.init_args.allow_missing_text_features=true",
        "--trainer.enable_progress_bar=false",
        "--trainer.enable_model_summary=false",
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "PARITY_REQUIRE": "1",
        "RADM_REFERENCE_DEVICE": "cuda:0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
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
        f"RADM CGL traingen fit exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    checkpoints = sorted(checkpoint_root.rglob("*.ckpt"))
    assert checkpoints, f"no ModelCheckpoint file under {checkpoint_root}"
    metrics = sorted(logger_root.rglob("metrics.csv"))
    assert metrics, f"no CSV logger output under {logger_root}"
    metric_text = metrics[-1].read_text(encoding="utf-8")
    assert "train_loss" in metric_text
    assert "val_loss" in metric_text

    payload = _checkpoint_payload(checkpoints[-1])
    required_fields = {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "global_step",
        "callbacks",
        "loops",
    }
    assert required_fields <= payload.keys()
    assert payload["global_step"] == 2
    optimizer_states = payload["optimizer_states"]
    assert isinstance(optimizer_states, list)
    assert len(optimizer_states) == 1
    loops = payload["loops"]
    assert isinstance(loops, dict)
    loops = cast(dict[str, object], loops)
    fit_loop = loops["fit_loop"]
    assert isinstance(fit_loop, dict)
    fit_loop = cast(dict[str, object], fit_loop)
    epoch_loop = fit_loop["epoch_loop.state_dict"]
    assert isinstance(epoch_loop, dict)
    epoch_loop = cast(dict[str, object], epoch_loop)
    optimizer_steps = epoch_loop["_batches_that_stepped"]
    assert optimizer_steps == 2
    scheduler_states = payload["lr_schedulers"]
    assert isinstance(scheduler_states, list) and len(scheduler_states) == 1
    scheduler_state = scheduler_states[0]
    assert isinstance(scheduler_state, dict)
    scheduler_state = cast(dict[str, object], scheduler_state)
    assert scheduler_state["last_epoch"] == 2

    effective = effective_radm_config()
    alpha = 2 / effective.warmup_iters
    expected_factor = effective.warmup_factor * (1 - alpha) + alpha
    expected_lr = effective.learning_rate * expected_factor
    last_lr = scheduler_state["_last_lr"]
    assert isinstance(last_lr, list) and last_lr
    last_lr = cast(list[float], last_lr)
    assert last_lr[0] == pytest.approx(expected_lr)
    callback_states = payload["callbacks"]
    assert isinstance(callback_states, dict)
    callback_states = cast(dict[str, object], callback_states)
    assert any(
        isinstance(state, dict) and state.get("best_model_path")
        for state in callback_states.values()
    )

    evidence = {
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "PARITY_REQUIRE": environment["PARITY_REQUIRE"],
            "RADM_REFERENCE_DEVICE": environment["RADM_REFERENCE_DEVICE"],
            "CUBLAS_WORKSPACE_CONFIG": environment["CUBLAS_WORKSPACE_CONFIG"],
        },
        "config_sha256": _sha256(config),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "logical_device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "physical_gpu": os.environ.get("RADM_PHYSICAL_GPU"),
        },
        "exit_code": completed.returncode,
        "global_step": payload["global_step"],
        "optimizer_steps": optimizer_steps,
        "scheduler_last_epoch": scheduler_state["last_epoch"],
        "scheduler_last_lr": last_lr,
        "expected_lr": expected_lr,
        "checkpoint_selected": True,
        "checkpoint_sha256": _sha256(checkpoints[-1]),
        "metrics_sha256": _sha256(metrics[-1]),
        "checkpoint_fields": sorted(payload),
        "metric_fields": metric_text.splitlines()[0].split(","),
    }
    evidence_path = Path(
        os.environ.get(
            "RADM_S3_WIRING_EVIDENCE_PATH",
            str(tmp_path / "s3-wiring-evidence.json"),
        )
    )
    if evidence_path.exists():
        raise AssertionError(f"refusing to overwrite wiring evidence: {evidence_path}")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "S3 wiring evidence "
        f"path={evidence_path} sha256={_sha256(evidence_path)} "
        f"exit_code={completed.returncode} global_step={payload['global_step']} "
        f"scheduler_last_epoch={scheduler_state['last_epoch']}"
    )
