"""CPU tests for shared PubLayNet pre-check evidence capture."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LAUNCHER_DIRECTORY = ROOT.parents[1] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(LAUNCHER_DIRECTORY))

import launch_track  # noqa: E402

from publaynet_precheck_support import (  # noqa: E402
    Campaign,
    capture_provenance,
    instrument_original_objective,
)


def _prepare_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    config = (
        repository
        / "models"
        / "layoutformerpp"
        / "configs"
        / "training"
        / "publaynet_label.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text("family: label\n", encoding="utf-8")
    tracked_file = repository / "tracked.txt"
    tracked_file.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "add", "models", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Pre-check test",
            "-c",
            "user.email=precheck@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(launch_track, "REPOSITORY", repository)
    monkeypatch.setattr(
        launch_track, "CONFIG_SHA256", {"label": launch_track.sha256(config)}
    )

    split_root = tmp_path / "data" / "publaynet" / "pre_processed_20_5"
    split_root.mkdir(parents=True)
    expected_splits: dict[str, tuple[int, str]] = {}
    for split in ("train", "val", "test"):
        split_path = split_root / f"{split}.pt"
        torch.save([f"{split}-row-0", f"{split}-row-1"], split_path)
        expected_splits[split] = (2, launch_track.sha256(split_path))
    monkeypatch.setattr(launch_track, "EXPECTED_SPLITS", expected_splits)
    return tmp_path / "data"


def test_provenance_uses_real_launcher_git_and_split_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = _prepare_campaign(monkeypatch, tmp_path)

    result = capture_provenance(cast(Campaign, launch_track), "label", data_root)

    input_splits = cast(dict[str, dict[str, int | str]], result["input_splits"])
    assert result["source_commit"] == launch_track.current_commit()
    assert result["clean_tree"] is True
    assert result["config_sha256"] == launch_track.CONFIG_SHA256["label"]
    assert set(input_splits) == {"train", "val", "test"}


def test_provenance_rejects_changed_resolved_config_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = _prepare_campaign(monkeypatch, tmp_path)
    monkeypatch.setattr(launch_track, "CONFIG_SHA256", {"label": "0" * 64})

    with pytest.raises(RuntimeError, match="configuration hash mismatch"):
        capture_provenance(cast(Campaign, launch_track), "label", data_root)


def test_provenance_rejects_dirty_tracked_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = _prepare_campaign(monkeypatch, tmp_path)
    (launch_track.REPOSITORY / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracked source tree must be clean"):
        capture_provenance(cast(Campaign, launch_track), "label", data_root)


def test_provenance_rejects_processed_split_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = _prepare_campaign(monkeypatch, tmp_path)
    expected_splits = dict(launch_track.EXPECTED_SPLITS)
    row_count, _ = expected_splits["train"]
    expected_splits["train"] = (row_count, "0" * 64)
    monkeypatch.setattr(launch_track, "EXPECTED_SPLITS", expected_splits)

    with pytest.raises(RuntimeError, match="processed train split hash check failed"):
        capture_provenance(cast(Campaign, launch_track), "label", data_root)


def test_objective_hook_records_exact_native_logger_values(tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []

    def log(data: object, *, step: object) -> None:
        calls.append((data, step))

    wandb = cast(ModuleType, SimpleNamespace(log=log))
    trace = tmp_path / "updates.jsonl"
    instrument_original_objective(wandb, trace, expected_updates=2)
    first = torch.tensor(1.234567890123, dtype=torch.float64)
    second = torch.tensor(2.345678901234, dtype=torch.float64)

    wandb.log({"training_loss": first}, step=0)
    wandb.log({"training_loss": second}, step=1)

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [record["logger_step"] for record in records] == [0, 1]
    assert [record["update"] for record in records] == [1, 2]
    assert [record["objective"] for record in records] == [
        first.item(),
        second.item(),
    ]
    assert [step for _, step in calls] == [0, 1]


@pytest.mark.vendor_parity
def test_objective_hook_survives_disabled_wandb_init_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    import wandb

    monkeypatch.setattr(wandb, "init", wandb.init)
    monkeypatch.setattr(wandb, "log", wandb.log)
    monkeypatch.setattr(wandb, "finish", wandb.finish)
    trace = tmp_path / "updates.jsonl"
    instrument_original_objective(wandb, trace, expected_updates=2)

    wandb.init()
    wandb.log({"training_loss": torch.tensor(1.25)}, step=0)
    wandb.init()
    wandb.log({"training_loss": torch.tensor(2.5)}, step=1)
    wandb.finish()

    records = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["logger_step"] for record in records] == [0, 1]
    assert [record["objective"] for record in records] == [1.25, 2.5]


def test_objective_hook_fails_if_a_native_logger_index_is_missing(
    tmp_path: Path,
) -> None:
    wandb = cast(ModuleType, SimpleNamespace(log=lambda *_args, **_kwargs: None))
    instrument_original_objective(wandb, tmp_path / "updates.jsonl", expected_updates=2)

    with pytest.raises(RuntimeError, match="contiguous native logger indices"):
        wandb.log({"training_loss": torch.tensor(1.0)}, step=1)


def test_objective_hook_rejects_missing_objective_at_expected_update(
    tmp_path: Path,
) -> None:
    wandb = cast(ModuleType, SimpleNamespace(log=lambda *_args, **_kwargs: None))
    instrument_original_objective(wandb, tmp_path / "updates.jsonl", expected_updates=2)

    with pytest.raises(
        RuntimeError, match="expected training objective at logger step 0"
    ):
        wandb.log({"other_metric": 1.0}, step=0)


def test_objective_hook_rejects_missing_final_update(tmp_path: Path) -> None:
    wandb = cast(
        ModuleType,
        SimpleNamespace(
            log=lambda *_args, **_kwargs: None,
            finish=lambda: None,
        ),
    )
    instrument_original_objective(wandb, tmp_path / "updates.jsonl", expected_updates=2)
    wandb.log({"training_loss": torch.tensor(1.0)}, step=0)

    with pytest.raises(RuntimeError, match="expected 2 objectives, recorded 1"):
        wandb.finish()
