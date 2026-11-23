"""CPU tests for shared PubLayNet pre-check evidence capture."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from publaynet_precheck_support import (  # noqa: E402
    Campaign,
    capture_provenance,
    instrument_original_objective,
)


@dataclass
class CampaignStub:
    CONFIG_SHA256: dict[str, str]
    REPOSITORY: Path
    config_path: Mock
    current_commit: Mock
    require_clean_tree: Mock
    sha256: Mock
    verify_splits: Mock


def _campaign(root: Path, config_hash: str = "config-hash") -> CampaignStub:
    return CampaignStub(
        CONFIG_SHA256={"label": "config-hash"},
        REPOSITORY=root,
        config_path=Mock(return_value=root / "publaynet_label.yaml"),
        current_commit=Mock(return_value="source-revision"),
        require_clean_tree=Mock(),
        sha256=Mock(return_value=config_hash),
        verify_splits=Mock(return_value={"train": {"sha256": "split-hash"}}),
    )


def test_provenance_reuses_launcher_checks(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    result = capture_provenance(cast(Campaign, campaign), "label", tmp_path)

    campaign.require_clean_tree.assert_called_once_with()
    campaign.verify_splits.assert_called_once_with(tmp_path)
    assert result == {
        "source_commit": "source-revision",
        "clean_tree": True,
        "config": "publaynet_label.yaml",
        "config_sha256": "config-hash",
        "input_splits": {"train": {"sha256": "split-hash"}},
    }


def test_provenance_rejects_a_changed_resolved_config(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, config_hash="changed-hash")

    with pytest.raises(RuntimeError, match="configuration hash"):
        capture_provenance(cast(Campaign, campaign), "label", tmp_path)

    campaign.verify_splits.assert_not_called()


def test_provenance_rejects_a_dirty_tracked_tree(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    campaign.require_clean_tree.side_effect = RuntimeError("tree is dirty")

    with pytest.raises(RuntimeError, match="tree is dirty"):
        capture_provenance(cast(Campaign, campaign), "label", tmp_path)

    campaign.config_path.assert_not_called()


def test_provenance_rejects_a_changed_processed_split(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    campaign.verify_splits.side_effect = RuntimeError("split hash check failed")

    with pytest.raises(RuntimeError, match="split hash check failed"):
        capture_provenance(cast(Campaign, campaign), "label", tmp_path)


def test_objective_hook_records_exact_native_logger_values(tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []

    def log(data: object, *, step: object) -> None:
        calls.append((data, step))

    wandb = cast(ModuleType, SimpleNamespace(log=log))
    trace = tmp_path / "updates.jsonl"
    instrument_original_objective(wandb, trace)
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


def test_objective_hook_fails_if_a_native_logger_index_is_missing(
    tmp_path: Path,
) -> None:
    wandb = cast(ModuleType, SimpleNamespace(log=lambda *_args, **_kwargs: None))
    instrument_original_objective(wandb, tmp_path / "updates.jsonl")

    with pytest.raises(RuntimeError, match="contiguous native logger indices"):
        wandb.log({"training_loss": torch.tensor(1.0)}, step=1)
