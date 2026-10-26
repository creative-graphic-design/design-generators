from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any  # noqa: TID251 - call spies record dynamic pipeline keyword arguments.

import pytest
import torch

from run_evaluation_path_parity import (
    _PackageAdapter,
    _apply_evaluation_environment,
    _campaign_eval_main_path,
    _record_entrypoint,
    _record_path,
    _resolved_argv,
)


REQUIRED_KEYS = {
    "record_type",
    "family",
    "mode",
    "training_seed",
    "same_weight_route",
    "runtime_freeze_sha256",
    "weights",
    "inputs",
    "settings",
    "systems",
    "comparison",
    "artifact",
}


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("family", "condition", "max_length"),
    [
        ("label", "label", 120),
        ("label_size", "label_size", 120),
        ("relation", "relation", 150),
    ],
)
def test_package_adapter_uses_public_pipeline_and_its_constraints(
    family: str, condition: str, max_length: int
) -> None:
    generated = torch.tensor([[1]])
    pipeline_calls: list[dict[str, Any]] = []
    processor_calls: list[dict[str, Any]] = []

    class Processor:
        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            processor_calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[True]]),
            }

    class Pipeline:
        processor = Processor()

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            pipeline_calls.append(kwargs)
            return SimpleNamespace(sequences=generated)

    adapter = _PackageAdapter(
        Pipeline(),
        family=family,
        index2label={1: "label_1", 2: "label_2"},
        label_names={1: "button", 2: "text"},
        label_ids_to_sequence_id={1: 2, 2: 1},
        x_grid=128,
        y_grid=128,
    )
    batch: dict[str, Any] = {
        "gold_labels": [torch.tensor([1, 2])],
        "input_labels": [torch.tensor([1, 2])],
    }
    if family == "label_size":
        batch["input_bboxes"] = [torch.tensor([[0, 0, 12, 20], [0, 0, 30, 40]])]
    if family == "relation":
        batch["relations"] = [torch.tensor([[1, 1, 2, 1, 3]])]
    adapter.set_batch(batch)

    result = adapter(
        torch.tensor([[1]]),
        torch.tensor([[False]]),
        max_length=max_length,
        constrained_decoding=True,
        generation_constraint_fn=object(),
    )

    assert result["output"] is generated
    assert len(processor_calls) == len(pipeline_calls) == 1
    assert pipeline_calls[0]["condition_type"] == condition
    assert pipeline_calls[0]["labels"] == [["button", "text"]]
    assert pipeline_calls[0]["max_length"] == max_length
    assert "constrained_decoding" not in pipeline_calls[0]
    assert "generation_constraint_fn" not in pipeline_calls[0]
    assert "do_sample" not in pipeline_calls[0]
    assert "top_k" not in pipeline_calls[0]
    assert "temperature" not in pipeline_calls[0]
    if family == "label_size":
        assert pipeline_calls[0]["bbox"] == [
            [[0.0, 0.0, 12 / 127, 20 / 127], [0.0, 0.0, 30 / 127, 40 / 127]]
        ]
    if family == "relation":
        assert pipeline_calls[0]["relations"] == [[(2, 1, 1, 1, 3)]]


@pytest.mark.vendor_parity
def test_evaluation_path_parity_artifact_schema(tmp_path: Path) -> None:
    artifact = {
        "record_type": "evaluation_path_parity",
        "family": "label",
        "mode": "bitwise",
        "training_seed": 0,
        "same_weight_route": "converted original checkpoint loaded by the package model and original evaluator",
        "runtime_freeze_sha256": "0" * 64,
        "weights": {"path": ".cache/example.pth", "sha256": "0" * 64},
        "inputs": {
            "path": "<dataset root>/rico/pre_processed_20_25/test.pt",
            "sha256": "0" * 64,
            "sample_count": 3729,
        },
        "settings": {"resolved": {}, "sampling_seed": 500},
        "systems": {"package": {}, "original": {}},
        "comparison": {
            "predictions_bitwise_equal": True,
            "metrics_bitwise_equal": True,
        },
        "artifact": {"path": ".cache/example.json", "sha256": "0" * 64},
    }
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert REQUIRED_KEYS <= loaded.keys()
    assert loaded["mode"] == "bitwise"
    assert loaded["inputs"]["sample_count"] == 3729


@pytest.mark.vendor_parity
def test_campaign_entrypoint_is_preserved() -> None:
    assert (
        _record_entrypoint(
            {"command": "AUDIT_RUNTIME/bin/python vendor/src/eval_main.py --test"}
        )
        == "eval_main.py"
    )
    assert (
        _record_entrypoint(
            {"command": "AUDIT_RUNTIME/bin/python vendor/src/main.py --test"}
        )
        == "main.py"
    )


@pytest.mark.vendor_parity
def test_resolved_argv_handles_wrapped_campaign_entrypoint() -> None:
    record = {
        "command": (
            "AUDIT_RUNTIME/bin/python -c 'import runpy, sys; "
            "runpy.run_path(sys.argv[1])' "
            "<repo root>/.cache/layoutformerpp/s5/current/eval_main.py "
            "--test --data_dir <your data root> --out_dir old --tasks completion"
        )
    }

    argv = _resolved_argv(record, data_root=Path("DATA_ROOT"), output=Path("OUTPUT"))

    assert argv == [
        "--test",
        "--data_dir",
        "DATA_ROOT",
        "--out_dir",
        "OUTPUT",
        "--tasks",
        "completion",
    ]


@pytest.mark.vendor_parity
def test_campaign_eval_entrypoint_is_found_above_vendor_source(tmp_path: Path) -> None:
    vendor_src = tmp_path / "vendor" / "source"
    vendor_src.mkdir(parents=True)
    eval_main = tmp_path.joinpath(
        ".cache", "layoutformerpp", "s5", "current", "eval_main.py"
    )
    eval_main.parent.mkdir(parents=True)
    eval_main.touch()

    assert _campaign_eval_main_path(vendor_src) == eval_main


@pytest.mark.vendor_parity
def test_campaign_record_path_selects_training_seed_and_system(tmp_path: Path) -> None:
    assert _record_path(tmp_path, "refinement", 2, "package") == tmp_path.joinpath(
        ".cache",
        "layoutformerpp",
        "s5",
        "current",
        "eval",
        "rico25-s5-7ea6a89a0f24",
        "refinement",
        "seed2",
        "package",
        "record.json",
    )


@pytest.mark.vendor_parity
def test_campaign_evaluation_thread_settings_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    torch = ModuleType("torch")
    setattr(torch, "set_num_threads", lambda value: calls.append(("threads", value)))
    setattr(
        torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    _apply_evaluation_environment(
        {
            "OMP_NUM_THREADS": 8,
            "torch_num_threads": 8,
            "torch_num_interop_threads": 1,
        }
    )

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert calls == [("threads", 8), ("interop", 1)]
