from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def load_runner() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "tests"
        / "vendor_parity"
        / "run_condition_parity.py"
    )
    spec = importlib.util.spec_from_file_location("ralf_condition_parity_runner", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_freeze_rejects_interpreter_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_path = Path(__file__).parents[1] / "tests" / "vendor_parity"
    monkeypatch.syspath_prepend(str(runner_path))
    runner = load_runner()
    expected_path = tmp_path / "runtime-freeze.txt"
    expected_path.write_bytes(b"package==expected\n")
    runtime_python = tmp_path / "venv" / "bin" / "python"

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == [
            "uv",
            "pip",
            "freeze",
            "--python",
            str(runtime_python),
        ]
        assert kwargs == {"check": True, "capture_output": True}
        return SimpleNamespace(stdout=b"package==wrong\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="refusing to launch"):
        runner.runtime_freeze(tmp_path / "output", runtime_python, expected_path)


def test_full_test_counts_are_dataset_specific() -> None:
    runner = load_runner()

    assert runner.TEST_COUNTS == {"cgl": 6002, "pku": 1000}


def test_localized_replay_restores_the_recorded_pre_batch_state() -> None:
    runner = load_runner()
    generator = torch.Generator(device="cpu").manual_seed(17)
    torch.rand(11, generator=generator)
    pre_batch_state = runner.capture_pre_batch_generator_state(generator)
    expected = torch.rand(8, generator=generator)
    expected_after = generator.get_state().clone()

    generator.manual_seed(999)
    replayed = runner.replay_batch_from_state(
        generator,
        pre_batch_state,
        lambda active_generator: torch.rand(8, generator=active_generator),
    )

    assert torch.equal(replayed, expected)
    assert torch.equal(generator.get_state(), expected_after)


def test_relation_comparison_records_unavoidable_difference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "INFERENCE_SEEDS", (0,))
    package_scores = tmp_path / "package.yaml"
    vendor_scores = tmp_path / "vendor.yaml"
    score = "average:\n  test:\n    metric: 1.0\n"
    package_scores.write_text(score, encoding="utf-8")
    vendor_scores.write_text(score, encoding="utf-8")

    result = runner.compare_predictions(
        {0: [{"label": [1]}]},
        {0: [{"label": [2]}]},
        package_scores,
        vendor_scores,
        allow_relation_difference=True,
    )

    assert result["status"] == "measured_relation_difference"
    assert result["prediction_mismatch_count"] == 1
