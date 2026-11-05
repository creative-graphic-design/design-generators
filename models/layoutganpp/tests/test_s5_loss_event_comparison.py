from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_s5_loss_events.py"
SPEC = importlib.util.spec_from_file_location("compare_s5_loss_events", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def test_loss_comparison_reports_event_index_and_step_offsets() -> None:
    result = COMPARISON.compare_series(
        [(0, 0.5), (50, 1.0), (100, 1.5), (150, 2.0)],
        [(49, 1.0), (99, 1.5), (149, 2.0)],
    )

    assert result["same_step_pairs"]["pairs"] == 0
    assert result["event_index_offsets"]["0"] == {
        "pairs": 3,
        "bitwise_equal": 0,
    }
    assert result["event_index_offsets"]["1"] == {
        "pairs": 3,
        "bitwise_equal": 3,
    }
    assert result["step_offsets"]["1"] == {
        "pairs": 3,
        "bitwise_equal": 3,
    }


def test_float_comparison_uses_float32_bits() -> None:
    assert COMPARISON._equal_float32(1.0, math.nextafter(1.0, 2.0))
    assert not COMPARISON._equal_float32(1.0, 1.0000002)


def test_duplicate_event_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="steps must be unique"):
        COMPARISON.compare_series([(0, 1.0), (0, 2.0)], [(0, 1.0)])


def test_main_prints_output_path_outside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        COMPARISON,
        "build_report",
        lambda: {"conclusion": "test", "runs": [], "aggregate": {}},
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--output", str(output)])

    COMPARISON.main()

    assert capsys.readouterr().out.splitlines()[0] == str(output)
    assert output.is_file()
