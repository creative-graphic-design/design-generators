"""Tests for the production-path loss-envelope comparison."""

import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_COMPARATOR = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_crello_loss_envelope.py"
    ),
    run_name="compare_crello_loss_envelope_test",
)
compare_loss_envelope = cast(
    Callable[..., dict[str, object]], _COMPARATOR["compare_loss_envelope"]
)
compare_main = cast(Callable[[], None], _COMPARATOR["main"])


def _write_trace(
    path: Path,
    *,
    system: str,
    seed: int,
    value: float,
    batch_hash: str,
    updates: int = 1501,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"record": "metadata", "system": system, "seed": seed},
        *(
            {
                "record": "update",
                "update": update,
                "objective": value,
                "batch_sha256": batch_hash,
            }
            for update in range(1, updates + 1)
        ),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_loss_envelope_uses_aligned_events_and_requires_same_batches(tmp_path):
    original_dir = tmp_path / "original"
    package_dir = tmp_path / "package"
    expected_hash = "a" * 64
    for seed in range(3):
        _write_trace(
            original_dir / f"seed-{seed}" / "updates.jsonl",
            system="original",
            seed=seed,
            value=10.0 + seed,
            batch_hash=expected_hash,
        )
        _write_trace(
            package_dir / f"seed-{seed}" / "updates.jsonl",
            system="package",
            seed=seed,
            value=10.5 + seed,
            batch_hash=expected_hash,
        )

    result = compare_loss_envelope(
        original_dir,
        package_dir,
        seeds=(0, 1, 2),
        updates=1501,
        logger_interval=19,
    )
    assert result["passed"] is True
    assert result["batch_sequence_mismatches"] == 0
    assert result["package_summary"] == 11.5
    assert result["original_range"] == [10.0, 12.0]

    _write_trace(
        package_dir / "seed-2" / "updates.jsonl",
        system="package",
        seed=2,
        value=10.5 + 2,
        batch_hash="b" * 64,
    )
    result = compare_loss_envelope(
        original_dir,
        package_dir,
        seeds=(0, 1, 2),
        updates=1501,
        logger_interval=19,
    )
    assert result["passed"] is False
    assert result["batch_sequence_mismatches"] == 1501


def test_loss_envelope_cli_rejects_short_update_count_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--original-dir",
            str(tmp_path / "original"),
            "--package-dir",
            str(tmp_path / "package"),
            "--output",
            str(tmp_path / "result.json"),
            "--seeds",
            "0",
            "1",
            "2",
            "--updates",
            "1499",
            "--logger-interval",
            "19",
        ],
    )

    with pytest.raises(ValueError, match="at least 1500"):
        compare_main()


def test_loss_envelope_cli_accepts_short_trace_with_explicit_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_dir = tmp_path / "original"
    package_dir = tmp_path / "package"
    for seed in range(3):
        for directory, system, loss in (
            (original_dir, "original", 10.0 + seed),
            (package_dir, "package", 10.5 + seed),
        ):
            _write_trace(
                directory / f"seed-{seed}" / "updates.jsonl",
                system=system,
                seed=seed,
                value=loss,
                batch_hash="a" * 64,
                updates=2,
            )

    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--original-dir",
            str(original_dir),
            "--package-dir",
            str(package_dir),
            "--output",
            str(result_path),
            "--seeds",
            "0",
            "1",
            "2",
            "--updates",
            "2",
            "--logger-interval",
            "1",
            "--minimum-updates",
            "2",
        ],
    )

    compare_main()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["passed"] is True
    assert result["updates_per_run"] == 2
