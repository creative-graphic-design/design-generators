"""CPU-only checks for production loss capture and envelope comparison."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import run_publaynet_loss_envelope as ENVELOPE


def _write_trace(path: Path, system: str, seed: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(record="metadata", system=system, seed=seed)) + "\n"
        )
        for step in range(1, 1501):
            digest = 1501 - step if system == "package" else step
            stream.write(
                json.dumps(
                    dict(
                        record="update",
                        update=step,
                        objective=loss,
                        batch_sha256=f"{digest:064x}",
                    )
                )
                + "\n"
            )


def test_entrypoints_capture_losses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "vendor" / "scripts" / "label.py"
    source.parent.mkdir(parents=True)
    source.write_text("--load_vocab", encoding="utf-8")
    original, package = tmp_path / "original", tmp_path / "package"
    original.mkdir()
    package.mkdir()
    metrics = package / "lightning" / "logs" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("train_loss_step\n" + "1.0\n" * 1500, encoding="utf-8")
    campaign = Mock(
        VENDOR_SOURCE=source.parents[1],
        VENDOR_SCRIPTS={"label": "label.py"},
        build_argv=Mock(side_effect=lambda *args: [["python", "entry.py"]]),
    )
    run = Mock(
        side_effect=[
            Mock(returncode=0, stdout="[0/1][0/1500] Loss: 1.0\n" * 1500),
            Mock(returncode=0),
        ]
    )
    capture = Mock()
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)
    monkeypatch.setattr(ENVELOPE, "_record_native_output", capture)
    environment = {"PACKAGE_ENV": str(tmp_path)}
    ENVELOPE._run_original(
        campaign, seed=0, directory=original, environment=environment
    )
    ENVELOPE._run_package(campaign, seed=0, directory=package, environment=environment)
    assert [call.args[2] for call in campaign.build_argv.call_args_list] == [
        "original",
        "package",
    ]
    assert "--train_log_step 1" in Path(run.call_args_list[0].args[0][1]).read_text()
    assert "--trainer.max_steps=1500" in run.call_args_list[1].args[0]
    assert [len(call.kwargs["loss_values"]) for call in capture.call_args_list] == [
        1500,
        1500,
    ]


def test_three_seeds_detect_batch_order(tmp_path: Path) -> None:
    for seed in range(3):
        for system, loss in (("original", 10.0 + seed), ("package", 10.5 + seed)):
            _write_trace(
                tmp_path / system / f"seed-{seed}" / "updates.jsonl", system, seed, loss
            )
    result_path = tmp_path / "result.json"
    result = ENVELOPE._compare(tmp_path / "original", tmp_path / "package", result_path)
    assert result["batch_sequence_mismatches"] == 4500
    assert result["updates_per_run"] == 1500
