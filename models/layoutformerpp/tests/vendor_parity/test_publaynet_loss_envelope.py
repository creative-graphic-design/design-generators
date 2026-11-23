"""CPU-only checks for native production loss collection and comparison."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "publaynet_loss_envelope", ROOT / "run_publaynet_loss_envelope.py"
)
assert SPEC is not None and SPEC.loader is not None
ENVELOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENVELOPE)


def _write_trace(path: Path, system: str, seed: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"record": "metadata", "system": system, "seed": seed}]
    rows.extend(
        {
            "record": "update",
            "update": update,
            "objective": loss,
            "batch_sha256": f"{1501 - update if system == 'package' else update:064x}",
        }
        for update in range(1, 1501)
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_production_loss_capture_and_native_losses(tmp_path: Path) -> None:
    assert ENVELOPE._original_losses(
        ["[0/200][0/4865]\tLoss: 3.125", "not a loss"]
    ) == [3.125]
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "step,train_loss_step,train_loss_epoch\n0,3.25,\n1,3.5,\n2,,3.375\n",
        encoding="utf-8",
    )
    assert ENVELOPE._package_losses(metrics) == [3.25, 3.5]
    package_body = (
        (ROOT / "run_publaynet_loss_envelope.py")
        .read_text()
        .split("def _run_package", 1)[1]
        .split("def _run(", 1)[0]
    )
    assert "campaign.build_argv(" in package_body and ")[0]" in package_body


def test_production_envelope_checks_three_seeds_and_batch_order(tmp_path: Path) -> None:
    for seed in range(3):
        for system, loss in (("original", 10.0 + seed), ("package", 10.5 + seed)):
            path = tmp_path / system / f"seed-{seed}" / "updates.jsonl"
            _write_trace(path, system, seed, loss)
    result_path = tmp_path / "result.json"
    result = ENVELOPE._compare(tmp_path / "original", tmp_path / "package", result_path)
    assert result["passed"] is False
    assert result["batch_sequence_mismatches"] == 4500
    assert result["updates_per_run"] == 1500
