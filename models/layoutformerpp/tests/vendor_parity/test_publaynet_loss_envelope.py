"""CPU-only checks for production loss capture and envelope comparison."""

import json
import sys
from pathlib import Path
from typing import cast
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


def test_entrypoints_capture_exact_objectives(
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
    metrics.write_text(
        "step,train_loss_step\n0,1.0000001\n1,1.0000002\n", encoding="utf-8"
    )
    campaign = Mock(
        VENDOR_SOURCE=source.parents[1],
        VENDOR_SCRIPTS={"label": "label.py"},
        build_argv=Mock(side_effect=lambda *args: [["python", "entry.py"]]),
    )
    original_values = (1.0000001, 1.0000002)

    def run_entrypoint(command, **kwargs):
        environment = kwargs["env"]
        with Path(environment["LAYOUTFORMERPP_BATCH_TRACE"]).open(
            "w", encoding="utf-8"
        ) as stream:
            for index in range(2):
                stream.write(
                    json.dumps(
                        {
                            "update": index + 1,
                            "sample_ids": [f"layout-{index}"],
                            "batch_sha256": f"{index:064x}",
                        }
                    )
                    + "\n"
                )
        trace = environment.get("LAYOUTFORMERPP_OBJECTIVE_TRACE")
        if trace:
            Path(trace).write_text(
                "".join(
                    json.dumps(
                        {
                            "record": "update",
                            "update": index + 1,
                            "logger_step": index,
                            "objective": loss,
                        }
                    )
                    + "\n"
                    for index, loss in enumerate(original_values)
                ),
                encoding="utf-8",
            )
        return Mock(
            returncode=0,
            stdout="[0/1][0/2] Loss: 1.000\n[0/1][1/2] Loss: 1.000\n",
        )

    run = Mock(side_effect=run_entrypoint)
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)
    monkeypatch.setattr(ENVELOPE, "UPDATES", 2)
    environment = {"PACKAGE_ENV": str(tmp_path)}
    ENVELOPE._run_original(
        campaign, seed=0, directory=original, environment=environment, updates=2
    )
    ENVELOPE._run_package(
        campaign, seed=0, directory=package, environment=environment, updates=2
    )
    assert [call.args[2] for call in campaign.build_argv.call_args_list] == [
        "original",
        "package",
    ]
    assert "--train_log_step 1" in Path(run.call_args_list[0].args[0][1]).read_text()
    assert "--trainer.max_steps=2" in run.call_args_list[1].args[0]
    assert "LAYOUTFORMERPP_OBJECTIVE_TRACE" in run.call_args_list[0].kwargs["env"]
    assert (
        run.call_args_list[0].kwargs["env"]["LAYOUTFORMERPP_EXPECTED_OBJECTIVE_UPDATES"]
        == "2"
    )
    assert "install_original_objective_capture()" in ENVELOPE._sitecustomize_source()
    original_rows = [
        json.loads(line)
        for line in (original / "updates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ][1:]
    package_rows = [
        json.loads(line)
        for line in (package / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    ][1:]
    assert [row["objective"] for row in original_rows] == list(original_values)
    assert [row["logger_step"] for row in original_rows] == [0, 1]
    assert [row["objective"] for row in package_rows] == list(original_values)
    assert [row["logger_step"] for row in package_rows] == [0, 1]


@pytest.mark.parametrize("system", ["original", "package"])
def test_child_failure_persists_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, system: str
) -> None:
    directory = tmp_path / system
    directory.mkdir()
    if system == "original":
        source = tmp_path / "vendor" / "scripts" / "label.py"
        source.parent.mkdir(parents=True)
        source.write_text("--load_vocab", encoding="utf-8")
        campaign = Mock(
            VENDOR_SOURCE=source.parents[1],
            VENDOR_SCRIPTS={"label": "label.py"},
            build_argv=Mock(return_value=[["python", "entry.py"]]),
        )
        run = ENVELOPE._run_original
    else:
        campaign = Mock(build_argv=Mock(return_value=[["python", "entry.py"]]))
        run = ENVELOPE._run_package

    monkeypatch.setattr(
        ENVELOPE.subprocess,
        "run",
        Mock(
            return_value=Mock(
                returncode=1,
                stdout="complete child stdout\n",
                stderr="complete child stderr\n",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="stdout.txt.*stderr.txt") as error:
        run(
            campaign,
            seed=0,
            directory=directory,
            environment={"PACKAGE_ENV": str(tmp_path)},
        )

    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    assert stdout_path.read_text(encoding="utf-8") == "complete child stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "complete child stderr\n"
    assert str(stdout_path) in str(error.value)
    assert str(stderr_path) in str(error.value)


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


def test_short_cpu_comparison_is_written_as_non_gate_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_path = tmp_path / "result.json"

    def compare(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            json.dumps({"passed": True, "updates_per_run": 2}), encoding="utf-8"
        )
        return Mock(returncode=0)

    run = Mock(side_effect=compare)
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)

    result = ENVELOPE._compare(
        tmp_path / "original",
        tmp_path / "package",
        result_path,
        updates=2,
        cpu_smoke=True,
    )

    assert result == json.loads(result_path.read_text(encoding="utf-8"))
    assert result["updates_per_run"] == 2
    assert result["run_kind"] == "cpu_smoke"
    assert result["gate_eligible"] is False
    assert result["comparison_verdict"] == "not_meaningful_for_cpu_smoke"
    assert "passed" not in result
    command = run.call_args.args[0]
    minimum_index = command.index("--minimum-updates")
    assert command[minimum_index + 1] == "2"


def test_production_comparator_keeps_production_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = tmp_path / "original"
    package = tmp_path / "package"
    result_path = tmp_path / "result.json"

    def compare(command, **kwargs):
        result_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
        return Mock(returncode=0)

    run = Mock(side_effect=compare)
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)

    ENVELOPE._compare(original, package, result_path)

    command = run.call_args.args[0]
    minimum_index = command.index("--minimum-updates")
    assert command[minimum_index + 1] == "1500"


def test_comparator_failure_persists_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_path = tmp_path / "result.json"
    run = Mock(
        return_value=Mock(
            returncode=1,
            stdout="complete comparator stdout\n",
            stderr="complete comparator stderr\n",
        )
    )
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="stdout.txt.*stderr.txt") as error:
        ENVELOPE._compare(
            tmp_path / "original",
            tmp_path / "package",
            result_path,
            updates=2,
            cpu_smoke=True,
        )

    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    assert stdout_path.read_text(encoding="utf-8") == "complete comparator stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "complete comparator stderr\n"
    assert str(stdout_path) in str(error.value)
    assert str(stderr_path) in str(error.value)


def test_objective_traces_reject_missing_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ENVELOPE, "UPDATES", 1)
    original = tmp_path / "original.jsonl"
    original.write_text("", encoding="utf-8")
    package = tmp_path / "metrics.csv"
    package.write_text("train_loss_step\n1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="objective trace expected 1 updates"):
        ENVELOPE._original_objectives(original, updates=1)

    with pytest.raises(RuntimeError, match="missing its native logger index"):
        ENVELOPE._package_objectives(package, updates=1)


def test_native_logger_alignment_rejects_invalid_indices(tmp_path: Path) -> None:
    original = tmp_path / "updates.jsonl"
    package = tmp_path / "package" / "updates.jsonl"
    package.parent.mkdir(parents=True)
    trace = '{{"record":"metadata"}}\n{{"logger_step":{}}}\n{{"logger_step":{}}}\n'
    for original_steps, package_steps, message in (
        ((0, 1), (0, 2), "different native logger indices"),
        ((0, 0), (0, 1), "duplicate native logger indices"),
        ((0, 1), (0, 0), "duplicate native logger indices"),
    ):
        for path, steps in ((original, original_steps), (package, package_steps)):
            path.write_text(trace.format(*steps), encoding="utf-8")

        with pytest.raises(RuntimeError, match=message):
            ENVELOPE._require_native_logger_alignment(original.parent, package.parent)


def test_sample_id_digest_normalizes_collator_keys() -> None:
    original = {"name": ["first", "second"], "input_ids": "original-only"}
    package = {"names": ["first", "second"], "task_ids": "package-only"}
    reordered = {"names": ["second", "first"], "labels": "package-only"}

    digest = ENVELOPE.sample_ids_sha256(original)
    assert digest == ENVELOPE.sample_ids_sha256(package)
    assert digest != ENVELOPE.sample_ids_sha256(reordered)


def test_run_records_provenance_and_uses_checked_gpu_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    config = repository / "publaynet_label.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("resolved config", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    campaign = Mock(
        REPOSITORY=repository,
        CONFIG_SHA256={"label": "config-hash"},
        config_path=Mock(return_value=config),
        sha256=Mock(return_value="config-hash"),
        current_commit=Mock(return_value="source-commit"),
        require_clean_tree=Mock(),
        verify_splits=Mock(return_value={"train": {"sha256": "split-hash"}}),
        read_data_root=Mock(return_value=data_root),
        gpu_info=Mock(return_value=("V100", "checked-uuid")),
        runtime_freeze_sha256=Mock(return_value="runtime-freeze-hash"),
        build_environment=Mock(return_value={"PACKAGE_ENV": str(tmp_path)}),
        vendor_commit=Mock(return_value="vendor-commit"),
    )
    monkeypatch.setitem(sys.modules, "launch_track", campaign)
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(tmp_path / "root.env"))
    monkeypatch.setenv("PACKAGE_ENV", str(tmp_path))
    monkeypatch.setattr(ENVELOPE, "_run_original", Mock())
    monkeypatch.setattr(ENVELOPE, "_run_package", Mock())
    monkeypatch.setattr(ENVELOPE, "_require_native_logger_alignment", Mock())
    monkeypatch.setattr(ENVELOPE, "_compare", Mock(return_value={"passed": True}))
    output = tmp_path / "output"
    output.mkdir()

    result = ENVELOPE._run(output, gpu_index=2)

    provenance = cast(dict[str, object], result["provenance"])
    input_splits = cast(dict[str, dict[str, int | str]], provenance["input_splits"])
    assert provenance["source_commit"] == "source-commit"
    assert input_splits["train"]["sha256"] == "split-hash"
    assert result["runtime_freeze_sha256"] == "runtime-freeze-hash"
    campaign.runtime_freeze_sha256.assert_called_once_with(tmp_path)
    campaign.gpu_info.assert_called_once_with(2)
    assert all(
        call.kwargs["gpu_uuid"] == "checked-uuid"
        for call in campaign.build_environment.call_args_list
    )


def test_gpu_index_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = Mock(return_value={})
    output_root = tmp_path / "cache"
    monkeypatch.setattr(ENVELOPE, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--output-dir", str(output_root / "run")]
    )
    monkeypatch.setattr(ENVELOPE, "_run", runner)

    with pytest.raises(SystemExit) as error:
        ENVELOPE.main()

    assert error.value.code == 2
    runner.assert_not_called()


@pytest.mark.parametrize(
    "relative_path_variable",
    [
        "PACKAGE_ENV",
        "LAYOUTFORMERPP_AUDIT_VENV",
        "LAYOUTFORMERPP_DATA_ROOT_ENV_FILE",
    ],
)
def test_relative_child_path_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path_variable: str,
) -> None:
    output_root = tmp_path / "cache"
    output = output_root / "run"
    monkeypatch.setattr(ENVELOPE, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("PACKAGE_ENV", str(tmp_path))
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(tmp_path / "root.env"))
    monkeypatch.delenv("LAYOUTFORMERPP_AUDIT_VENV", raising=False)
    monkeypatch.setenv(relative_path_variable, "relative-runtime")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--output-dir", str(output), "--gpu-index", "0"],
    )
    runner = Mock(return_value={})
    monkeypatch.setattr(ENVELOPE, "_run", runner)

    with pytest.raises(
        SystemExit, match=f"{relative_path_variable} must be an absolute path"
    ):
        ENVELOPE.main()

    runner.assert_not_called()
    assert not output.exists()


def test_cpu_smoke_cli_uses_separate_output_and_variable_update_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "prechecks"
    monkeypatch.setattr(ENVELOPE, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ENVELOPE, "_cuda_is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["runner", "--cpu-smoke", "--updates", "2"])
    runner = Mock(return_value={"updates_per_run": 2})
    monkeypatch.setattr(ENVELOPE, "_run", runner)

    ENVELOPE.main()

    smoke_root = output_root / "cpu-smoke" / "loss-envelope"
    runner.assert_called_once_with(smoke_root, None, updates=2, cpu_smoke=True)


def test_cpu_smoke_refuses_a_visible_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "prechecks"
    monkeypatch.setattr(ENVELOPE, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ENVELOPE, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["runner", "--cpu-smoke", "--updates", "2"])
    runner = Mock(return_value={})
    monkeypatch.setattr(ENVELOPE, "_run", runner)

    with pytest.raises(SystemExit, match="--cpu-smoke requires CUDA to be hidden"):
        ENVELOPE.main()

    runner.assert_not_called()
    assert not (output_root / "cpu-smoke" / "loss-envelope").exists()


def test_updates_can_be_lowered_only_with_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ENVELOPE, "OUTPUT_ROOT", tmp_path / "prechecks")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--updates", "2", "--gpu-index", "0"],
    )
    runner = Mock(return_value={})
    monkeypatch.setattr(ENVELOPE, "_run", runner)

    with pytest.raises(SystemExit) as error:
        ENVELOPE.main()

    assert error.value.code == 2
    assert "--updates must remain 1500 outside --cpu-smoke" in capsys.readouterr().err
    runner.assert_not_called()


def test_cpu_package_entrypoint_uses_only_accelerator_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "package"
    directory.mkdir()
    metrics = directory / "lightning" / "logs" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("step,train_loss_step\n0,1.0\n1,1.1\n", encoding="utf-8")
    campaign = Mock(
        build_argv=Mock(return_value=[["traingen", "fit", "--trainer.devices=1"]])
    )

    def run_entrypoint(command, **kwargs):
        Path(kwargs["env"]["LAYOUTFORMERPP_BATCH_TRACE"]).write_text(
            "".join(
                json.dumps({"update": index + 1, "batch_sha256": f"{index:064x}"})
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        return Mock(returncode=0, stdout="", stderr="")

    run = Mock(side_effect=run_entrypoint)
    monkeypatch.setattr(ENVELOPE.subprocess, "run", run)

    ENVELOPE._run_package(
        campaign,
        seed=0,
        directory=directory,
        environment={"PACKAGE_ENV": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""},
        updates=2,
        cpu_smoke=True,
    )

    command = run.call_args.args[0]
    assert command == [
        "traingen",
        "fit",
        "--trainer.devices=1",
        "--trainer.max_steps=2",
        "--trainer.log_every_n_steps=1",
        "--trainer.enable_progress_bar=false",
        "--trainer.accelerator=cpu",
    ]
    assert run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    rows = [
        json.loads(line)
        for line in (directory / "updates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    assert [row["logger_step"] for row in rows[1:]] == [0, 1]


def test_cpu_smoke_runs_three_seeds_with_full_preflight_and_marks_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    config = repository / "models" / "layoutformerpp" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("resolved config", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    package_environment = tmp_path / "package-env"
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text("", encoding="utf-8")
    campaign = Mock(
        REPOSITORY=repository,
        CONFIG_SHA256={"label": "config-hash"},
        config_path=Mock(return_value=config),
        sha256=Mock(return_value="config-hash"),
        current_commit=Mock(return_value="source-commit"),
        require_clean_tree=Mock(),
        verify_splits=Mock(return_value={"train": {"sha256": "split-hash"}}),
        read_data_root=Mock(return_value=data_root),
        runtime_freeze_sha256=Mock(return_value="runtime-freeze-hash"),
        gpu_info=Mock(side_effect=AssertionError("CPU smoke inspected a GPU")),
        build_environment=Mock(
            side_effect=lambda *args, **kwargs: {
                "PACKAGE_ENV": str(package_environment),
                "CUDA_VISIBLE_DEVICES": "visible-by-default",
            }
        ),
        vendor_commit=Mock(return_value="vendor-commit"),
    )
    monkeypatch.setitem(sys.modules, "launch_track", campaign)
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(environment_file))
    monkeypatch.setenv("PACKAGE_ENV", str(package_environment))
    original = Mock()
    package = Mock()
    compare = Mock(
        return_value={
            "updates_per_run": 2,
            "run_kind": "cpu_smoke",
            "gate_eligible": False,
            "comparison_verdict": "not_meaningful_for_cpu_smoke",
        }
    )
    monkeypatch.setattr(ENVELOPE, "_run_original", original)
    monkeypatch.setattr(ENVELOPE, "_run_package", package)
    monkeypatch.setattr(ENVELOPE, "_require_native_logger_alignment", Mock())
    monkeypatch.setattr(ENVELOPE, "_compare", compare)
    output = tmp_path / "cpu-smoke" / "loss-envelope"
    output.mkdir(parents=True)

    result = ENVELOPE._run(output, gpu_index=None, updates=2, cpu_smoke=True)

    campaign.require_clean_tree.assert_called_once_with()
    campaign.verify_splits.assert_called_once_with(data_root)
    campaign.runtime_freeze_sha256.assert_called_once_with(package_environment)
    campaign.gpu_info.assert_not_called()
    assert original.call_count == len(ENVELOPE.SEEDS)
    assert package.call_count == len(ENVELOPE.SEEDS)
    for call in (*original.call_args_list, *package.call_args_list):
        assert call.kwargs["updates"] == 2
        assert call.kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        assert call.kwargs["environment"]["LAYOUTFORMERPP_BATCH_LIMIT"] == "2"
    assert all(call.kwargs["cpu_smoke"] for call in package.call_args_list)
    compare.assert_called_once_with(
        output / "runs" / "original",
        output / "runs" / "package",
        output / "result.json",
        updates=2,
        cpu_smoke=True,
    )
    assert result["run_kind"] == "cpu_smoke"
    assert result["gate_eligible"] is False
    assert result["comparison_verdict"] == "not_meaningful_for_cpu_smoke"
    assert "passed" not in result
    assert result["runtime_freeze_sha256"] == "runtime-freeze-hash"
