import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias, cast

import pytest


EXECUTOR_PATH = Path(__file__).parents[1] / "scripts" / "s5_seed_executor.py"
EXECUTOR_SPEC = importlib.util.spec_from_file_location(
    "s5_seed_executor", EXECUTOR_PATH
)
assert EXECUTOR_SPEC is not None and EXECUTOR_SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(EXECUTOR_SPEC)
EXECUTOR_SPEC.loader.exec_module(EXECUTOR)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


def test_executor_requires_and_passes_hub_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = "example-owner/test-artifacts"
    monkeypatch.setattr(sys, "argv", [str(EXECUTOR_PATH), "--hub-repo", repository])
    args = EXECUTOR.parse_args()

    remote_source = EXECUTOR.make_wrapper_cell(
        "test-job", args.hub_repo, "s5/test-job", "30m", []
    )
    command_data = remote_source.split("subprocess.run(", maxsplit=1)[1].split(
        ", check=True)", maxsplit=1
    )[0]
    remote_command = json.loads(command_data)

    assert remote_command[remote_command.index("--hub-repo") + 1] == repository
    assert remote_command[-2:] == ["bash", "/content/s5-exec-step.sh"]


@pytest.mark.parametrize(
    "caller_sync_paths",
    [[], ["/content/out/checkpoints"], ["/content/out/logs"]],
)
def test_wrapper_always_syncs_its_log_directory(
    caller_sync_paths: list[str],
) -> None:
    remote_source = EXECUTOR.make_wrapper_cell(
        "test-job",
        "example-owner/test-artifacts",
        "s5/test-job",
        "30m",
        caller_sync_paths,
    )
    call = next(
        node
        for node in ast.walk(ast.parse(remote_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    )
    remote_command = ast.literal_eval(call.args[0])
    synced_paths = [
        remote_command[index + 1]
        for index, argument in enumerate(remote_command[:-1])
        if argument == "--sync-path"
    ]

    assert synced_paths.count("/content/out/logs") == 1
    assert all(path in synced_paths for path in caller_sync_paths)


def test_colab_exec_steps_persist_deliberate_failure_exit_codes(
    tmp_path: Path,
) -> None:
    failing_command = [
        sys.executable,
        "-c",
        "import sys; print('deliberate step failure'); raise SystemExit(19)",
    ]
    cell_exit_path = tmp_path / "preflight-exit-code"
    cell_log_path = tmp_path / "preflight.log"
    cell = EXECUTOR.make_colab_exec_cell(
        failing_command, str(cell_exit_path), str(cell_log_path)
    )
    cell_result = subprocess.run(
        [sys.executable, "-c", cell],
        check=False,
        capture_output=True,
        text=True,
    )

    assert cell_result.returncode == 19
    assert EXECUTOR.read_exit_code_file(cell_exit_path, "test preflight") == 19
    assert "deliberate step failure" in cell_log_path.read_text()

    job_exit_path = tmp_path / "job-exit-code"
    job_script = tmp_path / "exec-step.sh"
    job_script.write_text(
        EXECUTOR.make_exec_step_script(failing_command, str(job_exit_path))
    )
    job_script.chmod(0o700)
    job_result = subprocess.run(
        ["bash", str(job_script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert job_result.returncode == 19
    assert EXECUTOR.read_exit_code_file(job_exit_path, "test job") == 19
    assert "deliberate step failure" in job_result.stdout


def test_local_upload_preflight_checks_recorded_sha256(tmp_path: Path) -> None:
    upload = tmp_path / "upload.txt"
    upload.write_text("expected payload", encoding="utf-8")
    expected_sha256 = EXECUTOR.sha256_file(upload)

    EXECUTOR.validate_local_uploads(((upload, "/content/upload.txt", expected_sha256),))

    with pytest.raises(EXECUTOR.ExecutorError, match="SHA-256 differs"):
        EXECUTOR.validate_local_uploads(((upload, "/content/upload.txt", "0" * 64),))


def test_missing_local_upload_fails_before_colab_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_path = tmp_path / "wrapper.sh"
    preflight_path = tmp_path / "preflight.json"
    preflight_script = tmp_path / "preflight.py"
    run_script = tmp_path / "run.sh"
    token_path = tmp_path / "token"
    run_env = tmp_path / "run.env"
    for path in (
        wrapper_path,
        preflight_path,
        preflight_script,
        run_script,
        token_path,
        run_env,
    ):
        path.write_text("placeholder", encoding="utf-8")

    colab_calls: list[tuple[object, ...]] = []

    def record_colab_call(*args: object, **kwargs: object) -> str:
        colab_calls.append(args)
        return ""

    monkeypatch.setattr(EXECUTOR, "colab", record_colab_call)

    with pytest.raises(
        EXECUTOR.ExecutorError,
        match="local upload for /content/missing.txt is missing",
    ):
        EXECUTOR.run_named_session(
            session_name="missing-upload-test",
            gpu=None,
            wrapper_path=wrapper_path,
            token_path=token_path,
            preflight_path=preflight_path,
            preflight_script=preflight_script,
            run_script=run_script,
            job_name="missing-upload-test",
            hub_prefix="test/output",
            ttl="30m",
            run_command="bash /content/run-job.sh",
            sync_paths=("/content/out/logs",),
            run_env=run_env,
            output_capture=tmp_path / "output.log",
            hub_repository="example-owner/test-artifacts",
            download_inputs=False,
            additional_uploads=(
                (tmp_path / "missing.txt", "/content/missing.txt", None),
            ),
        )

    assert colab_calls == []


def test_staged_preflight_failure_uploads_sanitized_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_path = tmp_path / "wrapper.sh"
    preflight_path = tmp_path / "preflight.json"
    preflight_script = tmp_path / "preflight.py"
    run_script = tmp_path / "run.sh"
    token_path = tmp_path / "token"
    run_env = tmp_path / "run.env"
    for path in (
        wrapper_path,
        preflight_path,
        preflight_script,
        run_script,
        token_path,
        run_env,
    ):
        path.write_text("placeholder", encoding="utf-8")

    fake_token = "hf_" + "A" * 30
    uploaded_artifacts: dict[str, bytes] = {}
    verified_paths: list[str] = []
    stopped_sessions: list[str] = []

    def record_colab_call(*args: str, **kwargs: object) -> str:
        if args[0] == "download":
            remote_path = args[3]
            local_path = Path(args[4])
            if remote_path == "/content/s5-preflight-exit-code":
                local_path.write_text("23\n", encoding="utf-8")
            elif remote_path == "/content/s5-preflight.log":
                local_path.write_text(
                    f"setup failed with token {fake_token}\n", encoding="utf-8"
                )
        elif args[0] == "exec":
            raise EXECUTOR.ExecutorError(f"Colab exec failed with {fake_token}")
        elif args[0] == "stop":
            stopped_sessions.append(args[2])

        return ""

    def capture_hub_upload(
        token_file: Path,
        source: str,
        temporary_directory: Path,
        extra_environment: dict[str, str] | None = None,
    ) -> str:
        assert "api.upload_folder(" in source
        assert extra_environment is not None
        artifact_directory = Path(extra_environment["S5_PREFLIGHT_FAILURE_DIRECTORY"])
        for artifact in artifact_directory.iterdir():
            uploaded_artifacts[artifact.name] = artifact.read_bytes()

        return ""

    def record_hub_verification(
        token_file: Path,
        repository: str,
        expected_paths: list[str],
        description: str,
    ) -> None:
        assert description == "staged-input preflight failure artifacts"
        verified_paths.extend(expected_paths)

    monkeypatch.setattr(EXECUTOR, "assert_session_is_free", lambda name: None)
    monkeypatch.setattr(EXECUTOR, "colab", record_colab_call)
    monkeypatch.setattr(EXECUTOR, "hub_python", capture_hub_upload)
    monkeypatch.setattr(EXECUTOR, "verify_hub_paths", record_hub_verification)

    with pytest.raises(EXECUTOR.ExecutorError, match="redaction_count=2") as error:
        EXECUTOR.run_named_session(
            session_name="staged-preflight-failure-test",
            gpu=None,
            wrapper_path=wrapper_path,
            token_path=token_path,
            preflight_path=preflight_path,
            preflight_script=preflight_script,
            run_script=run_script,
            job_name="staged-preflight-failure-test",
            hub_prefix="test/output",
            ttl="30m",
            run_command="bash /content/run-job.sh",
            sync_paths=("/content/out/logs",),
            run_env=run_env,
            output_capture=tmp_path / "output.log",
            hub_repository="example-owner/test-artifacts",
            download_inputs=False,
        )

    uploaded_log = uploaded_artifacts["preflight.log"].decode("utf-8")
    uploaded_report = json.loads(uploaded_artifacts["report.json"])
    assert fake_token not in uploaded_log
    assert "[REDACTED_HF_TOKEN]" in uploaded_log
    assert uploaded_artifacts["exit_code"] == b"23\n"
    assert uploaded_report["exit_code"] == 23
    assert uploaded_report["redaction_count"] == 2
    assert "[REDACTED_HF_TOKEN]" in uploaded_report["colab_cli_error"]
    assert fake_token not in str(uploaded_report)
    assert (
        uploaded_report["sanitized_log_sha256"]
        == hashlib.sha256(uploaded_artifacts["preflight.log"]).hexdigest()
    )
    assert "[REDACTED_HF_TOKEN]" in str(error.value)
    assert fake_token not in str(error.value)
    assert verified_paths == [
        "test/output/logs/staged-preflight/staged-preflight-failure-test/preflight.log",
        "test/output/logs/staged-preflight/staged-preflight-failure-test/exit_code",
        "test/output/logs/staged-preflight/staged-preflight-failure-test/report.json",
    ]
    assert stopped_sessions == ["staged-preflight-failure-test"]


def test_executor_refuses_to_run_without_hub_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(EXECUTOR_PATH)])

    with pytest.raises(SystemExit) as error:
        EXECUTOR.main()

    assert error.value.code == 2


def test_runtime_hub_prefixes_are_resolved_without_committing_private_paths() -> None:
    resolved = EXECUTOR.resolve_runtime_prefixes(
        {
            "prefix": "{input_prefix}/crello-s5-inputs",
            "artifact_prefix": "{output_prefix}/crello/{run_id}",
            "fixture_prefix": "{probe_fixture_prefix}/cache",
        },
        "inputs/project-a",
        "outputs/project-a",
        "cache/project-a",
    )

    assert resolved == {
        "prefix": "inputs/project-a/crello-s5-inputs",
        "artifact_prefix": "outputs/project-a/crello/{run_id}",
        "fixture_prefix": "cache/project-a/cache",
    }


def test_crello_postprocess_uses_crello_vocab_and_evaluator() -> None:
    manifest = {
        "dataset": "crello",
        "runtime_plan": {"source_branch": "feat/canvas-vae-crello-model"},
        "dataset_paths": {
            "package_vocabulary": ".cache/canvas-vae/crello/package-run-1/vocabulary.json",
            "original_vocabulary": ".cache/canvas-vae/original/data/crello-s5/vocabulary.json",
        },
        "evaluator_command": [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            "/content/repo",
            "--package",
            "canvas-vae",
            "--extra",
            "training",
            "models/canvas-vae/scripts/evaluate_crello.py",
            "--checkpoint",
            "{converted_checkpoint}",
            "--output",
            "{evaluation_output}",
        ],
    }
    run = {
        "run_id": "crello-original-seed0",
        "system": "original",
        "converted_output_root": "/content/out/converted/crello-original-seed0/{segment_id}",
        "evaluation_output": "/content/out/results/crello-original-seed0/{segment_id}/evaluation.json",
        "epoch_500_checkpoint": "/content/out/checkpoints/crello-original-seed0/{segment_id}/epoch-500.ckpt",
    }

    command = EXECUTOR.build_postprocess_job(manifest, run, "segment-001", "a" * 40)

    assert "--dataset crello" in command
    assert ".cache/canvas-vae/original/data/crello-s5/vocabulary.json" in command
    assert "models/canvas-vae/scripts/evaluate_crello.py" in command
    assert "/content/colab-uv-setup.sh" not in command
    assert "--extra convert" in command
    assert "--extra training" in command
    assert "--no-sync" in command
    assert "uv sync" not in command
    assert "UV_PROJECT_ENVIRONMENT=/content/canvas-vae-postprocess-env" in command
    assert "unset UV_PROJECT_ENVIRONMENT" not in command
    assert "repository .venv exists; refusing environment drift" in command


def test_crello_package_training_dry_run_contains_final_checkpoint_gate() -> None:
    command = EXECUTOR.build_run_job(
        {
            "dataset": "crello",
            "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
            "package_checkpoint_expectation": {"max_epochs": 500, "global_step": 9500},
        },
        {
            "run_id": "crello-package-seed0",
            "system": "package",
            "seed": 0,
            "device": "cuda",
            "runtime_output_root": "/content/out/checkpoints/crello-package-seed0/{segment_id}",
            "training_overrides": ["--model.init_args.fixture_dir=.cache/embeddings"],
        },
        "a" * 40,
        "segment-001",
        resume_checkpoint=None,
    )

    assert "--expected-global-step 9500" in command
    assert "--max-epochs 500" in command
    assert "--model.init_args.fixture_dir=.cache/embeddings" in command
    assert "UV_PROJECT_ENVIRONMENT=/content/canvas-vae-training-env" in command
    assert "repository .venv exists; refusing environment drift" in command


def test_crello_remote_preflight_rejects_wrong_original_document_counts() -> None:
    source = EXECUTOR.remote_preflight_source()
    compile(source, "<crello-s5-preflight>", "exec")
    remote_module = ast.parse(source)
    validator = next(
        node
        for node in remote_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_original_document_counts"
    )
    validator_module = ast.Module(body=[validator], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.fix_missing_locations(validator_module),
            "<crello-s5-preflight>",
            "exec",
        ),
        namespace,
    )

    expected_counts = {"train": 2, "val": 1, "test": 1}
    manifest: dict[str, object] = {"document_counts": expected_counts}
    count_file = expected_counts.copy()
    actual_records = {"train": 1, "val": 1, "test": 1}
    validate_counts = cast(
        Callable[
            [Mapping[str, object], Mapping[str, int], Mapping[str, int]],
            dict[str, int],
        ],
        namespace["validate_original_document_counts"],
    )

    assert validate_counts(manifest, count_file, expected_counts) == expected_counts
    with pytest.raises(
        SystemExit,
        match="original train record count differs from manifest",
    ):
        validate_counts(manifest, count_file, actual_records)


def test_crello_executor_blocks_without_same_environment_cpu_rehearsal() -> None:
    manifest = {
        "dataset": "crello",
        "runtime_plan": {},
        "cpu_rehearsal": {"status": "pending"},
    }

    with pytest.raises(EXECUTOR.ExecutorError, match="same-environment CPU rehearsal"):
        EXECUTOR.verify_crello_launch_gates(manifest, "a" * 40)


def test_crello_cpu_smoke_runs_only_original_seed_path() -> None:
    cases = EXECUTOR.cpu_smoke_case_specs("crello")

    assert [(case[0], case[1]) for case in cases] == [("original-success", "original")]
    assert EXECUTOR.cpu_smoke_case_specs("rico")[-2][2] == "training"


def test_crello_training_job_uses_one_script_for_full_and_tiny_inputs() -> None:
    manifest = {
        "dataset": "crello",
        "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
        "dataset_paths": {
            "package_data_dir": ".cache/canvas-vae/crello/package-run-1",
            "original_data_dir": ".cache/canvas-vae/original/data/crello-s5",
            "original_fixture_dir": ".cache/canvas-vae/crello/s5/original-embeddings",
        },
    }
    full_command = EXECUTOR.build_run_job(
        manifest,
        {
            "run_id": "crello-original-seed0",
            "system": "original",
            "seed": 0,
            "device": "cuda",
            "cuda_visible_devices": "0",
            "runtime_output_root": "/content/out/checkpoints/crello-original-seed0/{segment_id}",
            "lockstep_probe_inputs": {
                "package_data_dir": ".cache/canvas-vae/crello/package-run-1",
                "fixture_dir": ".cache/canvas-vae/crello/s5/original-seed0-encoder-cache/cache",
                "fixture_archive": {
                    "prefix": "inputs/full-cache",
                    "revision": "d" * 40,
                    "archive": {"filename": "cache.tar", "sha256": "e" * 64},
                },
            },
            "train_original_arguments": [
                "--dataset-name",
                "crello-document",
                "--data-dir",
                "/content/repo/.cache/canvas-vae/original/data/crello-s5",
                "--job-dir",
                "{output_root}/original-training",
                "--seed",
                "{seed}",
                "--",
                "--batch-size",
                "1024",
                "--num-epochs",
                "500",
                "--validation-freq",
                "20",
            ],
        },
        "a" * 40,
        "segment-001",
        resume_checkpoint=None,
    )
    tiny_command = EXECUTOR.build_run_job(
        {
            "dataset": "crello",
            "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
            "dataset_paths": {
                "package_data_dir": ".cache/canvas-vae/crello/package-run-1",
                "original_data_dir": ".cache/canvas-vae/original/data/crello-s5",
                "original_fixture_dir": ".cache/canvas-vae/crello/s5/original-embeddings",
            },
        },
        {
            "run_id": "cpu-smoke-crello-original-success",
            "system": "original",
            "seed": 0,
            "device": "cpu",
            "cuda_visible_devices": "",
            "runtime_output_root": "/content/out/checkpoints/cpu-smoke-crello-original-success/segment-001",
            "lockstep_probe_inputs": {
                "package_data_dir": ".cache/canvas-vae/crello/s5/probe-package-run-1-tiny",
                "package_data_archive": {
                    "prefix": "inputs/tiny-data",
                    "revision": "b" * 40,
                    "archive": {"filename": "tiny.tar", "sha256": "c" * 64},
                },
                "fixture_dir": ".cache/canvas-vae/crello/s5/probe-original-cache-tiny/cache",
                "fixture_archive": {
                    "prefix": "inputs/tiny-cache",
                    "revision": "f" * 40,
                    "archive": {"filename": "cache.tar", "sha256": "e" * 64},
                },
            },
            "train_original_arguments": [
                "--dataset-name",
                "crello-document",
                "--data-dir",
                "/content/repo/.cache/canvas-vae/original/data/crello-s5",
                "--job-dir",
                "{output_root}/original-training",
                "--seed",
                "0",
                "--",
                "--batch-size",
                "2",
                "--num-epochs",
                "1",
                "--validation-freq",
                "1",
            ],
        },
        "a" * 40,
        "segment-001",
        resume_checkpoint=None,
        lockstep_probe_steps=3,
    )

    for command in (full_command, tiny_command):
        assert "git rev-parse HEAD" in command
        assert "uv sync" not in command
        assert "UV_PROJECT_ENVIRONMENT=/content/canvas-vae-training-env" in command
        assert "repository .venv exists; refusing environment drift" in command
        assert "unset UV_PROJECT_ENVIRONMENT" not in command
        assert "uv run --frozen --no-sync" in command
        assert "tiny_original_input_subset_passed" not in command
        assert "cpu_smoke" not in command
        assert "smoke_original" not in command
        assert "pre_training_commands" not in command
        assert "if [" not in command

    assert "crello_lockstep_probe.py" in full_command
    assert "--steps 300" in full_command
    assert (
        "--data-dir /content/repo/.cache/canvas-vae/crello/package-run-1"
        in full_command
    )
    assert (
        "--fixture-dir /content/repo/.cache/canvas-vae/crello/s5/original-seed0-encoder-cache/cache"
        in full_command
    )
    assert "validate_lockstep_probe_report" in full_command
    assert full_command.index("crello_lockstep_probe.py") < full_command.index(
        "train_original.py"
    )
    assert "crello_lockstep_probe.py" in tiny_command
    assert "--steps 3" in tiny_command
    assert "--device cpu" in tiny_command
    assert (
        "--data-dir /content/repo/.cache/canvas-vae/crello/s5/probe-package-run-1-tiny"
        in tiny_command
    )
    assert (
        "--fixture-dir /content/repo/.cache/canvas-vae/crello/s5/probe-original-cache-tiny/cache"
        in tiny_command
    )
    assert tiny_command.index("crello_lockstep_probe.py") < tiny_command.index(
        "train_original.py"
    )
    assert "export CUDA_VISIBLE_DEVICES=0" in full_command
    assert "export CUDA_VISIBLE_DEVICES=''" in tiny_command
    assert "--batch-size 1024" in full_command
    assert "--num-epochs 500" in full_command
    assert "--batch-size 2" in tiny_command
    assert "--num-epochs 1" in tiny_command
    assert "canvas-vae/original/data/crello-s5" in tiny_command
    assert (
        "--data-dir /content/repo/.cache/canvas-vae/original/data/crello-s5"
        in tiny_command
    )


def _lockstep_report(
    cross_differences: list[float], noise_differences: list[float]
) -> tuple[JsonObject, list[JsonObject]]:
    records: list[JsonObject] = []
    phases: dict[str, JsonValue] = {}
    for phase, comparison, differences in (
        ("cross_system", "tensorflow_vs_package", cross_differences),
        (
            "noise_floor",
            "package_vs_one_ulp_perturbed_package",
            noise_differences,
        ),
    ):
        phase_rows: list[JsonObject] = []
        for step, difference in enumerate(differences, start=1):
            row: JsonObject = {
                "phase": phase,
                "comparison": comparison,
                "step": step,
                "loss_left": 1.0,
                "loss_right": 1.0 + difference,
                "relative_loss_difference": difference,
                "gradient_norm_left": 1.0,
                "gradient_norm_right": 1.0,
                "parameter_max_abs_difference": difference,
                "parameter_with_max_difference": "encoder.weight",
                "step_seconds_report_only": 1.0,
                "documents_per_second_report_only": 1.0,
                "sampler_state": {
                    "seed": 0,
                    "source_indices": [step],
                    "batch_sha256": f"{step:064x}",
                    "posterior_noise_sha256": f"{step + 10:064x}",
                },
            }
            phase_rows.append(row)
            records.append(row)

        first = next(
            (
                row
                for row in phase_rows
                if cast(float, row["relative_loss_difference"]) > 1e-3
            ),
            None,
        )
        phases[phase] = {
            "required_steps": len(differences),
            "completed_steps": len(differences),
            "first_step_over_tolerance": None
            if first is None
            else {
                key: first[key]
                for key in (
                    "step",
                    "relative_loss_difference",
                    "parameter_with_max_difference",
                    "parameter_max_abs_difference",
                )
            },
            "sampler_batch_sha256": hashlib.sha256(
                "\n".join(
                    f"{step:064x}" for step in range(1, len(differences) + 1)
                ).encode("ascii")
            ).hexdigest(),
        }

    cross_phase = cast(JsonObject, phases["cross_system"])
    noise_phase = cast(JsonObject, phases["noise_floor"])
    cross_first = cast(JsonObject | None, cross_phase["first_step_over_tolerance"])
    noise_first = cast(JsonObject | None, noise_phase["first_step_over_tolerance"])
    passed = cross_first is None or (
        noise_first is not None
        and cast(int, cross_first["step"]) >= cast(int, noise_first["step"])
    )
    return (
        {
            "protocol": "crello-real-scale-lockstep-v2",
            "status": "completed",
            "source_commit": "a" * 40,
            "required_steps": len(cross_differences),
            "completed_steps_per_phase": len(cross_differences),
            "relative_loss_limit": 1e-3,
            "gate_fields": ["finite_metrics", "noise_floor_attribution"],
            "throughput_fields_are_report_only": True,
            "initial_state_sha256": "b" * 64,
            "noise_floor_initial_state_sha256": "b" * 64,
            "noise_floor_perturbed_parameter_values": 42,
            "phases": phases,
            "first_step_over_tolerance": cross_first,
            "noise_floor_first_step_over_tolerance": noise_first,
            "noise_floor_attribution_passed": passed,
        },
        records,
    )


@pytest.mark.parametrize(
    ("cross_differences", "noise_differences", "passes"),
    [
        ([0.0, 0.0012, 0.0013], [0.0, 0.0011, 0.0014], True),
        ([0.0, 0.0012, 0.0013], [0.0, 0.0, 0.0], False),
        ([0.0, 0.0, 0.0], [0.0, 0.0011, 0.0013], True),
    ],
)
def test_lockstep_noise_floor_gate_uses_fixed_first_exceedance_rule(
    cross_differences: list[float],
    noise_differences: list[float],
    passes: bool,
) -> None:
    summary, records = _lockstep_report(cross_differences, noise_differences)

    if passes:
        EXECUTOR.validate_lockstep_probe_report(
            summary,
            records,
            required_steps=3,
            expected_commit="a" * 40,
        )
    else:
        with pytest.raises(
            EXECUTOR.ExecutorError,
            match="cross-system divergence exceeds the package-only noise floor",
        ):
            EXECUTOR.validate_lockstep_probe_report(
                summary,
                records,
                required_steps=3,
                expected_commit="a" * 40,
            )


def test_original_seed_preflight_uses_shared_setup_and_vendor_extra(
    tmp_path: Path,
) -> None:
    preflight_path, preflight_script, _ = EXECUTOR.temporary_files(
        tmp_path,
        {
            "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
            "dataset": "crello",
            "input_data": {"revision": "a" * 40, "prefix": "inputs"},
            "runtime_plan": {"source_branch": "feat/canvas-vae-crello-model"},
        },
        {
            "system": "original",
            "lockstep_probe_inputs": {
                "package_data_dir": ".cache/probe-data",
                "package_data_archive": {
                    "prefix": "data-prefix",
                    "revision": "b" * 40,
                    "archive": {"filename": "data.tar", "sha256": "c" * 64},
                },
                "fixture_dir": ".cache/probe-fixture",
                "fixture_archive": {
                    "prefix": "fixture-prefix",
                    "revision": "c" * 40,
                    "archive": {"filename": "cache.tar", "sha256": "d" * 64},
                },
            },
        },
        "b" * 40,
        "example-owner/artifacts",
        download_inputs=True,
    )
    preflight = json.loads(preflight_path.read_text())
    source = preflight_script.read_text()

    assert preflight["uv_extras"] == ["training", "vendor"]
    assert preflight["uv_environment"] == "/content/canvas-vae-training-env"
    assert (
        preflight["lockstep_probe_inputs"]["fixture_archive"]["archive"]["sha256"]
        == "d" * 64
    )
    assert (
        preflight["lockstep_probe_inputs"]["package_data_archive"]["archive"]["sha256"]
        == "c" * 64
    )
    assert preflight["initialize_vendor_submodule"] is True
    assert '"/content/colab-uv-setup.sh"' in source
    assert "download_probe_archive" in source
    assert "extract_archive_root" in source
    assert "git clone" not in source
    assert "submodule update" not in source
    assert "uv sync" not in source


def test_original_postprocess_preflight_uses_one_combined_environment(
    tmp_path: Path,
) -> None:
    preflight_path, _, _ = EXECUTOR.temporary_files(
        tmp_path,
        {
            "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
            "dataset": "crello",
            "input_data": {"revision": "a" * 40, "prefix": "inputs"},
            "runtime_plan": {"source_branch": "feat/canvas-vae-crello-model"},
        },
        {"system": "original", "cpu_postprocess": True},
        "b" * 40,
        "example-owner/artifacts",
        download_inputs=False,
        phase="postprocess",
    )
    preflight = json.loads(preflight_path.read_text())

    assert preflight["uv_extras"] == ["training", "convert"]
    assert preflight["uv_environment"] == "/content/canvas-vae-postprocess-env"


def test_crello_executor_skips_production_gates_only_for_cpu_smoke() -> None:
    EXECUTOR.verify_crello_launch_gates(
        {
            "dataset": "crello",
            "runtime_plan": {"cpu_runtime_smoke": True},
            "cpu_rehearsal": {"status": "pending"},
        },
        "a" * 40,
    )
