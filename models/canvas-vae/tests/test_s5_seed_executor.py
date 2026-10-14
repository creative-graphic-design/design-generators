import ast
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest


EXECUTOR_PATH = Path(__file__).parents[1] / "scripts" / "s5_seed_executor.py"
EXECUTOR_SPEC = importlib.util.spec_from_file_location(
    "s5_seed_executor", EXECUTOR_PATH
)
assert EXECUTOR_SPEC is not None and EXECUTOR_SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(EXECUTOR_SPEC)
EXECUTOR_SPEC.loader.exec_module(EXECUTOR)


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
    assert "/content/colab-uv-setup.sh" in command
    assert "--extra convert" in command
    assert "uv sync" not in command


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
    assert "relative_loss_difference" in full_command
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


def test_crello_executor_skips_production_gates_only_for_cpu_smoke() -> None:
    EXECUTOR.verify_crello_launch_gates(
        {
            "dataset": "crello",
            "runtime_plan": {"cpu_runtime_smoke": True},
            "cpu_rehearsal": {"status": "pending"},
        },
        "a" * 40,
    )
