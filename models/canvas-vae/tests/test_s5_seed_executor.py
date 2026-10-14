import ast
import importlib.util
import json
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
        "test-job", args.hub_repo, "s5/test-job", "30m", "true", []
    )
    command_data = remote_source.split("subprocess.run(", maxsplit=1)[1].split(
        ", check=True)", maxsplit=1
    )[0]
    remote_command = json.loads(command_data)

    assert remote_command[remote_command.index("--hub-repo") + 1] == repository


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
        },
        "inputs/project-a",
        "outputs/project-a",
    )

    assert resolved == {
        "prefix": "inputs/project-a/crello-s5-inputs",
        "artifact_prefix": "outputs/project-a/crello/{run_id}",
    }


def test_crello_postprocess_uses_crello_vocab_and_evaluator() -> None:
    manifest = {
        "dataset": "crello",
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

    command = EXECUTOR.build_postprocess_job(manifest, run, "segment-001")

    assert "--dataset crello" in command
    assert ".cache/canvas-vae/original/data/crello-s5/vocabulary.json" in command
    assert "models/canvas-vae/scripts/evaluate_crello.py" in command


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


def test_crello_executor_blocks_without_rehearsed_launch_gates() -> None:
    manifest = {
        "dataset": "crello",
        "runtime_plan": {"integrated_first_run_gates": False},
        "cpu_rehearsal": {"status": "pending"},
    }

    with pytest.raises(EXECUTOR.ExecutorError, match="integrated lockstep"):
        EXECUTOR.verify_crello_launch_gates(manifest, "a" * 40)


def test_crello_cpu_smoke_runs_only_original_seed_path() -> None:
    cases = EXECUTOR.cpu_smoke_case_specs("crello")

    assert [(case[0], case[1]) for case in cases] == [("original-success", "original")]
    assert EXECUTOR.cpu_smoke_case_specs("rico")[-2][2] == "training"


def test_crello_training_job_uses_one_script_for_full_and_tiny_inputs() -> None:
    manifest = {
        "dataset": "crello",
        "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
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
        },
        {
            "run_id": "cpu-smoke-crello-original-success",
            "system": "original",
            "seed": 0,
            "device": "cpu",
            "cuda_visible_devices": "",
            "runtime_output_root": "/content/out/checkpoints/cpu-smoke-crello-original-success/segment-001",
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
    )

    for command in (full_command, tiny_command):
        assert "git rev-parse HEAD" in command
        assert (
            "uv sync --frozen --package canvas-vae --extra training --extra vendor"
            in command
        )
        assert "tiny_original_input_subset_passed" not in command
        assert "cpu_smoke" not in command
        assert "smoke_original" not in command
        assert "pre_training_commands" not in command
        assert "if [" not in command
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


def test_crello_executor_skips_production_gates_only_for_cpu_smoke() -> None:
    EXECUTOR.verify_crello_launch_gates(
        {
            "dataset": "crello",
            "runtime_plan": {"cpu_runtime_smoke": True},
            "cpu_rehearsal": {"status": "pending"},
        },
        "a" * 40,
    )
