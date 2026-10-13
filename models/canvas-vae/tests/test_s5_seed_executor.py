import importlib.util
import json
import sys
from pathlib import Path

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


def test_crello_remote_preflight_checks_both_encoder_caches() -> None:
    source = EXECUTOR.remote_preflight_source()
    compile(source, "<crello-s5-preflight>", "exec")

    assert 'paths["original_fixture_dir"]' in source
    assert 'paths["package_fixture_dir"]' in source
    assert "original_embedding_manifest_sha256" in source


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


def test_crello_cpu_smoke_original_job_uses_manifest_source_and_one_epoch() -> None:
    command = EXECUTOR.build_run_job(
        {
            "dataset": "crello",
            "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
        },
        {
            "run_id": "cpu-smoke-crello-original-success",
            "system": "original",
            "seed": 0,
            "runtime_output_root": "/content/out/checkpoints/cpu-smoke-crello-original-success/segment-001",
            "cpu_smoke": True,
            "smoke_original_data_dir": "/content/repo/.cache/canvas-vae/original-smoke/run/data/crello-s5",
            "smoke_original_source_data_dir": "/content/repo/.cache/canvas-vae/original/data/crello-s5",
            "smoke_original_record_limit": 2,
            "train_original_arguments": [
                "--dataset-name",
                "crello-document",
                "--data-dir",
                "/content/repo/.cache/canvas-vae/original-smoke/run/data/crello-s5",
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

    assert (
        "source = Path('/content/repo/.cache/canvas-vae/original/data/crello-s5')"
        in command
    )
    assert "crello-document" in command
    assert "--num-epochs 1" in command
    assert "canvas-vae/original/data/rico" not in command


def test_crello_executor_skips_production_gates_only_for_cpu_smoke() -> None:
    EXECUTOR.verify_crello_launch_gates(
        {
            "dataset": "crello",
            "runtime_plan": {"cpu_runtime_smoke": True},
            "cpu_rehearsal": {"status": "pending"},
        },
        "a" * 40,
    )
