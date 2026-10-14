import importlib.util
import json
import sys
from pathlib import Path

import pytest


QUEUE_PATH = Path(__file__).parents[1] / "scripts" / "s5_seed_queue.py"
QUEUE_SPEC = importlib.util.spec_from_file_location("s5_seed_queue", QUEUE_PATH)
assert QUEUE_SPEC is not None and QUEUE_SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(QUEUE_SPEC)
QUEUE_SPEC.loader.exec_module(QUEUE)


def test_queue_launch_forwards_hub_repository() -> None:
    repository = "example-owner/test-artifacts"
    manifest_path = QUEUE.DEFAULT_MANIFEST
    parity_path = QUEUE.REPOSITORY_ROOT / "models/canvas-vae/TRAINING.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_commit"] = "a" * 40
    manifest["queue_status"] = "ready"
    manifest["evaluation_path_parity"] = {
        "path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(parity_path),
    }
    manifest["runtime_plan"]["source_commit"] = "a" * 40

    _, command = QUEUE.launch_preconditions(manifest, manifest_path, repository)

    assert command[-2:] == ["--hub-repo", repository]


def test_crello_template_defines_three_seeds_per_system() -> None:
    path = (
        QUEUE.REPOSITORY_ROOT
        / "models/canvas-vae/configs/training/s5_crello_launch_manifest.template.json"
    )
    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest = QUEUE.load_manifest(path)

    assert [(run["system"], run["seed"]) for run in manifest["runs"]] == [
        (system, seed) for system in ("original", "package") for seed in (0, 1, 2)
    ]
    assert manifest["queue_status"].startswith("blocked_")
    assert manifest["input_data"]["source_archive_sha256"] == (
        "f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e"
    )
    assert raw_manifest["embedding_cache_plan"]["original_reference_dir"].endswith(
        "/original-run-1/document"
    )
    assert manifest["runtime_plan"]["one_training_run_at_a_time"] is True
    assert manifest["runtime_plan"]["lockstep_probe_steps"] == 300


def test_crello_launch_requires_runtime_hub_prefixes() -> None:
    path = (
        QUEUE.REPOSITORY_ROOT
        / "models/canvas-vae/configs/training/s5_crello_launch_manifest.template.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_commit"] = "a" * 40
    manifest["queue_status"] = "ready"
    manifest["runtime_plan"]["source_commit"] = "a" * 40
    manifest["runtime_plan"]["a100_wrapper_ttl_by_system"] = {
        "original": "30m",
        "package": "30m",
    }
    manifest["runtime_plan"]["cpu_wrapper_ttl"] = "30m"
    manifest["evaluation_path_parity"] = {
        "path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(
            QUEUE.REPOSITORY_ROOT / "models/canvas-vae/TRAINING.md"
        ),
    }
    manifest["cpu_rehearsal"] = {
        "status": "passed",
        "source_commit": "a" * 40,
        "artifact_path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(
            QUEUE.REPOSITORY_ROOT / "models/canvas-vae/TRAINING.md"
        ),
    }
    for run in manifest["runs"]:
        run["a100_wrapper_ttl"] = "30m"
        run["cpu_wrapper_ttl"] = "30m"

    with pytest.raises(QUEUE.QueueError, match="--input-prefix"):
        QUEUE.launch_preconditions(
            manifest,
            path,
            "example-owner/test-artifacts",
        )

    _, command = QUEUE.launch_preconditions(
        manifest,
        path,
        "example-owner/test-artifacts",
        input_prefix="inputs/project-a",
        output_prefix="outputs/project-a",
    )
    assert command[-4:] == [
        "--input-prefix",
        "inputs/project-a",
        "--output-prefix",
        "outputs/project-a",
    ]


def test_crello_launch_stays_blocked_until_cpu_rehearsal_passes() -> None:
    path = (
        QUEUE.REPOSITORY_ROOT
        / "models/canvas-vae/configs/training/s5_crello_launch_manifest.template.json"
    )
    manifest = QUEUE.load_manifest(path)
    manifest["source_commit"] = "a" * 40
    manifest["queue_status"] = "ready"
    manifest["runtime_plan"]["source_commit"] = "a" * 40
    manifest["evaluation_path_parity"] = {
        "path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(
            QUEUE.REPOSITORY_ROOT / "models/canvas-vae/TRAINING.md"
        ),
    }

    with pytest.raises(QUEUE.QueueError, match="same-environment CPU rehearsal"):
        QUEUE.launch_preconditions(manifest, path, "example-owner/test-artifacts")


def test_crello_cpu_runtime_smoke_does_not_claim_production_gates() -> None:
    path = (
        QUEUE.REPOSITORY_ROOT
        / "models/canvas-vae/configs/training/s5_crello_launch_manifest.template.json"
    )
    manifest = QUEUE.load_manifest(path)
    pinned = "a" * 40
    manifest["source_commit"] = pinned
    manifest["queue_status"] = "ready"
    manifest["runtime_plan"].update(
        {
            "cpu_runtime_smoke": True,
            "gpu": "CPU",
            "source_commit": pinned,
            "a100_wrapper_ttl_by_system": {"original": "30m", "package": "30m"},
            "cpu_wrapper_ttl": "30m",
            "lockstep_probe_steps": 3,
        }
    )
    for index, run in enumerate(manifest["runs"]):
        run["run_id"] = f"cpu-smoke-crello-{index}"
        run["a100_wrapper_ttl"] = "30m"
        run["cpu_wrapper_ttl"] = "30m"

    _, command = QUEUE.launch_preconditions(
        manifest,
        path,
        "example-owner/test-artifacts",
        input_prefix="inputs/project-a",
        output_prefix="outputs/project-a",
    )

    assert command[-4:] == [
        "--input-prefix",
        "inputs/project-a",
        "--output-prefix",
        "outputs/project-a",
    ]

    manifest["dataset"] = "rico"
    with pytest.raises(
        QUEUE.QueueError, match="evaluation_path_parity SHA-256 is missing"
    ):
        QUEUE.launch_preconditions(
            manifest,
            path,
            "example-owner/test-artifacts",
            input_prefix="inputs/project-a",
            output_prefix="outputs/project-a",
        )


@pytest.mark.parametrize("repository", [None, "   "])
def test_queue_refuses_launch_without_hub_repository(
    monkeypatch: pytest.MonkeyPatch,
    repository: str | None,
) -> None:
    args = [str(QUEUE_PATH), "--launch"]
    if repository is not None:
        args.extend(["--hub-repo", repository])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(QUEUE, "load_manifest", lambda _path: {})
    monkeypatch.setattr(QUEUE, "current_source_state", lambda: ("a" * 40, True))
    monkeypatch.setattr(
        QUEUE,
        "run_queue",
        lambda *_args: pytest.fail("queue must not run without a Hub repository"),
    )

    assert QUEUE.main() == QUEUE.BLOCKED_EXIT
