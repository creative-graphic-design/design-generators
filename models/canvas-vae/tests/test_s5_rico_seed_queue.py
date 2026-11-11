import importlib.util
import json
import sys
from pathlib import Path

import pytest


QUEUE_PATH = Path(__file__).parents[1] / "scripts" / "s5_rico_seed_queue.py"
QUEUE_SPEC = importlib.util.spec_from_file_location("s5_rico_seed_queue", QUEUE_PATH)
assert QUEUE_SPEC is not None and QUEUE_SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(QUEUE_SPEC)
QUEUE_SPEC.loader.exec_module(QUEUE)


def test_queue_launch_forwards_hub_repository() -> None:
    repository = "example-owner/test-artifacts"
    manifest_path = QUEUE.DEFAULT_MANIFEST
    parity_path = QUEUE.REPOSITORY_ROOT / "models" / "canvas-vae" / "TRAINING.md"
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


def test_queue_validates_generic_gpu_schema_and_label() -> None:
    manifest_path = QUEUE.DEFAULT_MANIFEST
    parity_path = QUEUE.REPOSITORY_ROOT / "models" / "canvas-vae" / "TRAINING.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_commit"] = "a" * 40
    manifest["queue_status"] = "ready"
    manifest["evaluation_path_parity"] = {
        "path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(parity_path),
    }
    manifest["runtime_plan"]["source_commit"] = "a" * 40

    assert manifest["runtime_plan"]["gpu"] == "GPU"
    assert manifest["runtime_plan"]["one_named_gpu_session_per_run"] is True
    assert "gpu_wrapper_ttl_by_system" in manifest["runtime_plan"]
    _, command = QUEUE.launch_preconditions(
        manifest, manifest_path, "example-owner/test-artifacts"
    )
    assert command[-2:] == ["--hub-repo", "example-owner/test-artifacts"]

    for invalid_value in ("OTHER", None):
        invalid_manifest = json.loads(json.dumps(manifest))
        if invalid_value is None:
            invalid_manifest["runtime_plan"].pop("gpu")
        else:
            invalid_manifest["runtime_plan"]["gpu"] = invalid_value

        with pytest.raises(QUEUE.QueueError, match="manifest runtime plan"):
            QUEUE.launch_preconditions(
                invalid_manifest,
                manifest_path,
                "example-owner/test-artifacts",
            )


def test_queue_forwards_colab_accelerator_to_each_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = QUEUE.load_manifest(QUEUE.DEFAULT_MANIFEST)
    commands: list[list[str]] = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(QUEUE, "source_gate", lambda *_args, **_kwargs: None)

    def capture_subprocess(command: list[str], **_kwargs: object) -> Completed:
        commands.append(command)
        return Completed()

    monkeypatch.setattr(QUEUE.subprocess, "run", capture_subprocess)
    QUEUE.run_queue(
        manifest,
        "a" * 40,
        ["python3", "executor.py"],
        False,
        "configured-accelerator",
    )

    assert commands
    assert all(
        command[-2:] == ["--colab-accelerator", "configured-accelerator"]
        for command in commands
    )


def test_cpu_smoke_queue_launch_omits_colab_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = QUEUE.load_manifest(QUEUE.DEFAULT_MANIFEST)
    manifest["runtime_plan"]["cpu_runtime_smoke"] = True
    commands: list[list[str]] = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(QUEUE, "source_gate", lambda *_args, **_kwargs: None)

    def capture_subprocess(command: list[str], **_kwargs: object) -> Completed:
        commands.append(command)
        return Completed()

    monkeypatch.setattr(QUEUE.subprocess, "run", capture_subprocess)
    QUEUE.run_queue(manifest, "a" * 40, ["python3", "executor.py"], False)

    assert commands
    assert all("--colab-accelerator" not in command for command in commands)


def test_non_smoke_queue_launch_requires_colab_accelerator() -> None:
    manifest = QUEUE.load_manifest(QUEUE.DEFAULT_MANIFEST)

    with pytest.raises(
        QUEUE.QueueError,
        match="--colab-accelerator is required for a non-smoke queue launch",
    ):
        QUEUE.run_queue(manifest, "a" * 40, ["python3", "executor.py"], False)


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


@pytest.mark.parametrize(
    ("cpu_runtime_smoke", "expected_exit"),
    [(True, 0), (False, QUEUE.BLOCKED_EXIT)],
)
def test_queue_launch_requires_accelerator_only_for_non_smoke_manifest(
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime_smoke: bool,
    expected_exit: int,
) -> None:
    manifest = {"runtime_plan": {"cpu_runtime_smoke": cpu_runtime_smoke}}
    run_calls: list[str | None] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(QUEUE_PATH),
            "--launch",
            "--hub-repo",
            "example-owner/test-artifacts",
        ],
    )
    monkeypatch.setattr(QUEUE, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(QUEUE, "current_source_state", lambda: ("a" * 40, True))
    monkeypatch.setattr(
        QUEUE,
        "launch_preconditions",
        lambda *_args, **_kwargs: ("a" * 40, ["python3", "executor.py"]),
    )
    monkeypatch.setattr(
        QUEUE,
        "run_queue",
        lambda _manifest, _commit, _runner, _dry_run, accelerator: run_calls.append(
            accelerator
        ),
    )

    assert QUEUE.main() == expected_exit
    assert run_calls == ([None] if cpu_runtime_smoke else [])
