import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


EXECUTOR_PATH = Path(__file__).parents[1] / "scripts" / "s5_rico_seed_executor.py"
EXECUTOR_SPEC = importlib.util.spec_from_file_location(
    "s5_rico_seed_executor", EXECUTOR_PATH
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


@pytest.mark.parametrize(
    ("cpu_runtime_smoke", "error_match"),
    [(False, "--colab-accelerator is required"), (True, "source_commit")],
)
def test_executor_requires_accelerator_only_for_non_smoke_manifests(
    tmp_path: Path, cpu_runtime_smoke: bool, error_match: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"runtime_plan": {"cpu_runtime_smoke": cpu_runtime_smoke}}),
        encoding="utf-8",
    )
    with pytest.raises(
        EXECUTOR.ExecutorError,
        match=error_match,
    ):
        EXECUTOR.execute_run(
            manifest_path,
            "run-1",
            "segment-001",
            "example-owner/test-artifacts",
            None,
        )


def test_execute_run_passes_supplied_accelerator_to_colab_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = "configured-accelerator"
    template_path = (
        EXECUTOR.REPOSITORY_ROOT
        / "models"
        / "canvas-vae"
        / "configs"
        / "training"
        / "s5_rico_launch_manifest.template.json"
    )
    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    pinned_commit = EXECUTOR.git_output("rev-parse", "HEAD")
    manifest["source_commit"] = pinned_commit
    manifest["queue_status"] = "ready"
    config_path = EXECUTOR.REPOSITORY_ROOT / manifest["config"]
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest["runtime_plan"]["source_commit"] = pinned_commit
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrapper_path = tmp_path / "wrapper.sh"
    wrapper_path.write_text("#!/bin/sh\n", encoding="utf-8")
    token_path = tmp_path / "token"
    token_path.write_text("hf_test-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setenv("S5_WRAPPER_PATH", str(wrapper_path))
    monkeypatch.setenv("S5_HF_TOKEN_FILE", str(token_path))

    colab_commands: list[list[str]] = []
    original_subprocess_run = subprocess.run

    class StopAfterColabCreate(Exception):
        pass

    def capture_subprocess(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[: len(EXECUTOR.COLAB_COMMAND)] == EXECUTOR.COLAB_COMMAND:
            colab_args = command[len(EXECUTOR.COLAB_COMMAND) :]
            if colab_args[0] == "sessions":
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if colab_args[0] == "new":
                colab_commands.append(command)
                raise StopAfterColabCreate
        if command[0] == "git":
            return original_subprocess_run(
                command,
                cwd=EXECUTOR.REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(EXECUTOR.subprocess, "run", capture_subprocess)
    run = manifest["runs"][0]
    with pytest.raises(StopAfterColabCreate):
        EXECUTOR.execute_run(
            manifest_path,
            run["run_id"],
            "segment-001",
            "example-owner/test-artifacts",
            None,
            accelerator,
        )

    assert colab_commands[-1][len(EXECUTOR.COLAB_COMMAND) :] == [
        "new",
        "--session",
        run["session_name"],
        "--gpu",
        accelerator,
    ]
