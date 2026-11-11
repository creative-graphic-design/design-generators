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


def test_full_run_requires_colab_accelerator_before_loading_manifest() -> None:
    with pytest.raises(
        EXECUTOR.ExecutorError,
        match="--colab-accelerator is required for a non-smoke Colab launch",
    ):
        EXECUTOR.execute_run(
            Path("missing-manifest.json"),
            "run-1",
            "segment-001",
            "example-owner/test-artifacts",
            None,
        )


def test_run_named_session_passes_supplied_accelerator_to_colab_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = "configured-accelerator"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXECUTOR_PATH),
            "--hub-repo",
            "example-owner/test-artifacts",
            "--colab-accelerator",
            accelerator,
        ],
    )
    args = EXECUTOR.parse_args()

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

    commands: list[list[str]] = []

    class StopAfterLaunch(Exception):
        pass

    def capture_subprocess(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[4] == "new":
            raise StopAfterLaunch

        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(EXECUTOR.subprocess, "run", capture_subprocess)
    with pytest.raises(StopAfterLaunch):
        EXECUTOR.run_named_session(
            session_name="accelerator-forwarding-test",
            gpu=args.colab_accelerator,
            wrapper_path=wrapper_path,
            token_path=token_path,
            preflight_path=preflight_path,
            preflight_script=preflight_script,
            run_script=run_script,
            job_name="accelerator-forwarding-test",
            hub_prefix="test/output",
            ttl="30m",
            run_command="bash /content/run-job.sh",
            sync_paths=("/content/out/logs",),
            run_env=run_env,
            output_capture=tmp_path / "output.log",
            hub_repository="example-owner/test-artifacts",
            download_inputs=False,
        )

    assert args.colab_accelerator == accelerator
    assert commands[-1][4:] == [
        "new",
        "--session",
        "accelerator-forwarding-test",
        "--gpu",
        accelerator,
    ]
