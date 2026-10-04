"""Execute one pinned CanvasVAE full-run seed on a named Colab session.

The queue supervisor invokes this script once per run. It verifies both the
local source and the remote checkout before starting the artifact-syncing
wrapper, then lets the bundled wrapper upload results and release the session.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypeAlias


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPOSITORY_ROOT / ".cache/canvas-vae/full-run/rico/manifest.json"
COLAB_COMMAND = ["mise", "exec", "--", "colab"]
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
HUB_REPOSITORY_PREFIX = "design-generators/canvas-vae/s5"
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


class ExecutorError(Exception):
    """A run precondition or Colab operation failed."""


def load_json(path: Path) -> JsonObject:
    try:
        value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot read JSON file {path}: {error}") from error

    if not isinstance(value, dict):
        raise ExecutorError(f"JSON file must contain an object: {path}")

    return value


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExecutorError(f"git {' '.join(args)} failed")

    return result.stdout.strip()


def verify_local_source(pinned_commit: str) -> None:
    if not COMMIT_PATTERN.fullmatch(pinned_commit):
        raise ExecutorError("source_commit must be a full 40-character commit")
    if git_output("rev-parse", "HEAD") != pinned_commit:
        raise ExecutorError("source gate failed: local HEAD differs from source_commit")
    if git_output("status", "--porcelain"):
        raise ExecutorError("source gate failed: local worktree is dirty")


def require_token_file() -> Path:
    token_value = os.environ.get("S5_HF_TOKEN_FILE")
    if token_value is None:
        raise ExecutorError("S5_HF_TOKEN_FILE must name a restricted token file")

    token_path = Path(token_value).expanduser().resolve()
    if not token_path.is_file() or stat.S_IMODE(token_path.stat().st_mode) != 0o600:
        raise ExecutorError("the Hugging Face token file must be mode 0600")
    if not token_path.read_text(encoding="utf-8").strip().startswith("hf_"):
        raise ExecutorError("the restricted Hugging Face token file is malformed")

    return token_path


def colab(
    *args: str,
    capture: bool = False,
    allow_nonzero: bool = False,
    output_path: Path | None = None,
    timeout: float = 120.0,
) -> str:
    command = [*COLAB_COMMAND, *args]
    if output_path is None:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
    else:
        with output_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
    if result.returncode != 0 and not allow_nonzero:
        raise ExecutorError(
            f"Colab command {args[0]} failed with exit code {result.returncode}"
        )

    return result.stdout.strip() if capture else ""


def assert_session_is_free(session_name: str) -> None:
    sessions = colab("sessions", capture=True)
    if re.search(rf"(?m)^\s*{re.escape(session_name)}(?:\s|$)", sessions):
        raise ExecutorError(
            f"named session {session_name!r} already exists; it was not touched"
        )


def remote_preflight_source() -> str:
    return """from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

config = json.loads(Path("/content/s5-preflight.json").read_text())
repository = Path("/content/repo")
if repository.exists():
    raise SystemExit("source gate failed: /content/repo already exists")
subprocess.run(
    ["git", "clone", "--depth=1", "--no-checkout", "--branch",
     config["source_branch"], config["source_url"], str(repository)],
    check=True,
)
subprocess.run(["git", "-C", str(repository), "checkout", "--detach",
                config["source_commit"]], check=True)
head = subprocess.run(
    ["git", "-C", str(repository), "rev-parse", "HEAD"],
    capture_output=True, check=True, text=True
).stdout.strip()
if head != config["source_commit"]:
    raise SystemExit(f"source gate failed: remote HEAD {head} differs from the pinned commit")
if subprocess.run(["git", "-C", str(repository), "status", "--porcelain"],
                  capture_output=True, check=True, text=True).stdout.strip():
    raise SystemExit("source gate failed: remote source checkout is dirty")
config_path = repository / config["training_config"]
if not config_path.is_file():
    raise SystemExit(f"source gate failed: training config is missing: {config_path}")
if config["config_sha256"] is not None:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if digest != config["config_sha256"]:
        raise SystemExit("source gate failed: training config hash differs from the manifest")
print(f"source_gate_passed commit={head}")

if config["download_inputs"]:
    token_path = Path("/content/s5-download-token")
    try:
        if token_path.stat().st_mode & 0o777 != 0o600:
            raise SystemExit("input token file must have mode 0600")
        environment = os.environ.copy()
        environment["HF_TOKEN"] = token_path.read_text().strip()
        stage = Path("/content/s5-inputs")
        subprocess.run(
            ["uvx", "--from", "huggingface_hub", "hf", "download",
             config["hub_repository"], "--repo-type", "model", "--revision",
             config["hub_revision"], "--include", config["hub_prefix"] + "/data/rico/**",
             "--include", config["hub_prefix"] + "/original/data/rico/**",
             "--local-dir", str(stage)],
            check=True, env=environment,
        )
        sources = (
            (stage / config["hub_prefix"] / "data" / "rico",
             repository / ".cache/canvas-vae/data/rico"),
            (stage / config["hub_prefix"] / "original" / "data" / "rico",
             repository / ".cache/canvas-vae/original/data/rico"),
        )
        for source, target in sources:
            if not source.is_dir() or not any(source.iterdir()):
                raise SystemExit(f"input preflight is empty: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        print("input_preflight_passed")
    finally:
        token_path.unlink(missing_ok=True)

if config["initialize_vendor_submodule"]:
    subprocess.run(["git", "-C", str(repository), "submodule", "update", "--init",
                    "--recursive"], check=True)
"""


def remote_failure_injection_source(source_commit: str) -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

root = Path("/content/repo")
module_path = root / "models/canvas-vae/scripts/s5_rico_seed_queue.py"
spec = importlib.util.spec_from_file_location("s5_rico_seed_queue", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("could not import the queue supervisor")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
calls_path = Path("/content/out/results/injected-failure-calls.json")
runner_path = Path("/content/injected-failure.py")
runner_path.write_text(
    "import json, sys\\nfrom pathlib import Path\\n"
    "p=Path('/content/out/results/injected-failure-calls.json')\\n"
    "calls=json.loads(p.read_text()) if p.exists() else []\\n"
    "calls.append(sys.argv[-1])\\n"
    "p.write_text(json.dumps(calls))\\nraise SystemExit(37)\\n",
    encoding="utf-8",
)
queue.source_gate = lambda run, pinned_commit, require_clean: None
runs = [
    {"run_id": "first", "system": "package", "seed": 0,
     "session_name": "cpu-smoke", "artifact_prefix": "first"},
    {"run_id": "second", "system": "package", "seed": 1,
     "session_name": "cpu-smoke", "artifact_prefix": "second"},
]
try:
    queue.run_queue({"dataset": "rico", "runs": runs}, "a" * 40,
                    [sys.executable, str(runner_path)], dry_run=False)
except queue.QueueError as error:
    if error.exit_code != 37:
        raise SystemExit(f"child failure propagated as {error.exit_code}")
else:
    raise SystemExit("injected child failure did not stop the queue")
if json.loads(calls_path.read_text()) != ["first"]:
    raise SystemExit("queue dispatched another run after the injected failure")
result = {
    "tiny_training_config": "models/canvas-vae/configs/training/smoke.yaml",
    "tiny_training_test": "test_lightning_fit_smoke",
    "tiny_training_exit_code": 0,
    "injected_child_exit_code": 37,
    "queue_exit_code": 37,
    "queue_stopped_before_second_run": True,
    "source_commit": "__SOURCE_COMMIT__",
}
Path("/content/out/results/cpu-executor-smoke.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\\n"
)
print("queue_failure_gate_passed exit_code=37")
""".replace("__SOURCE_COMMIT__", source_commit)


def temporary_files(
    temporary_directory: Path,
    manifest: JsonObject,
    run: JsonObject,
    pinned_commit: str,
    download_inputs: bool,
) -> tuple[Path, Path, Path]:
    input_data = manifest.get("input_data")
    if not isinstance(input_data, dict):
        raise ExecutorError("manifest input_data must be an object")
    plan = manifest.get("runtime_plan")
    if not isinstance(plan, dict):
        raise ExecutorError("manifest runtime_plan must be an object")

    training_config = manifest.get("config")
    source_branch = plan.get("source_branch")
    if not isinstance(training_config, str) or not isinstance(source_branch, str):
        raise ExecutorError("manifest must define its config and source branch")

    preflight_config: JsonObject = {
        "source_url": "https://github.com/creative-graphic-design/design-generators.git",
        "source_branch": source_branch,
        "source_commit": pinned_commit,
        "training_config": training_config,
        "config_sha256": manifest.get("config_sha256"),
        "download_inputs": download_inputs,
        "initialize_vendor_submodule": run.get("system") == "original",
        "hub_repository": input_data.get("repository"),
        "hub_revision": input_data.get("revision"),
        "hub_prefix": input_data.get("prefix"),
    }
    preflight_path = temporary_directory / "preflight.json"
    preflight_path.write_text(json.dumps(preflight_config, indent=2) + "\n")
    preflight_script = temporary_directory / "preflight.py"
    preflight_script.write_text(remote_preflight_source(), encoding="utf-8")
    return preflight_path, preflight_script, temporary_directory / "run-job.sh"


def build_run_job(
    manifest: JsonObject,
    run: JsonObject,
    pinned_commit: str,
    segment_id: str,
    resume_checkpoint: Path | None,
) -> str:
    run_id = run.get("run_id")
    system = run.get("system")
    seed = run.get("seed")
    config = manifest.get("config")
    evaluator = manifest.get("evaluator_command")
    if (
        not isinstance(run_id, str)
        or not isinstance(system, str)
        or not isinstance(seed, int)
    ):
        raise ExecutorError("run metadata must include run_id, system, and seed")
    if not isinstance(config, str) or not isinstance(evaluator, list):
        raise ExecutorError("manifest config and evaluator_command are required")
    output_template = run.get("runtime_output_root")
    if not isinstance(output_template, str):
        raise ExecutorError("run needs a runtime_output_root template")
    output_root = output_template.format(run_id=run_id, segment_id=segment_id)
    expected_output_root = f"/content/out/checkpoints/{run_id}/{segment_id}"
    if output_root != expected_output_root:
        raise ExecutorError(
            "run runtime_output_root must be unique under its run and segment"
        )
    training_root = f"{output_root}/training"
    converted_root = f"{output_root}/converted"
    evaluation_template = run.get("evaluation_output")
    if not isinstance(evaluation_template, str):
        raise ExecutorError("run needs a per-segment evaluation_output template")
    evaluation_output = evaluation_template.format(run_id=run_id, segment_id=segment_id)
    expected_evaluation_output = (
        f"/content/out/results/{run_id}/{segment_id}/evaluation.json"
    )
    if evaluation_output != expected_evaluation_output:
        raise ExecutorError(
            "run evaluation_output must be unique under its run and segment"
        )
    log_root = f"/content/out/logs/{run_id}/{segment_id}"
    result_root = f"/content/out/results/{run_id}/{segment_id}"
    env_lines = (
        "set -Eeuo pipefail",
        "cd /content/repo",
        f'test "$(git rev-parse HEAD)" = {shlex.quote(pinned_commit)}',
        'test -z "$(git status --porcelain)"',
        "unset MPLBACKEND",
        "unset UV_PROJECT_ENVIRONMENT",
        "export NVIDIA_TF32_OVERRIDE=0",
        "export TF_ENABLE_ONEDNN_OPTS=0",
        "export CUDA_VISIBLE_DEVICES=0",
        f"for output_path in {shlex.quote(output_root)} {shlex.quote(log_root)} {shlex.quote(result_root)}; do test ! -e \"$output_path\" || {{ echo 'run output exists; refusing overwrite' >&2; exit 78; }}; done",
        f"mkdir -p {shlex.quote(training_root)} {shlex.quote(converted_root)} {shlex.quote(log_root)} {shlex.quote(result_root)}",
    )

    if system == "package":
        uv_sync = "uv sync --frozen --package canvas-vae --extra training"
        training_args = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--package",
            "canvas-vae",
            "--extra",
            "training",
            "traingen",
            "fit",
            "--config",
            config,
            f"--seed_everything={seed}",
            "--trainer.devices=1",
            "--trainer.precision=32-true",
            f"--trainer.default_root_dir={training_root}",
        ]
        if resume_checkpoint is not None:
            training_args.append("--ckpt_path=/content/out/resume/checkpoint.ckpt")
        checkpoint_path = None
        converter_args = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--package",
            "canvas-vae",
            "--extra",
            "training",
            "models/canvas-vae/scripts/convert_original_checkpoint.py",
            "--vocabulary",
            "/content/repo/.cache/canvas-vae/data/rico/vocabulary.json",
            "--output-dir",
            converted_root,
        ]
        converter_setup = ""
    elif system == "original":
        if resume_checkpoint is not None:
            raise ExecutorError(
                "original TensorFlow runs do not support optimizer-state resume"
            )
        uv_sync = (
            "uv sync --frozen --package canvas-vae --extra training --extra vendor"
        )
        original_args = run.get("train_original_arguments")
        if not isinstance(original_args, list) or not all(
            isinstance(value, str) for value in original_args
        ):
            raise ExecutorError(
                "original run is missing resolved train_original.py arguments"
            )
        resolved = [
            value.format(seed=seed, output_root=output_root) for value in original_args
        ]
        training_args = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--package",
            "canvas-vae",
            "--extra",
            "training",
            "--extra",
            "vendor",
            "python",
            "models/canvas-vae/scripts/train_original.py",
            *resolved,
        ]
        training_root = f"{output_root}/original-training"
        env_lines += (f"mkdir -p {shlex.quote(training_root)}",)
        checkpoint_path = f"{training_root}/checkpoints/final.ckpt"
        converter_setup = (
            "export UV_PROJECT_ENVIRONMENT=/content/canvas-vae-convert-env\n"
            "uv sync --frozen --project /content/repo --package canvas-vae --extra convert\n"
        )
        converter_args = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            "/content/repo",
            "--package",
            "canvas-vae",
            "models/canvas-vae/scripts/convert_original_checkpoint.py",
            "--checkpoint",
            checkpoint_path,
            "--vocabulary",
            "/content/repo/.cache/canvas-vae/original/data/rico/vocabulary.json",
            "--output-dir",
            converted_root,
        ]
    else:
        raise ExecutorError(f"unsupported CanvasVAE run system: {system}")

    evaluator_args: list[str] = []
    for part in evaluator:
        if not isinstance(part, str):
            raise ExecutorError("evaluator_command elements must be strings")
        evaluator_args.append(
            part.format(
                run_id=run_id,
                converted_checkpoint=converted_root,
                evaluation_output=evaluation_output,
                repository="/content/repo",
                segment_id=segment_id,
            )
        )

    if system == "package":
        converter_command = (
            f'{shlex.join(converter_args[:9])} --checkpoint "$last_checkpoints" '
            f"{shlex.join(converter_args[9:])} > {log_root}/convert.log 2>&1"
        )
        checkpoint_hash_command = (
            f'sha256sum "$last_checkpoints" {shlex.quote(evaluation_output)} '
            f"> {result_root}/artifact-hashes.txt"
        )
        post_train_lines = [
            f"last_checkpoints=\"$(find {shlex.quote(training_root)} -path '*/checkpoints/last.ckpt' -type f -print)\"",
            'test "$(printf \'%s\\n\' "$last_checkpoints" | wc -l)" -eq 1',
            'test -s "$last_checkpoints"',
            converter_command,
        ]
    else:
        post_train_lines = [
            f"test -s {shlex.quote(checkpoint_path or '')}",
            converter_setup.rstrip(),
            f"{shlex.join(converter_args)} > {log_root}/convert.log 2>&1",
            "unset UV_PROJECT_ENVIRONMENT",
        ]
        checkpoint_hash_command = (
            f"sha256sum {shlex.quote(checkpoint_path or '')} "
            f"{shlex.quote(evaluation_output)} > {result_root}/artifact-hashes.txt"
        )

    command_lines = [
        *env_lines,
        uv_sync,
        f"{shlex.join(training_args)} > {log_root}/train.log 2>&1",
        *post_train_lines,
        f"{shlex.join(evaluator_args)} > {log_root}/evaluate.log 2>&1",
        f"test -s {shlex.quote(evaluation_output)}",
        checkpoint_hash_command,
    ]
    return "\n".join(line for line in command_lines if line) + "\n"


def make_wrapper_cell(
    job_name: str,
    hub_prefix: str,
    ttl: str,
    run_command: str,
) -> str:
    command = [
        "bash",
        "/content/colab-job.sh",
        "--job",
        job_name,
        "--hub-repo",
        "shunk031/colab-jobs",
        "--hub-prefix",
        hub_prefix,
        "--hub-token-file",
        "/content/s5-upload-token",
        "--sync-path",
        "/content/out/checkpoints",
        "--sync-path",
        "/content/out/logs",
        "--sync-path",
        "/content/out/results",
        "--sync-interval",
        "15m",
        "--ttl",
        ttl,
        "--env-file",
        "/content/s5-run.env",
        "--",
        *shlex.split(run_command),
    ]

    return "!" + shlex.join(command) + "\n"


def run_named_session(
    session_name: str,
    gpu: str | None,
    wrapper_path: Path,
    token_path: Path,
    preflight_path: Path,
    preflight_script: Path,
    run_script: Path,
    job_name: str,
    hub_prefix: str,
    ttl: str,
    run_command: str,
    run_env: Path,
    output_capture: Path,
    resume_checkpoint: Path | None = None,
    download_inputs: bool = True,
    additional_uploads: tuple[tuple[Path, str], ...] = (),
    allow_nonzero_job: bool = False,
) -> None:
    assert_session_is_free(session_name)
    create_args = ["new", "--session", session_name]
    if gpu is not None:
        create_args.extend(["--gpu", gpu])
    colab(*create_args, timeout=300)

    wrapper_started = False
    try:
        uploads = [
            (wrapper_path, "/content/colab-job.sh"),
            (preflight_path, "/content/s5-preflight.json"),
            (preflight_script, "/content/s5-preflight.py"),
            (run_script, "/content/run-job.sh"),
            (run_env, "/content/s5-run.env"),
            (token_path, "/content/s5-upload-token"),
            *additional_uploads,
        ]
        if download_inputs:
            uploads.append((token_path, "/content/s5-download-token"))
        for local_path, remote_path in uploads:
            colab("upload", "--session", session_name, str(local_path), remote_path)

        remote_preflight_cell = (
            "from pathlib import Path\n"
            "import subprocess\n"
            "restricted=['/content/s5-upload-token','/content/s5-run.env']\n"
            "if Path('/content/s5-download-token').exists(): restricted.append('/content/s5-download-token')\n"
            "subprocess.run(['chmod','600',*restricted],check=True)\n"
            "subprocess.run(['chmod', '700', '/content/colab-job.sh', "
            "'/content/run-job.sh'], check=True)\n"
            "Path('/content/out/resume').mkdir(parents=True, exist_ok=True)\n"
            "exec(compile(Path('/content/s5-preflight.py').read_text(), "
            "'/content/s5-preflight.py', 'exec'))\n"
        )
        cell_path = preflight_path.with_name("preflight-cell.py")
        cell_path.write_text(remote_preflight_cell, encoding="utf-8")
        colab(
            "exec",
            "--session",
            session_name,
            "--timeout",
            "1800",
            "-f",
            str(cell_path.resolve()),
            timeout=1900,
        )

        if resume_checkpoint is not None:
            colab(
                "upload",
                "--session",
                session_name,
                str(resume_checkpoint),
                "/content/out/resume/checkpoint.ckpt",
            )

        launch_cell = preflight_path.with_name("launch-cell.py")
        launch_cell.write_text(
            make_wrapper_cell(job_name, hub_prefix, ttl, run_command), encoding="utf-8"
        )
        wrapper_started = True
        colab(
            "exec",
            "--session",
            session_name,
            "--timeout",
            "7800",
            "-f",
            str(launch_cell.resolve()),
            output_path=output_capture,
            timeout=7920,
            allow_nonzero=allow_nonzero_job,
        )
    except (ExecutorError, OSError, subprocess.SubprocessError):
        if not wrapper_started:
            colab("stop", "--session", session_name, timeout=300)

        # A retained endpoint may still be the only result copy after a failed final upload.
        raise

    sessions = colab("sessions", capture=True)
    if re.search(rf"(?m)^\s*{re.escape(session_name)}(?:\s|$)", sessions):
        raise ExecutorError(
            f"wrapper did not release task-owned session {session_name!r}"
        )


def execute_run(
    manifest_path: Path,
    run_id: str,
    segment_id: str,
    resume_checkpoint: Path | None,
) -> None:
    manifest = load_json(manifest_path)
    pinned = manifest.get("source_commit")
    if not isinstance(pinned, str):
        raise ExecutorError("manifest source_commit must be finalized")
    verify_local_source(pinned)
    if manifest.get("queue_status") != "ready":
        raise ExecutorError("manifest queue_status is not ready; refusing B2 dispatch")
    config_digest = manifest.get("config_sha256")
    if not isinstance(config_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", config_digest
    ):
        raise ExecutorError("manifest config_sha256 must be a full SHA-256 digest")

    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list):
        raise ExecutorError("manifest runs must be an array")
    for field in (
        "run_id",
        "session_name",
        "artifact_prefix",
        "runtime_output_root",
        "evaluation_output",
    ):
        values = [
            candidate.get(field)
            for candidate in raw_runs
            if isinstance(candidate, dict)
        ]
        if (
            len(values) != len(raw_runs)
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ExecutorError(f"manifest runs need unique {field} values")
    matches = [
        run for run in raw_runs if isinstance(run, dict) and run.get("run_id") == run_id
    ]
    if len(matches) != 1 or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ExecutorError(f"manifest must define exactly one valid run_id {run_id!r}")
    run = matches[0]
    system = run.get("system")
    if system not in {"package", "original"}:
        raise ExecutorError("run system must be package or original")
    if resume_checkpoint is not None and system != "package":
        raise ExecutorError("original TensorFlow runs cannot resume optimizer state")
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise ExecutorError("resume requires an existing explicit checkpoint file")
    if not RUN_ID_PATTERN.fullmatch(segment_id):
        raise ExecutorError(
            "segment_id must use lowercase letters, digits, and hyphens"
        )
    if resume_checkpoint is not None and segment_id == "segment-001":
        raise ExecutorError(
            "resume requires a new segment_id to keep prior artifacts immutable"
        )

    runner = manifest.get("runner_command")
    if runner != ["python3", "models/canvas-vae/scripts/s5_rico_seed_executor.py"]:
        raise ExecutorError("manifest runner_command does not identify this executor")
    runtime_plan = manifest.get("runtime_plan")
    if not isinstance(runtime_plan, dict):
        raise ExecutorError("manifest runtime_plan must be an object")
    if (
        runtime_plan.get("gpu") != "A100"
        or runtime_plan.get("one_named_session_per_run") is not True
    ):
        raise ExecutorError("runtime must use one named A100 session per run")
    if runtime_plan.get("one_training_run_at_a_time") is not True:
        raise ExecutorError("runtime must execute only one training run at a time")
    if runtime_plan.get("queue_supervisor_sequential") is not True:
        raise ExecutorError("queue supervisor must execute runs sequentially")
    if (
        runtime_plan.get("self_release") is not True
        or runtime_plan.get("sync_interval") != "15m"
    ):
        raise ExecutorError("runtime must self-release and sync every 15 minutes")
    if (
        runtime_plan.get("wrapper_ttl") != "2h"
        or runtime_plan.get("wrapper_script") != "${S5_WRAPPER_PATH}"
    ):
        raise ExecutorError("runtime wrapper TTL must remain 2h")
    resume_policy = runtime_plan.get("resume_policy")
    if (
        not isinstance(resume_policy, dict)
        or resume_policy.get("explicit_download_required") is not True
        or resume_policy.get("downloaded_checkpoint_path")
        != "${S5_RUNTIME_ROOT}/out/resume/checkpoint.ckpt"
        or resume_policy.get("package_resume_flag")
        != "--ckpt_path=${S5_RUNTIME_ROOT}/out/resume/checkpoint.ckpt"
        or resume_policy.get("implicit_latest_checkpoint_resume") is not False
    ):
        raise ExecutorError(
            "runtime resume policy must require an explicit checkpoint path"
        )
    if runtime_plan.get("source_gate_before_artifact_sync") is not True:
        raise ExecutorError("source gate must pass before artifact synchronization")
    expected_sync_paths = [
        "${S5_RUNTIME_ROOT}/out/checkpoints",
        "${S5_RUNTIME_ROOT}/out/logs",
        "${S5_RUNTIME_ROOT}/out/results",
    ]
    if runtime_plan.get("sync_paths") != expected_sync_paths:
        raise ExecutorError("runtime must sync checkpoints, logs, and results")
    wrapper_value = os.environ.get("S5_WRAPPER_PATH")
    if wrapper_value is None:
        raise ExecutorError("S5_WRAPPER_PATH must name the installed bundled wrapper")
    wrapper_path = Path(wrapper_value).expanduser()
    if not wrapper_path.is_file():
        raise ExecutorError("installed bundled wrapper was not found")
    token_path = require_token_file()
    session_name = run.get("session_name")
    artifact_prefix = run.get("artifact_prefix")
    if not isinstance(session_name, str) or not isinstance(artifact_prefix, str):
        raise ExecutorError("run needs a named session and unique Hub artifact prefix")
    if not RUN_ID_PATTERN.fullmatch(session_name):
        raise ExecutorError(
            "run session_name must use lowercase letters, digits, and hyphens"
        )

    with tempfile.TemporaryDirectory(prefix="cvae-s5-run-") as temporary:
        temporary_directory = Path(temporary)
        preflight_path, preflight_script, run_path = temporary_files(
            temporary_directory, manifest, run, pinned, download_inputs=True
        )
        run_path.write_text(
            build_run_job(manifest, run, pinned, segment_id, resume_checkpoint),
            encoding="utf-8",
        )
        run_env = temporary_directory / "run.env"
        run_env.write_text(
            "CANVAS_VAE_S5_SOURCE_COMMIT="
            + pinned
            + "\n"
            + "CANVAS_VAE_S5_RUN_ID="
            + run_id
            + "\n",
            encoding="utf-8",
        )
        run_env.chmod(0o600)
        capture_path = (
            REPOSITORY_ROOT
            / ".cache/canvas-vae/full-run/rico/executor-logs"
            / f"{run_id}-{segment_id}.log"
        )
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        job_name = f"cvae-s5-{run_id}-{segment_id}"
        run_named_session(
            session_name,
            "A100",
            wrapper_path,
            token_path,
            preflight_path,
            preflight_script,
            run_path,
            job_name,
            f"{artifact_prefix}/{segment_id}",
            str(runtime_plan["wrapper_ttl"]),
            "bash /content/run-job.sh",
            run_env,
            capture_path,
            resume_checkpoint,
        )


def execute_cpu_smoke(session_name: str, source_commit: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(session_name):
        raise ExecutorError("CPU smoke requires a valid named session")
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ExecutorError("CPU smoke source must be a full commit SHA")
    has_source = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if has_source.returncode != 0:
        raise ExecutorError("CPU smoke source commit is not present locally")

    wrapper_value = os.environ.get("S5_WRAPPER_PATH")
    if wrapper_value is None:
        raise ExecutorError("S5_WRAPPER_PATH must name the installed bundled wrapper")
    wrapper_path = Path(wrapper_value).expanduser()
    if not wrapper_path.is_file():
        raise ExecutorError("installed bundled wrapper was not found")

    with tempfile.TemporaryDirectory(prefix="cvae-s5-cpu-smoke-") as temporary:
        temporary_directory = Path(temporary)
        config_path = "models/canvas-vae/configs/training/smoke.yaml"
        config_result = subprocess.run(
            ["git", "show", f"{source_commit}:{config_path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        smoke_manifest: JsonObject = {
            "config": config_path,
            "config_sha256": hashlib.sha256(config_result.stdout).hexdigest(),
            "input_data": {},
            "runtime_plan": {"source_branch": "feat/canvas-vae"},
        }
        smoke_run: JsonObject = {"system": "package"}
        preflight_path, preflight_script, run_path = temporary_files(
            temporary_directory,
            smoke_manifest,
            smoke_run,
            source_commit,
            download_inputs=False,
        )
        injected = temporary_directory / "injected-failure.py"
        injected.write_text(
            remote_failure_injection_source(source_commit), encoding="utf-8"
        )
        injected.chmod(0o700)
        run_path.write_text(
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            "cd /content/repo\n"
            f'test "$(git rev-parse HEAD)" = {shlex.quote(source_commit)}\n'
            'test -z "$(git status --porcelain)"\n'
            "unset MPLBACKEND\nexport NVIDIA_TF32_OVERRIDE=0\n"
            "export TF_ENABLE_ONEDNN_OPTS=0\nexport CUDA_VISIBLE_DEVICES=''\n"
            "uv run --frozen --package canvas-vae --extra training --with pytest pytest "
            "models/canvas-vae/tests/test_training.py::test_lightning_fit_smoke -m training\n"
            "mkdir -p /content/out/results\n"
            "chmod 700 /content/injected-failure.py\n"
            "/content/injected-failure.py\n",
            encoding="utf-8",
        )
        run_path.chmod(0o700)
        run_env = temporary_directory / "run.env"
        run_env.write_text(
            f"CANVAS_VAE_S5_SOURCE_COMMIT={source_commit}\n", encoding="utf-8"
        )
        run_env.chmod(0o600)
        capture_path = REPOSITORY_ROOT / ".cache/canvas-vae/s5" / f"{session_name}.log"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        run_named_session(
            session_name=session_name,
            gpu=None,
            wrapper_path=wrapper_path,
            token_path=require_token_file(),
            preflight_path=preflight_path,
            preflight_script=preflight_script,
            run_script=run_path,
            job_name=session_name,
            hub_prefix=f"{HUB_REPOSITORY_PREFIX}/{session_name}",
            ttl="2h",
            run_command="bash /content/run-job.sh",
            run_env=run_env,
            output_capture=capture_path,
            download_inputs=False,
            additional_uploads=((injected, "/content/injected-failure.py"),),
            allow_nonzero_job=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--segment-id", default="segment-001")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--session-name")
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.cpu_smoke:
            if args.session_name is None or args.source_commit is None:
                raise ExecutorError(
                    "CPU smoke requires --session-name and --source-commit"
                )
            execute_cpu_smoke(args.session_name, args.source_commit)
        else:
            if args.run_id is None:
                raise ExecutorError("--run-id is required for a queue run")
            execute_run(
                args.manifest.resolve(),
                args.run_id,
                args.segment_id,
                args.resume_checkpoint,
            )
    except (ExecutorError, OSError, subprocess.SubprocessError) as error:
        print(f"Full-run seed failed closed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
