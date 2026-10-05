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

checkpoint_download = config.get("checkpoint_download")
if checkpoint_download is not None:
    if not isinstance(checkpoint_download, dict):
        raise SystemExit("checkpoint download settings must be an object")
    token_path = Path("/content/s5-download-token")
    try:
        if token_path.stat().st_mode & 0o777 != 0o600:
            raise SystemExit("checkpoint download token file must have mode 0600")
        environment = os.environ.copy()
        environment["HF_TOKEN"] = token_path.read_text().strip()
        stage = Path("/content/s5-training-artifacts")
        checkpoint_glob = checkpoint_download["hub_glob"]
        checkpoint_prefix = checkpoint_download["hub_prefix"]
        subprocess.run(
            ["uvx", "--from", "huggingface_hub", "hf", "download",
             checkpoint_download["hub_repository"], "--repo-type", "model",
             "--revision", checkpoint_download["revision"], "--include",
             checkpoint_glob, "--local-dir", str(stage)],
            check=True, env=environment,
        )
        source_prefix = stage / checkpoint_prefix
        destination_prefix = Path(checkpoint_download["local_checkpoint"])
        if destination_prefix.exists():
            raise SystemExit(f"refusing to overwrite downloaded checkpoint: {destination_prefix}")
        destination_prefix.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_download["system"] == "package":
            if not source_prefix.is_file():
                raise SystemExit(f"package checkpoint is missing: {source_prefix}")
            shutil.copy2(source_prefix, destination_prefix)
        else:
            sidecars = sorted(source_prefix.parent.glob(source_prefix.name + ".*"))
            if not any(path.suffix == ".index" for path in sidecars):
                raise SystemExit(f"TensorFlow checkpoint index is missing: {source_prefix}")
            for sidecar in sidecars:
                shutil.copy2(
                    sidecar,
                    destination_prefix.with_name(destination_prefix.name + sidecar.name[len(source_prefix.name):]),
                )
        print(f"checkpoint_download_passed path={destination_prefix}")
    finally:
        token_path.unlink(missing_ok=True)
else:
    Path("/content/s5-download-token").unlink(missing_ok=True)

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

module_path = Path("/content/repo/models/canvas-vae/scripts/s5_rico_seed_queue.py")
module_path.write_bytes(Path("/content/s5_rico_seed_queue.py").read_bytes())
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
     "session_name": "cpu-smoke-a100", "cpu_session_name": "cpu-smoke-cpu",
     "artifact_prefix": "first"},
    {"run_id": "second", "system": "package", "seed": 1,
     "session_name": "cpu-smoke-a100-2", "cpu_session_name": "cpu-smoke-cpu-2",
     "artifact_prefix": "second"},
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
    "tiny_training_entrypoint": "traingen fit --config /content/s5-smoke.yaml",
    "tiny_training_exit_code": 0,
    "tiny_conversion_entrypoint": "convert_original_checkpoint.py --checkpoint /content/s5-training-checkpoint/epoch-500.ckpt",
    "tiny_conversion_exit_code": 0,
    "tiny_evaluation_entrypoint": "evaluate.py --checkpoint /content/out/converted/cpu-smoke-package/smoke-001 --output /content/out/results/cpu-smoke-package/smoke-001/evaluation.json",
    "tiny_evaluation_exit_code": 0,
    "per_run_evaluation_output": "/content/out/results/cpu-smoke-package/smoke-001/evaluation.json",
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
    checkpoint_download: JsonObject | None = None,
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
        "initialize_vendor_submodule": run.get("system") == "original"
        and run.get("cpu_postprocess") is not True,
        "hub_repository": input_data.get("repository"),
        "hub_revision": input_data.get("revision"),
        "hub_prefix": input_data.get("prefix"),
        "checkpoint_download": checkpoint_download,
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
    if (
        not isinstance(run_id, str)
        or not isinstance(system, str)
        or not isinstance(seed, int)
    ):
        raise ExecutorError("run metadata must include run_id, system, and seed")
    if not isinstance(config, str):
        raise ExecutorError("manifest config is required")
    training_config = run.get("training_config_override", config)
    if not isinstance(training_config, str):
        raise ExecutorError("training_config_override must be a path string")
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
    log_root = f"/content/out/logs/{run_id}/{segment_id}"
    epoch_500_checkpoint = f"{output_root}/epoch-500.ckpt"
    env_lines = (
        "set -Eeuo pipefail",
        "cd /content/repo",
        f'test "$(git rev-parse HEAD)" = {shlex.quote(pinned_commit)}',
        'test -z "$(git status --porcelain)"',
        "unset MPLBACKEND",
        "unset UV_PROJECT_ENVIRONMENT",
        "export NVIDIA_TF32_OVERRIDE=0",
        "export TF_ENABLE_ONEDNN_OPTS=0",
        "export CUDA_VISIBLE_DEVICES=''"
        if run.get("cpu_smoke") is True
        else "export CUDA_VISIBLE_DEVICES=0",
        f"for output_path in {shlex.quote(output_root)} {shlex.quote(log_root)}; do test ! -e \"$output_path\" || {{ echo 'run output exists; refusing overwrite' >&2; exit 78; }}; done",
        f"mkdir -p {shlex.quote(training_root)} {shlex.quote(log_root)}",
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
            training_config,
            f"--seed_everything={seed}",
            "--trainer.devices=1",
            "--trainer.precision=32-true",
            f"--trainer.default_root_dir={training_root}",
        ]
        if resume_checkpoint is not None:
            training_args.append("--ckpt_path=/content/out/resume/checkpoint.ckpt")
        checkpoint_path = None
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
    else:
        raise ExecutorError(f"unsupported CanvasVAE run system: {system}")

    if system == "package":
        post_train_lines = [
            f"last_checkpoints=\"$(find {shlex.quote(training_root)} -path '*/checkpoints/last.ckpt' -type f -print)\"",
            'test "$(printf \'%s\\n\' "$last_checkpoints" | wc -l)" -eq 1',
            'test -s "$last_checkpoints"',
            f'cp -- "$last_checkpoints" {shlex.quote(epoch_500_checkpoint)}',
        ]
    else:
        post_train_lines = [
            f"checkpoint_prefix={shlex.quote(checkpoint_path or '')}",
            'test -s "${checkpoint_prefix}.index"',
            'for part in "${checkpoint_prefix}".*; do test -f "$part" || continue; suffix="${part#"${checkpoint_prefix}"}"; cp -- "$part" "'
            + epoch_500_checkpoint
            + '${suffix}"; done',
            f"test -s {shlex.quote(epoch_500_checkpoint + '.index')}",
        ]
    if run.get("cpu_smoke") is True:
        training_args.append("--trainer.enable_checkpointing=true")
    training_overrides = run.get("training_overrides", [])
    if not isinstance(training_overrides, list) or not all(
        isinstance(value, str) for value in training_overrides
    ):
        raise ExecutorError("training_overrides must be a list of strings")
    training_args.extend(training_overrides)

    command_lines = [
        *env_lines,
        uv_sync,
        f"{shlex.join(training_args)} > {log_root}/train.log 2>&1",
        *post_train_lines,
    ]
    return "\n".join(line for line in command_lines if line) + "\n"


def build_postprocess_job(
    manifest: JsonObject,
    run: JsonObject,
    segment_id: str,
) -> str:
    run_id = run.get("run_id")
    system = run.get("system")
    evaluator = run.get("evaluator_command", manifest.get("evaluator_command"))
    if not isinstance(run_id, str) or system not in {"package", "original"}:
        raise ExecutorError("postprocess run metadata is invalid")
    if not isinstance(evaluator, list):
        raise ExecutorError("manifest evaluator_command is required")
    converted_template = run.get("converted_output_root")
    evaluation_template = run.get("evaluation_output")
    checkpoint_template = run.get("epoch_500_checkpoint")
    if (
        not isinstance(converted_template, str)
        or not isinstance(evaluation_template, str)
        or not isinstance(checkpoint_template, str)
    ):
        raise ExecutorError(
            "run needs checkpoint, converted, and evaluation output templates"
        )
    converted_root = converted_template.format(run_id=run_id, segment_id=segment_id)
    evaluation_output = evaluation_template.format(run_id=run_id, segment_id=segment_id)
    checkpoint_input = "/content/s5-training-checkpoint/epoch-500.ckpt"
    if converted_root != f"/content/out/converted/{run_id}/{segment_id}":
        raise ExecutorError(
            "run converted_output_root must be unique under its run and segment"
        )
    if (
        evaluation_output
        != f"/content/out/results/{run_id}/{segment_id}/evaluation.json"
    ):
        raise ExecutorError(
            "run evaluation_output must be unique under its run and segment"
        )
    if (
        checkpoint_template.format(run_id=run_id, segment_id=segment_id)
        != f"/content/out/checkpoints/{run_id}/{segment_id}/epoch-500.ckpt"
    ):
        raise ExecutorError(
            "run epoch_500_checkpoint must be unique under its run and segment"
        )
    log_root = f"/content/out/logs/{run_id}/{segment_id}"
    result_root = f"/content/out/results/{run_id}/{segment_id}"
    vocabulary = "/content/repo/.cache/canvas-vae/data/rico/vocabulary.json"
    if system == "original":
        vocabulary = (
            "/content/repo/.cache/canvas-vae/original/data/rico/vocabulary.json"
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
    ]
    if system == "package":
        converter_args.extend(["--extra", "training"])
    else:
        converter_args.extend(["--extra", "convert"])
    converter_args.extend(
        [
            "models/canvas-vae/scripts/convert_original_checkpoint.py",
            "--checkpoint",
            checkpoint_input,
            "--vocabulary",
            vocabulary,
            "--output-dir",
            converted_root,
        ]
    )
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
    checkpoint_for_hash = checkpoint_input
    if system == "original":
        checkpoint_for_hash += ".index"
    uv_setup = "uv sync --frozen --project /content/repo --package canvas-vae --extra training\n"
    if system == "original":
        uv_setup += (
            "export UV_PROJECT_ENVIRONMENT=/content/canvas-vae-convert-env\n"
            "uv sync --frozen --project /content/repo --package canvas-vae --extra convert\n"
        )
    return (
        "\n".join(
            [
                "set -Eeuo pipefail",
                "cd /content/repo",
                'test -z "$(git status --porcelain)"',
                "unset MPLBACKEND UV_PROJECT_ENVIRONMENT",
                "export NVIDIA_TF32_OVERRIDE=0",
                "export TF_ENABLE_ONEDNN_OPTS=0",
                "export CUDA_VISIBLE_DEVICES=''",
                f"for output_path in {shlex.quote(converted_root)} {shlex.quote(result_root)} {shlex.quote(log_root)}; do test ! -e \"$output_path\" || {{ echo 'postprocess output exists; refusing overwrite' >&2; exit 78; }}; done",
                f"mkdir -p {shlex.quote(converted_root)} {shlex.quote(result_root)} {shlex.quote(log_root)}",
                f"test -s {shlex.quote(checkpoint_for_hash)}",
                uv_setup.rstrip(),
                f"{shlex.join(converter_args)} 2>&1 | tee {shlex.quote(log_root + '/convert.log')}",
                "unset UV_PROJECT_ENVIRONMENT",
                f"{shlex.join(evaluator_args)} 2>&1 | tee {shlex.quote(log_root + '/evaluate.log')}",
                f"test -s {shlex.quote(evaluation_output)}",
                f"sha256sum {shlex.quote(checkpoint_for_hash)} {shlex.quote(evaluation_output)} > {shlex.quote(result_root + '/artifact-hashes.txt')}",
            ]
        )
        + "\n"
    )


def make_wrapper_cell(
    job_name: str,
    hub_prefix: str,
    ttl: str,
    run_command: str,
    sync_paths: list[str],
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
        *[part for path in sync_paths for part in ("--sync-path", path)],
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
    sync_paths: tuple[str, ...],
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
            make_wrapper_cell(job_name, hub_prefix, ttl, run_command, list(sync_paths)),
            encoding="utf-8",
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
        "cpu_session_name",
        "artifact_prefix",
        "runtime_output_root",
        "epoch_500_checkpoint",
        "epoch_500_checkpoint_hub_path",
        "converted_output_root",
        "evaluation_output",
        "training_artifact_prefix",
        "postprocess_artifact_prefix",
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
    session_names = [
        session_name
        for candidate in raw_runs
        if isinstance(candidate, dict)
        for session_name in (
            candidate.get("session_name"),
            candidate.get("cpu_session_name"),
        )
        if isinstance(session_name, str)
    ]
    if len(session_names) != 2 * len(raw_runs) or len(session_names) != len(
        set(session_names)
    ):
        raise ExecutorError(
            "manifest A100 and CPU session names must be globally unique"
        )
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
    if runtime_plan.get("source_commit") != pinned:
        raise ExecutorError(
            "runtime_plan source_commit must match the manifest source_commit"
        )
    if (
        runtime_plan.get("gpu") != "A100"
        or runtime_plan.get("one_named_a100_session_per_run") is not True
        or runtime_plan.get("one_named_cpu_session_per_run") is not True
    ):
        raise ExecutorError("runtime must name one A100 and one CPU session per run")
    if runtime_plan.get("one_training_run_at_a_time") is not True:
        raise ExecutorError("runtime must execute only one training run at a time")
    if runtime_plan.get("queue_supervisor_sequential") is not True:
        raise ExecutorError("queue supervisor must execute runs sequentially")
    if (
        runtime_plan.get("self_release") is not True
        or runtime_plan.get("sync_interval") != "15m"
    ):
        raise ExecutorError("runtime must self-release and sync every 15 minutes")
    a100_ttls = runtime_plan.get("a100_wrapper_ttl_by_system")
    if (
        not isinstance(a100_ttls, dict)
        or not all(
            isinstance(a100_ttls.get(system_name), str)
            and re.fullmatch(r"[1-9][0-9]*m", a100_ttls[system_name])
            for system_name in ("package", "original")
        )
        or runtime_plan.get("cpu_wrapper_ttl") != "38m"
        or runtime_plan.get("wrapper_script") != "${S5_WRAPPER_PATH}"
    ):
        raise ExecutorError(
            "runtime wrapper TTLs must define bounded A100 and CPU jobs"
        )
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
    expected_a100_sync_paths = [
        "${S5_RUNTIME_ROOT}/out/checkpoints",
        "${S5_RUNTIME_ROOT}/out/logs",
    ]
    expected_cpu_sync_paths = [
        "${S5_RUNTIME_ROOT}/out/converted",
        "${S5_RUNTIME_ROOT}/out/results",
        "${S5_RUNTIME_ROOT}/out/logs",
    ]
    if runtime_plan.get("a100_sync_paths") != expected_a100_sync_paths:
        raise ExecutorError("A100 runtime must sync only checkpoints and logs")
    if runtime_plan.get("cpu_sync_paths") != expected_cpu_sync_paths:
        raise ExecutorError(
            "CPU runtime must sync converted weights, results, and logs"
        )
    wrapper_value = os.environ.get("S5_WRAPPER_PATH")
    if wrapper_value is None:
        raise ExecutorError("S5_WRAPPER_PATH must name the installed bundled wrapper")
    wrapper_path = Path(wrapper_value).expanduser()
    if not wrapper_path.is_file():
        raise ExecutorError("installed bundled wrapper was not found")
    token_path = require_token_file()
    session_name = run.get("session_name")
    cpu_session_name = run.get("cpu_session_name")
    artifact_prefix = run.get("artifact_prefix")
    if not all(
        isinstance(value, str)
        for value in (session_name, cpu_session_name, artifact_prefix)
    ):
        raise ExecutorError(
            "run needs named A100 and CPU sessions and a unique Hub artifact prefix"
        )
    if not RUN_ID_PATTERN.fullmatch(session_name) or not RUN_ID_PATTERN.fullmatch(
        cpu_session_name
    ):
        raise ExecutorError(
            "runtime session names must use lowercase letters, digits, and hyphens"
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
        training_prefix = run.get("training_artifact_prefix")
        postprocess_prefix = run.get("postprocess_artifact_prefix")
        if not isinstance(training_prefix, str) or not isinstance(
            postprocess_prefix, str
        ):
            raise ExecutorError(
                "run needs separate training and postprocess Hub prefixes"
            )
        training_prefix = training_prefix.format(run_id=run_id, segment_id=segment_id)
        postprocess_prefix = postprocess_prefix.format(
            run_id=run_id, segment_id=segment_id
        )
        expected_training_prefix = f"{artifact_prefix}/train/{segment_id}"
        expected_postprocess_prefix = f"{artifact_prefix}/postprocess/{segment_id}"
        if (
            training_prefix != expected_training_prefix
            or postprocess_prefix != expected_postprocess_prefix
        ):
            raise ExecutorError(
                "training and postprocess Hub prefixes must be unique per run and segment"
            )
        checkpoint_template = run.get("epoch_500_checkpoint")
        checkpoint_hub_template = run.get("epoch_500_checkpoint_hub_path")
        if not isinstance(checkpoint_template, str) or not isinstance(
            checkpoint_hub_template, str
        ):
            raise ExecutorError("run needs an epoch-500 checkpoint path and Hub path")
        checkpoint_hub_path = checkpoint_hub_template.format(
            run_id=run_id, segment_id=segment_id
        )
        expected_checkpoint_hub_path = (
            f"{training_prefix}/checkpoints/{run_id}/{segment_id}/epoch-500.ckpt"
        )
        if checkpoint_hub_path != expected_checkpoint_hub_path:
            raise ExecutorError(
                "checkpoint Hub path must name this run's epoch-500 artifact"
            )
        checkpoint_download: JsonObject = {
            "hub_repository": "shunk031/colab-jobs",
            "revision": "main",
            "hub_prefix": checkpoint_hub_path,
            "hub_glob": checkpoint_hub_path
            if system == "package"
            else checkpoint_hub_path + ".*",
            "local_checkpoint": "/content/s5-training-checkpoint/epoch-500.ckpt",
            "system": system,
        }
        cpu_run = dict(run)
        cpu_run["cpu_postprocess"] = True
        cpu_preflight_path, cpu_preflight_script, cpu_run_path = temporary_files(
            temporary_directory,
            manifest,
            cpu_run,
            pinned,
            download_inputs=True,
            checkpoint_download=checkpoint_download,
        )
        cpu_run_path.write_text(
            build_postprocess_job(manifest, run, segment_id), encoding="utf-8"
        )
        cpu_env = temporary_directory / "cpu-run.env"
        cpu_env.write_text(
            "CANVAS_VAE_S5_SOURCE_COMMIT="
            + pinned
            + "\nCANVAS_VAE_S5_RUN_ID="
            + run_id
            + "\n",
            encoding="utf-8",
        )
        cpu_env.chmod(0o600)
        cpu_capture_path = capture_path.with_name(f"{run_id}-{segment_id}-cpu.log")
        expected_a100_ttl = a100_ttls.get(system)
        a100_ttl = run.get("a100_wrapper_ttl")
        cpu_ttl = run.get("cpu_wrapper_ttl")
        if (
            not isinstance(expected_a100_ttl, str)
            or not isinstance(a100_ttl, str)
            or not isinstance(cpu_ttl, str)
            or a100_ttl != expected_a100_ttl
            or cpu_ttl != runtime_plan.get("cpu_wrapper_ttl")
        ):
            raise ExecutorError(
                "per-run wrapper TTLs must match the validated runtime plan"
            )
        run_named_session(
            session_name,
            "A100",
            wrapper_path,
            token_path,
            preflight_path,
            preflight_script,
            run_path,
            job_name,
            training_prefix,
            a100_ttl,
            "bash /content/run-job.sh",
            (
                "/content/out/checkpoints",
                "/content/out/logs",
            ),
            run_env,
            capture_path,
            resume_checkpoint,
        )
        run_named_session(
            cpu_session_name,
            None,
            wrapper_path,
            token_path,
            cpu_preflight_path,
            cpu_preflight_script,
            cpu_run_path,
            f"{job_name}-postprocess",
            postprocess_prefix,
            str(cpu_ttl),
            "bash /content/run-job.sh",
            (
                "/content/out/converted",
                "/content/out/results",
                "/content/out/logs",
            ),
            cpu_env,
            cpu_capture_path,
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
            "evaluator_command": [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "--package",
                "canvas-vae",
                "--extra",
                "training",
                "models/canvas-vae/scripts/evaluate.py",
                "--checkpoint",
                "{converted_checkpoint}",
                "--data-dir",
                "/content/repo/.cache/canvas-vae/data/rico",
                "--seeds",
                "0",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--output",
                "{evaluation_output}",
            ],
            "runtime_plan": {"source_branch": "feat/canvas-vae"},
        }
        smoke_run: JsonObject = {
            "run_id": "cpu-smoke-package",
            "system": "package",
            "seed": 0,
            "runtime_output_root": "/content/out/checkpoints/cpu-smoke-package/{segment_id}",
            "epoch_500_checkpoint": "/content/out/checkpoints/cpu-smoke-package/{segment_id}/epoch-500.ckpt",
            "converted_output_root": "/content/out/converted/cpu-smoke-package/{segment_id}",
            "evaluation_output": "/content/out/results/cpu-smoke-package/{segment_id}/evaluation.json",
            "cpu_smoke": True,
            "training_config_override": "/content/s5-smoke.yaml",
            "training_overrides": [
                "--model.init_args.data_dir=/content/repo/.cache/canvas-vae/data/rico",
                "--data.init_args.data_dir=/content/repo/.cache/canvas-vae/data/rico",
            ],
        }
        training_plan = build_run_job(
            smoke_manifest, smoke_run, source_commit, "smoke-001", None
        )
        postprocess_plan = build_postprocess_job(smoke_manifest, smoke_run, "smoke-001")
        if any(
            token in training_plan
            for token in (
                "convert_original_checkpoint.py",
                "evaluate.py",
                "/out/results/",
            )
        ):
            raise ExecutorError(
                "A100 training plan must not convert or evaluate checkpoints"
            )
        if "last.ckpt" not in training_plan or "epoch-500.ckpt" not in training_plan:
            raise ExecutorError(
                "training plan does not upload its explicit epoch-500 checkpoint"
            )
        if (
            "convert_original_checkpoint.py" not in postprocess_plan
            or "evaluate.py" not in postprocess_plan
        ):
            raise ExecutorError(
                "CPU postprocess plan must convert and evaluate the checkpoint"
            )
        if "/content/s5-training-checkpoint/epoch-500.ckpt" not in postprocess_plan:
            raise ExecutorError(
                "CPU postprocess plan must use the downloaded explicit checkpoint path"
            )
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
        queue_script = temporary_directory / "s5_rico_seed_queue.py"
        queue_script.write_bytes(
            (
                REPOSITORY_ROOT / "models/canvas-vae/scripts/s5_rico_seed_queue.py"
            ).read_bytes()
        )
        training_path = temporary_directory / "training-job.sh"
        training_path.write_text(training_plan, encoding="utf-8")
        training_path.chmod(0o700)
        postprocess_path = temporary_directory / "postprocess-job.sh"
        postprocess_path.write_text(postprocess_plan, encoding="utf-8")
        postprocess_path.chmod(0o700)
        smoke_data = temporary_directory / "create-smoke-data.py"
        smoke_data.write_text(
            "from importlib.util import module_from_spec, spec_from_file_location\n"
            "from pathlib import Path\n"
            "from canvas_vae.processing_canvas_vae import prepare_rico_cache\n"
            "path=Path('/content/repo/models/canvas-vae/tests/conftest.py')\n"
            "spec=spec_from_file_location('canvas_vae_test_fixtures',path)\n"
            "assert spec is not None and spec.loader is not None\n"
            "fixtures=module_from_spec(spec)\n"
            "spec.loader.exec_module(fixtures)\n"
            "archive=fixtures.write_archive(Path('/content/s5-smoke-rico.zip'),fixtures.synthetic_screens())\n"
            "prepare_rico_cache(archive,Path('/content/repo/.cache/canvas-vae/data/rico'))\n",
            encoding="utf-8",
        )
        smoke_config = temporary_directory / "smoke.yaml"
        smoke_config.write_text(
            "seed_everything: 0\n"
            "trainer:\n"
            "  accelerator: cpu\n"
            "  devices: 1\n"
            "  precision: 32-true\n"
            "  max_epochs: 1\n"
            "  limit_train_batches: 2\n"
            "  check_val_every_n_epoch: 1\n"
            "  num_sanity_val_steps: 0\n"
            "  enable_checkpointing: true\n"
            "  default_root_dir: /content/out/checkpoints/cpu-smoke-package/smoke-001/training\n"
            "  callbacks:\n"
            "    - class_path: lightning.pytorch.callbacks.ModelCheckpoint\n"
            "      init_args:\n"
            "        dirpath: /content/out/checkpoints/cpu-smoke-package/smoke-001/training/checkpoints\n"
            "        save_top_k: 0\n"
            "        save_last: true\n"
            "model:\n"
            "  class_path: canvas_vae.training.CanvasVAETrainingModule\n"
            "  init_args:\n"
            "    data_dir: /content/repo/.cache/canvas-vae/data/rico\n"
            "    latent_dim: 256\n"
            "    num_heads: 8\n"
            "data:\n"
            "  class_path: canvas_vae.training.CanvasVAEDataModule\n"
            "  init_args:\n"
            "    data_dir: /content/repo/.cache/canvas-vae/data/rico\n"
            "    batch_size: 2\n"
            "    num_workers: 0\n",
            encoding="utf-8",
        )
        run_path.write_text(
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            "cd /content/repo\n"
            f'test "$(git rev-parse HEAD)" = {shlex.quote(source_commit)}\n'
            'test -z "$(git status --porcelain)"\n'
            "unset MPLBACKEND\nexport NVIDIA_TF32_OVERRIDE=0\n"
            "export TF_ENABLE_ONEDNN_OPTS=0\nexport CUDA_VISIBLE_DEVICES=''\n"
            "echo cpu_smoke_stage=data_preparation_started\n"
            "uv run --project /content/repo --frozen --package canvas-vae --extra training --with pytest /content/create-smoke-data.py\n"
            "echo cpu_smoke_stage=training_started\n"
            "bash /content/training-job.sh\n"
            "echo cpu_smoke_stage=training_completed\n"
            "mkdir -p /content/s5-training-checkpoint\n"
            "cp -- /content/out/checkpoints/cpu-smoke-package/smoke-001/epoch-500.ckpt /content/s5-training-checkpoint/epoch-500.ckpt\n"
            "test ! -e /content/smoke-a100-stage\n"
            "mkdir -p /content/smoke-a100-stage\n"
            "mv -- /content/out/logs/cpu-smoke-package/smoke-001 /content/smoke-a100-stage/logs\n"
            "echo cpu_smoke_stage=postprocess_started\n"
            "bash /content/postprocess-job.sh\n"
            "echo cpu_smoke_stage=postprocess_completed\n"
            "chmod 700 /content/injected-failure.py\n"
            "echo cpu_smoke_stage=injected_failure_started\n"
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
            ttl="30m",
            run_command="bash /content/run-job.sh",
            run_env=run_env,
            output_capture=capture_path,
            download_inputs=False,
            additional_uploads=(
                (injected, "/content/injected-failure.py"),
                (training_path, "/content/training-job.sh"),
                (postprocess_path, "/content/postprocess-job.sh"),
                (smoke_data, "/content/create-smoke-data.py"),
                (queue_script, "/content/s5_rico_seed_queue.py"),
                (smoke_config, "/content/s5-smoke.yaml"),
            ),
            sync_paths=(
                "/content/out/checkpoints",
                "/content/out/converted",
                "/content/out/results",
                "/content/out/logs",
                "/content/smoke-a100-stage",
            ),
        )
        execution_log = capture_path.read_text(encoding="utf-8")
        for required_log in (
            "queue_failure_gate_passed exit_code=37",
            "finished with exit code 0",
            "results uploaded to shunk031/colab-jobs/",
            "releasing the VM",
        ):
            if required_log not in execution_log:
                raise ExecutorError(
                    f"CPU smoke wrapper log is missing expected evidence: {required_log}"
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
