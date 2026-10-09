"""Execute one pinned CanvasVAE full-run seed on a named Colab session.

The queue supervisor invokes this script once per run. It verifies both the
local source and the remote checkout before starting the artifact-syncing
wrapper, then lets the bundled wrapper upload results and release the session.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import uuid
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
    if result.returncode != 0:
        raise ExecutorError(
            f"Colab command {args[0]} failed with exit code {result.returncode}"
        )

    return result.stdout.strip() if capture else ""


def hub_python(
    token_path: Path,
    source: str,
    temporary_directory: Path,
    extra_environment: dict[str, str] | None = None,
) -> str:
    environment = os.environ.copy()
    if extra_environment is not None:
        environment.update(extra_environment)
    environment["S5_HF_TOKEN_FILE"] = str(token_path)
    environment["S5_HF_TEMP_DIR"] = str(temporary_directory)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "huggingface_hub",
            "--",
            "python",
            "-c",
            source,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise ExecutorError("private Hub verification failed: " + result.stderr.strip())

    return result.stdout.strip()


def verify_checkpoint_on_hub(
    token_path: Path,
    checkpoint_hub_path: str,
    system: str,
) -> None:
    source = "\n".join(
        [
            "import json, os",
            "from pathlib import Path",
            "from huggingface_hub import HfApi",
            "token = Path(os.environ['S5_HF_TOKEN_FILE']).read_text().strip()",
            "files = HfApi(token=token).list_repo_files('shunk031/colab-jobs', repo_type='model')",
            "prefix = os.environ['S5_CHECKPOINT_HUB_PATH']",
            "print(json.dumps([path for path in files if path == prefix or path.startswith(prefix + '.')]))",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="cvae-s5-hub-checkpoint-") as temporary:
        output = hub_python(
            token_path,
            source,
            Path(temporary),
            {"S5_CHECKPOINT_HUB_PATH": checkpoint_hub_path},
        )
    try:
        files = json.loads(output)
    except json.JSONDecodeError as error:
        raise ExecutorError(
            "private Hub checkpoint listing was not valid JSON"
        ) from error

    if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
        raise ExecutorError("private Hub checkpoint listing returned invalid paths")
    if system == "package":
        present = checkpoint_hub_path in files
    else:
        present = checkpoint_hub_path + ".index" in files and any(
            path.startswith(checkpoint_hub_path + ".data-") for path in files
        )
    if not present:
        raise ExecutorError(
            f"epoch-500 {system} checkpoint is missing from the private Hub: {checkpoint_hub_path}"
        )


def verify_hub_paths(
    token_path: Path,
    expected_paths: list[str],
    description: str,
) -> None:
    source = "\n".join(
        [
            "import json, os",
            "from pathlib import Path",
            "from huggingface_hub import HfApi",
            "token = Path(os.environ['S5_HF_TOKEN_FILE']).read_text().strip()",
            "files = HfApi(token=token).list_repo_files('shunk031/colab-jobs', repo_type='model')",
            "expected = json.loads(os.environ['S5_EXPECTED_HUB_PATHS'])",
            "print(json.dumps([path for path in expected if path in files]))",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="cvae-s5-hub-artifacts-") as temporary:
        output = hub_python(
            token_path,
            source,
            Path(temporary),
            {"S5_EXPECTED_HUB_PATHS": json.dumps(expected_paths)},
        )
    try:
        present_paths = json.loads(output)
    except json.JSONDecodeError as error:
        raise ExecutorError(
            f"private Hub {description} listing was not valid JSON"
        ) from error
    if present_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(present_paths))
        raise ExecutorError(f"private Hub {description} is missing: {missing}")


def upload_smoke_result(token_path: Path, result_path: Path, hub_path: str) -> None:
    source = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "from huggingface_hub import HfApi",
            "token = Path(os.environ['S5_HF_TOKEN_FILE']).read_text().strip()",
            "HfApi(token=token).upload_file(path_or_fileobj=os.environ['S5_SMOKE_RESULT_PATH'], path_in_repo=os.environ['S5_SMOKE_HUB_PATH'], repo_id='shunk031/colab-jobs', repo_type='model', commit_message='Record CanvasVAE S5 executor CPU smoke')",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="cvae-s5-hub-smoke-result-") as temporary:
        hub_python(
            token_path,
            source,
            Path(temporary),
            {
                "S5_SMOKE_RESULT_PATH": str(result_path),
                "S5_SMOKE_HUB_PATH": hub_path,
            },
        )

    verify_hub_paths(token_path, [hub_path], "CPU smoke result")


def read_hub_exit_code(token_path: Path, hub_path: str) -> int:
    source = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "from huggingface_hub import hf_hub_download",
            "token = Path(os.environ['S5_HF_TOKEN_FILE']).read_text().strip()",
            "artifact = hf_hub_download(repo_id='shunk031/colab-jobs', filename=os.environ['S5_EXIT_CODE_HUB_PATH'], repo_type='model', revision='main', token=token, local_dir=os.environ['S5_HF_TEMP_DIR'])",
            "print(Path(artifact).read_text().strip())",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="cvae-s5-hub-exit-") as temporary:
        output = hub_python(
            token_path,
            source,
            Path(temporary),
            {"S5_EXIT_CODE_HUB_PATH": hub_path},
        )
    try:
        exit_code = int(output)
    except ValueError as error:
        raise ExecutorError(
            f"Hub wrapper exit code is not an integer: {hub_path}"
        ) from error
    if not 0 <= exit_code <= 255:
        raise ExecutorError(f"Hub wrapper exit code is out of range: {hub_path}")

    return exit_code


def record_smoke_exit_code(
    run_id: str,
    session_type: str,
    session_name: str,
    job_name: str,
    hub_prefix: str,
    exit_code: int,
) -> None:
    result_path_value = os.environ.get("CANVAS_VAE_S5_SMOKE_RESULT_PATH")
    if result_path_value is None:
        return
    result_path = Path(result_path_value).resolve()
    try:
        result_path.relative_to(REPOSITORY_ROOT / ".cache/canvas-vae/s5")
    except ValueError as error:
        raise ExecutorError("smoke result path must stay under the S5 cache") from error
    result = load_json(result_path)
    session_exit_codes = result.get("session_exit_codes")
    if not isinstance(session_exit_codes, dict):
        raise ExecutorError("smoke result is missing its session_exit_codes object")
    session_exit_codes[f"{run_id}:{session_type}"] = {
        "run_id": run_id,
        "session_name": session_name,
        "job_name": job_name,
        "hub_exit_code_path": f"{hub_prefix}/logs/{job_name}/exit_code",
        "exit_code": exit_code,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


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


def temporary_files(
    temporary_directory: Path,
    manifest: JsonObject,
    run: JsonObject,
    pinned_commit: str,
    download_inputs: bool,
    checkpoint_download: JsonObject | None = None,
    phase: str = "training",
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
    preflight_path = temporary_directory / f"{phase}-preflight.json"
    preflight_path.write_text(json.dumps(preflight_config, indent=2) + "\n")
    preflight_script = temporary_directory / f"{phase}-preflight.py"
    preflight_script.write_text(remote_preflight_source(), encoding="utf-8")
    return (
        preflight_path,
        preflight_script,
        temporary_directory / f"{phase}-run-job.sh",
    )


def build_run_job(
    manifest: JsonObject,
    run: JsonObject,
    pinned_commit: str,
    segment_id: str,
    resume_checkpoint: Path | None,
    failure_exit_code: int | None = None,
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
    checkpoint_max_epochs: int | None = None
    checkpoint_global_step: int | None = None
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
        checkpoint_path = f"{training_root}/checkpoints/last.ckpt"
        if run.get("cpu_smoke") is True:
            checkpoint_max_epochs = 1
            checkpoint_global_step = 2
        else:
            checkpoint_expectation = manifest.get("package_checkpoint_expectation")
            if not isinstance(checkpoint_expectation, dict):
                raise ExecutorError(
                    "manifest package_checkpoint_expectation is required"
                )
            max_epochs = checkpoint_expectation.get("max_epochs")
            global_step = checkpoint_expectation.get("global_step")
            if (
                type(max_epochs) is not int
                or type(global_step) is not int
                or max_epochs < 1
                or global_step < 0
            ):
                raise ExecutorError(
                    "package checkpoint expectation needs positive max_epochs and a nonnegative global_step"
                )
            checkpoint_max_epochs = max_epochs
            checkpoint_global_step = global_step
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
        if checkpoint_max_epochs is None or checkpoint_global_step is None:
            raise ExecutorError("package checkpoint gate was not configured")

        checkpoint_gate = (
            "uv run --frozen --no-sync --package canvas-vae --extra training "
            'python -m canvas_vae.training.checkpoints "$last_checkpoints" '
            f"--max-epochs {checkpoint_max_epochs} "
            f"--expected-global-step {checkpoint_global_step}"
        )
        post_train_lines = [
            f"last_checkpoints=\"$(find {shlex.quote(training_root)} -path '*/checkpoints/last.ckpt' -type f -print)\"",
            'test "$(printf \'%s\\n\' "$last_checkpoints" | wc -l)" -eq 1',
            'test -s "$last_checkpoints"',
            checkpoint_gate,
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
    if run.get("cpu_smoke") is True and system == "package":
        training_args.append("--trainer.enable_checkpointing=true")
    training_overrides = run.get("training_overrides", [])
    if not isinstance(training_overrides, list) or not all(
        isinstance(value, str) for value in training_overrides
    ):
        raise ExecutorError("training_overrides must be a list of strings")
    training_args.extend(training_overrides)

    smoke_data_prep = []
    if run.get("cpu_smoke") is True and system == "original":
        smoke_data_dir = run.get("smoke_original_data_dir")
        smoke_record_limit = run.get("smoke_original_record_limit")
        if (
            not isinstance(smoke_data_dir, str)
            or not smoke_data_dir.startswith("/content/repo/.cache/canvas-vae/")
            or not isinstance(smoke_record_limit, int)
            or smoke_record_limit < 1
        ):
            raise ExecutorError(
                "original CPU smoke needs a bounded local TFRecord subset"
            )
        smoke_data_prep = [
            "python - <<'PY'\n"
            "import json, shutil\n"
            "from pathlib import Path\n"
            "import tensorflow as tf\n"
            "source = Path('/content/repo/.cache/canvas-vae/original/data/rico')\n"
            f"target = Path({smoke_data_dir!r})\n"
            f"limit = {smoke_record_limit}\n"
            "target.mkdir(parents=True, exist_ok=False)\n"
            "counts = {}\n"
            "for split in ('train', 'val', 'test'):\n"
            "    files = sorted(source.glob(split + '-*.tfrecord'))\n"
            "    if not files:\n"
            "        raise SystemExit(f'missing downloaded {split} TFRecords')\n"
            "    output = target / f'{split}-00000-of-00001.tfrecord'\n"
            "    count = 0\n"
            "    with tf.io.TFRecordWriter(str(output)) as writer:\n"
            "        for record in tf.data.TFRecordDataset([str(path) for path in files]).take(limit):\n"
            "            writer.write(record.numpy())\n"
            "            count += 1\n"
            "    if count != limit:\n"
            "        raise SystemExit(f'expected {limit} {split} records, got {count}')\n"
            "    counts[split] = count\n"
            "for name in ('vocabulary.json',):\n"
            "    shutil.copy2(source / name, target / name)\n"
            "(target / 'count.json').write_text(json.dumps(counts), encoding='utf-8')\n"
            "print(f'tiny_original_input_subset_passed counts={counts}')\n"
            "PY",
        ]

    command_lines = [
        *env_lines,
        uv_sync,
        *smoke_data_prep,
        f"{shlex.join(training_args)} > {log_root}/train.log 2>&1",
        *post_train_lines,
    ]
    if failure_exit_code is not None:
        command_lines.append(f"exit {failure_exit_code}")

    return "\n".join(line for line in command_lines if line) + "\n"


def build_postprocess_job(
    manifest: JsonObject,
    run: JsonObject,
    segment_id: str,
    failure_exit_code: int | None = None,
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
                *(
                    [f"exit {failure_exit_code}"]
                    if failure_exit_code is not None
                    else []
                ),
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


def cpu_smoke_training_config(run_id: str, segment_id: str) -> str:
    training_root = f"/content/out/checkpoints/{run_id}/{segment_id}/training"
    return (
        "seed_everything: 0\n"
        "trainer:\n"
        "  accelerator: cpu\n"
        "  devices: 1\n"
        "  precision: 32-true\n"
        "  max_epochs: 1\n"
        "  limit_train_batches: 2\n"
        "  limit_val_batches: 1\n"
        "  check_val_every_n_epoch: 1\n"
        "  num_sanity_val_steps: 0\n"
        "  logger: false\n"
        "  enable_checkpointing: true\n"
        f"  default_root_dir: {training_root}\n"
        "  callbacks:\n"
        "    - class_path: lightning.pytorch.callbacks.ModelCheckpoint\n"
        "      init_args:\n"
        f"        dirpath: {training_root}/checkpoints\n"
        "        save_top_k: 0\n"
        "        save_last: true\n"
        "model:\n"
        "  class_path: canvas_vae.training.CanvasVAETrainingModule\n"
        "  init_args:\n"
        "    data_dir: /content/repo/.cache/canvas-vae/data/rico\n"
        "    latent_dim: 256\n"
        "    num_blocks: 1\n"
        "    num_heads: 8\n"
        "    dropout: 0.1\n"
        "    kl_weight: 16.0\n"
        "    l2_weight: 1.0e-6\n"
        "    learning_rate: 0.001\n"
        "    clip_norm: 1.0\n"
        "data:\n"
        "  class_path: canvas_vae.training.CanvasVAEDataModule\n"
        "  init_args:\n"
        "    data_dir: /content/repo/.cache/canvas-vae/data/rico\n"
        "    batch_size: 2\n"
        "    num_workers: 0\n"
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

    return (
        "import subprocess\nsubprocess.run(" + json.dumps(command) + ", check=True)\n"
    )


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
) -> int:
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
            "subprocess.run(['python', '/content/s5-preflight.py'], check=True)\n"
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
        wrapper_cli_error: ExecutorError | None = None
        try:
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
            )
        except ExecutorError as error:
            # A nonzero wrapper exit is read from its authoritative Hub artifact below.
            wrapper_cli_error = error
    except (ExecutorError, OSError, subprocess.SubprocessError):
        if not wrapper_started:
            colab("stop", "--session", session_name, timeout=300)

        # A retained endpoint may still be the only result copy after a failed final upload.
        raise

    exit_code = read_hub_exit_code(
        token_path, f"{hub_prefix}/logs/{job_name}/exit_code"
    )
    sessions = colab("sessions", capture=True)
    if re.search(rf"(?m)^\s*{re.escape(session_name)}(?:\s|$)", sessions):
        raise ExecutorError(
            f"wrapper exit code {exit_code} is on the Hub but session {session_name!r} was not released"
        )
    if wrapper_cli_error is not None and exit_code == 0:
        raise ExecutorError(
            f"Colab reported a wrapper error although Hub recorded exit code 0: {wrapper_cli_error}"
        )

    return exit_code


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
    cpu_runtime_smoke = runtime_plan.get("cpu_runtime_smoke") is True
    expected_gpu = "CPU" if cpu_runtime_smoke else "A100"
    if (
        runtime_plan.get("gpu") != expected_gpu
        or runtime_plan.get("one_named_a100_session_per_run") is not True
        or runtime_plan.get("one_named_cpu_session_per_run") is not True
    ):
        raise ExecutorError("runtime must name one A100 and one CPU session per run")
    if cpu_runtime_smoke and not run_id.startswith("cpu-smoke-"):
        raise ExecutorError("CPU runtime smoke accepts only cpu-smoke run IDs")
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
        or not isinstance(runtime_plan.get("cpu_wrapper_ttl"), str)
        or not re.fullmatch(r"[1-9][0-9]*m", runtime_plan["cpu_wrapper_ttl"])
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
    if (
        runtime_plan.get("require_hub_wrapper_exit_code_after_each_session") is not True
        or runtime_plan.get("hub_wrapper_exit_code_path") != "logs/<job>/exit_code"
        or runtime_plan.get("checkpoint_presence_gate_before_cpu_dispatch") is not True
        or runtime_plan.get("branch_must_remain_frozen_while_b2_running") is not True
    ):
        raise ExecutorError(
            "runtime must verify Hub completion/checkpoint artifacts and freeze the source branch"
        )
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
    failure_stage = run.get("smoke_failure_stage")
    failure_exit_code = run.get("smoke_failure_exit_code")
    if failure_stage is not None and (
        not cpu_runtime_smoke
        or failure_stage not in {"training", "postprocess"}
        or not isinstance(failure_exit_code, int)
        or not 1 <= failure_exit_code <= 125
    ):
        raise ExecutorError("smoke failure injection must be a bounded nonzero status")
    if cpu_runtime_smoke and failure_stage is None and failure_exit_code is not None:
        raise ExecutorError("smoke failure code requires an injected stage")

    with tempfile.TemporaryDirectory(prefix="cvae-s5-run-") as temporary:
        temporary_directory = Path(temporary)
        additional_training_uploads: tuple[tuple[Path, str], ...] = ()
        training_config_override = run.get("training_config_override")
        if cpu_runtime_smoke and system == "package":
            if training_config_override != "/content/s5-smoke.yaml":
                raise ExecutorError("CPU package smoke must use its tiny config")
            smoke_config_path = temporary_directory / "s5-smoke.yaml"
            smoke_config_path.write_text(
                cpu_smoke_training_config(run_id, segment_id), encoding="utf-8"
            )
            additional_training_uploads = (
                (smoke_config_path, "/content/s5-smoke.yaml"),
            )
        if cpu_runtime_smoke and system == "original":
            original_arguments = run.get("train_original_arguments")
            if not isinstance(original_arguments, list):
                raise ExecutorError(
                    "CPU original smoke needs explicit trainer arguments"
                )
            if (
                "--num-epochs" not in original_arguments
                or "1" not in original_arguments
            ):
                raise ExecutorError("CPU original smoke must train for one epoch")
        preflight_path, preflight_script, run_path = temporary_files(
            temporary_directory,
            manifest,
            run,
            pinned,
            download_inputs=True,
            phase="training",
        )
        run_path.write_text(
            build_run_job(
                manifest,
                run,
                pinned,
                segment_id,
                resume_checkpoint,
                failure_exit_code if failure_stage == "training" else None,
            ),
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
            phase="postprocess",
        )
        cpu_run_path.write_text(
            build_postprocess_job(
                manifest,
                run,
                segment_id,
                failure_exit_code if failure_stage == "postprocess" else None,
            ),
            encoding="utf-8",
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
        training_exit_code = run_named_session(
            session_name,
            None if cpu_runtime_smoke else "A100",
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
            additional_uploads=additional_training_uploads,
        )
        if cpu_runtime_smoke:
            record_smoke_exit_code(
                run_id,
                "training_session",
                session_name,
                job_name,
                training_prefix,
                training_exit_code,
            )
        if training_exit_code != 0:
            raise ExecutorError(
                f"training wrapper failed with Hub exit code {training_exit_code}"
            )
        verify_checkpoint_on_hub(token_path, checkpoint_hub_path, system)
        postprocess_exit_code = run_named_session(
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
        if cpu_runtime_smoke:
            record_smoke_exit_code(
                run_id,
                "cpu_postprocess_session",
                cpu_session_name,
                f"{job_name}-postprocess",
                postprocess_prefix,
                postprocess_exit_code,
            )
        if postprocess_exit_code != 0:
            raise ExecutorError(
                f"CPU postprocess wrapper failed with Hub exit code {postprocess_exit_code}"
            )
        verify_hub_paths(
            token_path,
            [
                f"{postprocess_prefix}/results/{run_id}/{segment_id}/evaluation.json",
                f"{postprocess_prefix}/results/{run_id}/{segment_id}/artifact-hashes.txt",
            ],
            "CPU evaluation artifacts",
        )


def execute_cpu_smoke(session_name: str, source_commit: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(session_name):
        raise ExecutorError("CPU smoke requires a valid smoke identifier")
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ExecutorError("CPU smoke source must be a full commit SHA")
    verify_local_source(source_commit)

    wrapper_value = os.environ.get("S5_WRAPPER_PATH")
    if wrapper_value is None:
        raise ExecutorError("S5_WRAPPER_PATH must name the installed bundled wrapper")
    wrapper_path = Path(wrapper_value).expanduser()
    if not wrapper_path.is_file():
        raise ExecutorError("installed bundled wrapper was not found")
    token_path = require_token_file()
    base_manifest = load_json(DEFAULT_MANIFEST)
    raw_runs = base_manifest.get("runs")
    if not isinstance(raw_runs, list):
        raise ExecutorError("B2 manifest does not define its source run templates")
    package_template = next(
        (
            run
            for run in raw_runs
            if isinstance(run, dict) and run.get("system") == "package"
        ),
        None,
    )
    original_template = next(
        (
            run
            for run in raw_runs
            if isinstance(run, dict) and run.get("system") == "original"
        ),
        None,
    )
    if package_template is None or original_template is None:
        raise ExecutorError(
            "B2 manifest must define package and original run templates"
        )

    smoke_id = uuid.uuid4().hex[:8]
    smoke_root = (
        REPOSITORY_ROOT / ".cache/canvas-vae/s5" / f"cpu-e2e-{session_name}-{smoke_id}"
    )
    smoke_root.mkdir(parents=True, exist_ok=False)
    hub_prefix = f"{HUB_REPOSITORY_PREFIX}/{session_name}-{smoke_id}"
    result_path = smoke_root / "smoke-result.json"
    result: JsonObject = {
        "source_commit": source_commit,
        "hub_prefix": hub_prefix,
        "session_exit_codes": {},
        "cases": {},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    raw_runtime_plan = base_manifest.get("runtime_plan")
    if not isinstance(raw_runtime_plan, dict):
        raise ExecutorError("B2 manifest runtime_plan must be an object")
    runtime_plan = dict(raw_runtime_plan)
    runtime_root = "${S5_RUNTIME_ROOT}"
    runtime_plan.update(
        {
            "cpu_runtime_smoke": True,
            "gpu": "CPU",
            "source_commit": source_commit,
            "one_named_a100_session_per_run": True,
            "one_named_cpu_session_per_run": True,
            "one_training_run_at_a_time": True,
            "queue_supervisor_sequential": True,
            "self_release": True,
            "sync_interval": "15m",
            "wrapper_script": "${S5_WRAPPER_PATH}",
            "a100_wrapper_ttl_by_system": {"package": "30m", "original": "30m"},
            "cpu_wrapper_ttl": "30m",
            "a100_sync_paths": [
                f"{runtime_root}/out/checkpoints",
                f"{runtime_root}/out/logs",
            ],
            "cpu_sync_paths": [
                f"{runtime_root}/out/converted",
                f"{runtime_root}/out/results",
                f"{runtime_root}/out/logs",
            ],
            "source_gate_before_artifact_sync": True,
        }
    )

    def make_run(
        template: JsonObject,
        case_id: str,
        system: str,
        failure_stage: str | None = None,
    ) -> JsonObject:
        run_id = f"cpu-smoke-{smoke_id}-{case_id}"
        artifact_prefix = f"{hub_prefix}/{run_id}"
        training_prefix = f"{artifact_prefix}/train/{{segment_id}}"
        postprocess_prefix = f"{artifact_prefix}/postprocess/{{segment_id}}"
        run: JsonObject = dict(template)
        run.update(
            {
                "run_id": run_id,
                "system": system,
                "seed": 0,
                "session_name": f"cvae-s5-{smoke_id}-{case_id}-a",
                "cpu_session_name": f"cvae-s5-{smoke_id}-{case_id}-c",
                "artifact_prefix": artifact_prefix,
                "runtime_output_root": f"/content/out/checkpoints/{run_id}/{{segment_id}}",
                "epoch_500_checkpoint": f"/content/out/checkpoints/{run_id}/{{segment_id}}/epoch-500.ckpt",
                "epoch_500_checkpoint_hub_path": (
                    f"{training_prefix}/checkpoints/{run_id}/{{segment_id}}/epoch-500.ckpt"
                ),
                "converted_output_root": f"/content/out/converted/{run_id}/{{segment_id}}",
                "evaluation_output": f"/content/out/results/{run_id}/{{segment_id}}/evaluation.json",
                "training_artifact_prefix": training_prefix,
                "postprocess_artifact_prefix": postprocess_prefix,
                "a100_wrapper_ttl": "30m",
                "cpu_wrapper_ttl": "30m",
                "cpu_smoke": True,
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
                    "1024",
                    "--device",
                    "cpu",
                    "--output",
                    "{evaluation_output}",
                ],
            }
        )
        if system == "original":
            run["train_original_arguments"] = [
                "--data-dir",
                f"/content/repo/.cache/canvas-vae/original-smoke/{run_id}/data/rico",
                "--job-dir",
                "{output_root}/original-training",
                "--seed",
                "0",
                "--",
                "--batch-size",
                "1024",
                "--num-epochs",
                "1",
                "--validation-freq",
                "1",
            ]
            run["smoke_original_data_dir"] = (
                f"/content/repo/.cache/canvas-vae/original-smoke/{run_id}/data/rico"
            )
            run["smoke_original_record_limit"] = 4
        else:
            run["training_config_override"] = "/content/s5-smoke.yaml"
            run["training_overrides"] = []
        if failure_stage is not None:
            run["smoke_failure_stage"] = failure_stage
            run["smoke_failure_exit_code"] = 37

        return run

    def write_case_manifest(case_id: str, runs: list[JsonObject]) -> Path:
        manifest = dict(base_manifest)
        manifest.update(
            {
                "queue_status": "ready",
                "source_commit": source_commit,
                "runtime_plan": runtime_plan,
                "runs": runs,
            }
        )
        manifest_path = smoke_root / f"{case_id}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        return manifest_path

    def run_case(
        case_id: str,
        runs: list[JsonObject],
        expected_queue_success: bool,
        expected_session_codes: dict[str, int],
    ) -> None:
        manifest_path = write_case_manifest(case_id, runs)
        log_path = smoke_root / f"{case_id}.queue.log"
        environment = os.environ.copy()
        environment["CANVAS_VAE_S5_SMOKE_RESULT_PATH"] = str(result_path)
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    REPOSITORY_ROOT / "models/canvas-vae/scripts/s5_rico_seed_queue.py"
                ),
                "--manifest",
                str(manifest_path),
                "--launch",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=21600,
        )
        combined_output = completed.stdout + completed.stderr
        log_path.write_text(combined_output, encoding="utf-8")
        queue_records = []
        for line in completed.stdout.splitlines():
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                queue_records.append(decoded)
        event_names = [record.get("event") for record in queue_records]
        dispatched_run_ids = [
            record.get("run_id")
            for record in queue_records
            if record.get("event") == "run_dispatch"
        ]
        smoke_result = load_json(result_path)
        all_session_codes = smoke_result.get("session_exit_codes")
        if not isinstance(all_session_codes, dict):
            raise ExecutorError(f"{case_id} did not record Hub exit codes")
        run_id = runs[0]["run_id"]
        actual_codes = {
            key.split(":", 1)[1]: value.get("exit_code")
            for key, value in all_session_codes.items()
            if key.startswith(f"{run_id}:") and isinstance(value, dict)
        }
        if actual_codes != expected_session_codes:
            raise ExecutorError(
                f"{case_id} Hub exit-code evidence differs from expectation: {actual_codes}"
            )
        if expected_queue_success:
            if completed.returncode != 0 or "queue_complete" not in event_names:
                raise ExecutorError(f"{case_id} queue did not complete successfully")
            if dispatched_run_ids != [run_id]:
                raise ExecutorError(f"{case_id} dispatched an unexpected run sequence")
        else:
            sentinel_id = runs[1]["run_id"]
            if (
                completed.returncode == 0
                or "run_failed" not in event_names
                or dispatched_run_ids != [run_id]
                or sentinel_id in dispatched_run_ids
            ):
                raise ExecutorError(
                    f"{case_id} did not stop the queue after its injected session failure"
                )
        cases = smoke_result.get("cases")
        if not isinstance(cases, dict):
            raise ExecutorError("smoke result is missing its cases object")
        cases[case_id] = {
            "system": runs[0]["system"],
            "run_id": run_id,
            "queue_return_code": completed.returncode,
            "queue_events": event_names,
            "dispatched_run_ids": dispatched_run_ids,
            "session_exit_codes": actual_codes,
            "hub_training_prefix": runs[0]["training_artifact_prefix"],
            "hub_postprocess_prefix": runs[0]["postprocess_artifact_prefix"],
            "queue_log": str(log_path.relative_to(REPOSITORY_ROOT)),
        }
        result_path.write_text(
            json.dumps(smoke_result, indent=2) + "\n", encoding="utf-8"
        )

    cases = [
        (
            "package-success",
            package_template,
            "package",
            None,
            True,
            {"training_session": 0, "cpu_postprocess_session": 0},
        ),
        (
            "original-success",
            original_template,
            "original",
            None,
            True,
            {"training_session": 0, "cpu_postprocess_session": 0},
        ),
        (
            "training-failure",
            package_template,
            "package",
            "training",
            False,
            {"training_session": 37},
        ),
        (
            "postprocess-failure",
            package_template,
            "package",
            "postprocess",
            False,
            {"training_session": 0, "cpu_postprocess_session": 37},
        ),
    ]
    for case_id, template, system, failure_stage, success, expected_codes in cases:
        target = make_run(template, case_id, system, failure_stage)
        runs = [target]
        if not success:
            runs.append(make_run(package_template, f"{case_id}-sentinel", "package"))
        run_case(case_id, runs, success, expected_codes)

    result = load_json(result_path)
    result["status"] = "passed"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    hub_result_path = f"{hub_prefix}/smoke-result.json"
    upload_smoke_result(token_path, result_path, hub_result_path)
    print(
        json.dumps(
            {
                "cpu_e2e_smoke": "passed",
                "result": str(result_path),
                "hub_prefix": hub_prefix,
                "hub_result_path": hub_result_path,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--segment-id", default="segment-001")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--session-name")
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            if args.cpu_smoke or args.resume_checkpoint is not None:
                raise ExecutorError(
                    "dry-run cannot be combined with --cpu-smoke or --resume-checkpoint"
                )
            if args.run_id is None:
                raise ExecutorError("--run-id is required for a dry run")

            manifest = load_json(args.manifest.resolve())
            pinned_commit = args.source_commit or manifest.get("source_commit")
            if not isinstance(pinned_commit, str):
                raise ExecutorError("dry-run requires a full source_commit")
            verify_local_source(pinned_commit)
            raw_runs = manifest.get("runs")
            if not isinstance(raw_runs, list):
                raise ExecutorError("manifest must define a runs list")
            run = next(
                (
                    candidate
                    for candidate in raw_runs
                    if isinstance(candidate, dict)
                    and candidate.get("run_id") == args.run_id
                ),
                None,
            )
            if run is None:
                raise ExecutorError(f"manifest has no run {args.run_id!r}")
            if not RUN_ID_PATTERN.fullmatch(args.run_id):
                raise ExecutorError(f"invalid run_id {args.run_id!r}")
            if not RUN_ID_PATTERN.fullmatch(args.segment_id):
                raise ExecutorError(f"invalid segment_id {args.segment_id!r}")
            checkpoint_template = run.get("epoch_500_checkpoint")
            checkpoint_hub_template = run.get("epoch_500_checkpoint_hub_path")
            if not isinstance(checkpoint_template, str) or not isinstance(
                checkpoint_hub_template, str
            ):
                raise ExecutorError("run is missing checkpoint output templates")
            try:
                checkpoint_path = checkpoint_template.format(
                    run_id=args.run_id,
                    segment_id=args.segment_id,
                )
                checkpoint_hub_path = checkpoint_hub_template.format(
                    run_id=args.run_id,
                    segment_id=args.segment_id,
                )
            except (KeyError, ValueError) as error:
                raise ExecutorError(
                    "run checkpoint output templates are invalid"
                ) from error
            expected_checkpoint_path = f"/content/out/checkpoints/{args.run_id}/{args.segment_id}/epoch-500.ckpt"
            artifact_prefix = run.get("artifact_prefix")
            expected_checkpoint_hub_path = (
                f"{artifact_prefix}/train/{args.segment_id}/checkpoints/"
                f"{args.run_id}/{args.segment_id}/epoch-500.ckpt"
            )
            if checkpoint_path != expected_checkpoint_path or (
                checkpoint_hub_path != expected_checkpoint_hub_path
            ):
                raise ExecutorError(
                    "run checkpoint paths do not match the selected run and segment"
                )

            run_job = build_run_job(
                manifest,
                run,
                pinned_commit,
                args.segment_id,
                resume_checkpoint=None,
            )
            has_package_checkpoint_gate = "canvas_vae.training.checkpoints" in run_job
            if run.get("system") == "package" and not has_package_checkpoint_gate:
                raise ExecutorError(
                    "package dry-run is missing the final-checkpoint gate"
                )
            result: JsonObject = {
                "event": "executor_dry_run_passed",
                "source_commit": pinned_commit,
                "run_id": args.run_id,
                "segment_id": args.segment_id,
                "epoch_500_checkpoint": checkpoint_path,
                "epoch_500_checkpoint_hub_path": checkpoint_hub_path,
                "system": run.get("system"),
                "seed": run.get("seed"),
                "training_job_contains_final_checkpoint_gate": (
                    has_package_checkpoint_gate
                ),
            }
            if run.get("system") == "package":
                expectation = manifest.get("package_checkpoint_expectation")
                if not isinstance(expectation, dict):
                    raise ExecutorError(
                        "manifest package_checkpoint_expectation is required"
                    )
                max_epochs = expectation.get("max_epochs")
                global_step = expectation.get("global_step")
                if type(max_epochs) is not int or type(global_step) is not int:
                    raise ExecutorError("package checkpoint expectation is invalid")
                result["checkpoint_gate"] = {
                    "expected_epoch": max_epochs - 1,
                    "expected_global_step": global_step,
                }
            print(json.dumps(result, sort_keys=True))
        elif args.cpu_smoke:
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
