"""Plan or supervise the CanvasVAE RICO full-run seed queue.

The queue pins one clean source commit and checks it before every run. Its
per-run executor is intentionally supplied only after all launch gates,
including conversion and evaluation-path parity, are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TypeAlias, TypedDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "models/canvas-vae/configs/training/s5_rico_launch_manifest.template.json"
)
BLOCKED_EXIT = 78
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class QueueRun(TypedDict):
    """One original or package training run."""

    run_id: str
    system: str
    seed: int
    session_name: str
    cpu_session_name: str
    a100_wrapper_ttl: str
    cpu_wrapper_ttl: str
    artifact_prefix: str
    runtime_output_root: str
    epoch_500_checkpoint: str
    epoch_500_checkpoint_hub_path: str
    converted_output_root: str
    evaluation_output: str
    training_artifact_prefix: str
    postprocess_artifact_prefix: str


class ParityArtifact(TypedDict):
    """Repository-relative evaluation parity file and digest."""

    path: str
    sha256: str


class QueueManifest(TypedDict):
    """Validated queue fields used by the supervisor."""

    dataset: str
    queue_status: str
    source_commit: str | None
    evaluation_path_parity: ParityArtifact
    runner_command: list[str] | None
    runtime_plan: dict[str, JsonValue]
    runs: list[QueueRun]


class QueueError(Exception):
    """A queue precondition or source gate failed."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise QueueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")

    return result.stdout.strip()


def current_source_state() -> tuple[str, bool]:
    commit = git_output("rev-parse", "HEAD")
    clean = not git_output("status", "--porcelain")

    return commit, clean


def emit(record: dict[str, JsonValue]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def source_gate(run: QueueRun, pinned_commit: str, require_clean: bool) -> None:
    current_commit, clean = current_source_state()
    passed = current_commit == pinned_commit and (clean or not require_clean)
    emit(
        {
            "event": "source_gate_passed" if passed else "source_gate_failed",
            "run_id": run["run_id"],
            "pinned_source_commit": pinned_commit,
            "current_source_commit": current_commit,
            "worktree_clean": clean,
        }
    )
    if not passed:
        raise QueueError(
            f"source gate failed before {run['run_id']}; queue stopped before dispatch",
            BLOCKED_EXIT,
        )


def load_manifest(path: Path) -> QueueManifest:
    try:
        decoded: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QueueError(f"cannot read manifest {path}: {error}") from error

    if not isinstance(decoded, dict):
        raise QueueError("manifest root must be a JSON object")

    dataset = decoded.get("dataset")
    queue_status = decoded.get("queue_status")
    source_commit = decoded.get("source_commit")
    if not isinstance(dataset, str) or not isinstance(queue_status, str):
        raise QueueError("manifest needs dataset and queue_status strings")
    if source_commit is not None and not isinstance(source_commit, str):
        raise QueueError("manifest source_commit must be a string or null")

    parity = decoded.get("evaluation_path_parity")
    if not isinstance(parity, dict):
        raise QueueError("manifest evaluation_path_parity must be an object")
    parity_path = parity.get("path")
    parity_sha256 = parity.get("sha256")
    if not isinstance(parity_path, str) or not isinstance(parity_sha256, str):
        raise QueueError(
            "manifest evaluation_path_parity needs path and sha256 strings"
        )

    runner = decoded.get("runner_command")
    if runner is not None and (
        not isinstance(runner, list)
        or not all(isinstance(part, str) for part in runner)
    ):
        raise QueueError("manifest runner_command must be a string list or null")
    runtime_plan = decoded.get("runtime_plan")
    if not isinstance(runtime_plan, dict):
        raise QueueError("manifest runtime_plan must be an object")

    raw_runs = decoded.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise QueueError("manifest must define a non-empty runs list")
    runs: list[QueueRun] = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise QueueError(
                "each run needs run_id, system, session_name, and artifact_prefix"
            )
        run_id = raw_run.get("run_id")
        system = raw_run.get("system")
        seed = raw_run.get("seed")
        session_name = raw_run.get("session_name")
        cpu_session_name = raw_run.get("cpu_session_name")
        a100_wrapper_ttl = raw_run.get("a100_wrapper_ttl")
        cpu_wrapper_ttl = raw_run.get("cpu_wrapper_ttl")
        artifact_prefix = raw_run.get("artifact_prefix")
        runtime_output_root = raw_run.get("runtime_output_root")
        epoch_500_checkpoint = raw_run.get("epoch_500_checkpoint")
        epoch_500_checkpoint_hub_path = raw_run.get("epoch_500_checkpoint_hub_path")
        converted_output_root = raw_run.get("converted_output_root")
        evaluation_output = raw_run.get("evaluation_output")
        training_artifact_prefix = raw_run.get("training_artifact_prefix")
        postprocess_artifact_prefix = raw_run.get("postprocess_artifact_prefix")
        if not all(
            isinstance(value, str) and value
            for value in (
                run_id,
                system,
                session_name,
                cpu_session_name,
                a100_wrapper_ttl,
                cpu_wrapper_ttl,
                artifact_prefix,
                runtime_output_root,
                epoch_500_checkpoint,
                epoch_500_checkpoint_hub_path,
                converted_output_root,
                evaluation_output,
                training_artifact_prefix,
                postprocess_artifact_prefix,
            )
        ) or not isinstance(seed, int):
            raise QueueError(
                "each run needs named A100/CPU runtimes, unique output templates, and an integer seed"
            )
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
            raise QueueError(f"invalid run_id {run_id!r}")
        if system not in {"package", "original"}:
            raise QueueError(f"invalid run system {system!r}")
        if runtime_output_root != f"/content/out/checkpoints/{run_id}/{{segment_id}}":
            raise QueueError(
                f"run {run_id} has a non-unique checkpoint output template"
            )
        if (
            epoch_500_checkpoint
            != f"/content/out/checkpoints/{run_id}/{{segment_id}}/epoch-500.ckpt"
        ):
            raise QueueError(
                f"run {run_id} has a non-unique epoch-500 checkpoint template"
            )
        if epoch_500_checkpoint_hub_path != (
            f"{artifact_prefix}/train/{{segment_id}}/checkpoints/"
            f"{run_id}/{{segment_id}}/epoch-500.ckpt"
        ):
            raise QueueError(
                f"run {run_id} has an invalid epoch-500 checkpoint Hub path"
            )
        if converted_output_root != f"/content/out/converted/{run_id}/{{segment_id}}":
            raise QueueError(
                f"run {run_id} has a non-unique converted checkpoint template"
            )
        if (
            evaluation_output
            != f"/content/out/results/{run_id}/{{segment_id}}/evaluation.json"
        ):
            raise QueueError(
                f"run {run_id} has a non-unique evaluation output template"
            )
        if training_artifact_prefix != f"{artifact_prefix}/train/{{segment_id}}":
            raise QueueError(f"run {run_id} has an invalid training Hub prefix")
        if (
            postprocess_artifact_prefix
            != f"{artifact_prefix}/postprocess/{{segment_id}}"
        ):
            raise QueueError(f"run {run_id} has an invalid postprocess Hub prefix")
        runs.append(
            {
                "run_id": run_id,
                "system": system,
                "seed": seed,
                "session_name": session_name,
                "cpu_session_name": cpu_session_name,
                "a100_wrapper_ttl": a100_wrapper_ttl,
                "cpu_wrapper_ttl": cpu_wrapper_ttl,
                "artifact_prefix": artifact_prefix,
                "runtime_output_root": runtime_output_root,
                "epoch_500_checkpoint": epoch_500_checkpoint,
                "epoch_500_checkpoint_hub_path": epoch_500_checkpoint_hub_path,
                "converted_output_root": converted_output_root,
                "evaluation_output": evaluation_output,
                "training_artifact_prefix": training_artifact_prefix,
                "postprocess_artifact_prefix": postprocess_artifact_prefix,
            }
        )

    unique_fields = (
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
    )
    for field in unique_fields:
        values = [run[field] for run in runs]
        if len(values) != len(set(values)):
            raise QueueError(f"manifest runs must have unique {field} values")
    session_names = [
        session_name
        for run in runs
        for session_name in (run["session_name"], run["cpu_session_name"])
    ]
    if len(session_names) != len(set(session_names)):
        raise QueueError("manifest A100 and CPU session names must be globally unique")

    return {
        "dataset": dataset,
        "queue_status": queue_status,
        "source_commit": source_commit,
        "evaluation_path_parity": {"path": parity_path, "sha256": parity_sha256},
        "runner_command": runner,
        "runtime_plan": runtime_plan,
        "runs": runs,
    }


def launch_preconditions(
    manifest: QueueManifest, manifest_path: Path
) -> tuple[str, list[str]]:
    pinned_commit = manifest.get("source_commit")
    if not isinstance(pinned_commit, str) or not COMMIT_PATTERN.fullmatch(
        pinned_commit
    ):
        raise QueueError("launch requires a finalized 40-character source_commit")
    if manifest.get("queue_status") != "ready":
        raise QueueError(
            "launch refused: manifest queue_status is not ready", BLOCKED_EXIT
        )

    parity = manifest["evaluation_path_parity"]
    parity_path = parity.get("path")
    parity_sha256 = parity.get("sha256")
    if not isinstance(parity_path, str) or not parity_path:
        raise QueueError(
            "launch refused: evaluation_path_parity path is missing", BLOCKED_EXIT
        )
    if not isinstance(parity_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", parity_sha256
    ):
        raise QueueError(
            "launch refused: evaluation_path_parity SHA-256 is missing", BLOCKED_EXIT
        )
    parity_file = (REPOSITORY_ROOT / parity_path).resolve()
    try:
        parity_file.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise QueueError(
            "evaluation_path_parity path must stay inside the repository"
        ) from error
    if not parity_file.is_file():
        raise QueueError(
            "launch refused: evaluation_path_parity artifact is not present",
            BLOCKED_EXIT,
        )
    digest = sha256_file(parity_file)
    if digest != parity_sha256:
        raise QueueError(
            "launch refused: evaluation_path_parity SHA-256 does not match",
            BLOCKED_EXIT,
        )

    runner = manifest.get("runner_command")
    if not isinstance(runner, list) or runner != [
        "python3",
        "models/canvas-vae/scripts/s5_rico_seed_executor.py",
    ]:
        raise QueueError(
            "launch refused: runner_command must call the fail-closed per-run executor",
            BLOCKED_EXIT,
        )
    runtime_plan = manifest["runtime_plan"]
    a100_ttls = runtime_plan.get("a100_wrapper_ttl_by_system")
    cpu_ttl = runtime_plan.get("cpu_wrapper_ttl")
    cpu_runtime_smoke = runtime_plan.get("cpu_runtime_smoke") is True
    expected_gpu = "CPU" if cpu_runtime_smoke else "A100"
    if (
        runtime_plan.get("source_commit") != pinned_commit
        or runtime_plan.get("gpu") != expected_gpu
        or runtime_plan.get("one_named_a100_session_per_run") is not True
        or runtime_plan.get("one_named_cpu_session_per_run") is not True
        or runtime_plan.get("one_training_run_at_a_time") is not True
        or runtime_plan.get("queue_supervisor_sequential") is not True
        or runtime_plan.get("self_release") is not True
        or runtime_plan.get("sync_interval") != "15m"
        or not isinstance(a100_ttls, dict)
        or not all(
            isinstance(a100_ttls.get(system), str)
            and re.fullmatch(r"[1-9][0-9]*m", a100_ttls[system])
            for system in ("package", "original")
        )
        or not isinstance(cpu_ttl, str)
        or not re.fullmatch(r"[1-9][0-9]*m", cpu_ttl)
        or runtime_plan.get("wrapper_script") != "${S5_WRAPPER_PATH}"
        or runtime_plan.get("a100_sync_paths")
        != ["${S5_RUNTIME_ROOT}/out/checkpoints", "${S5_RUNTIME_ROOT}/out/logs"]
        or runtime_plan.get("cpu_sync_paths")
        != [
            "${S5_RUNTIME_ROOT}/out/converted",
            "${S5_RUNTIME_ROOT}/out/results",
            "${S5_RUNTIME_ROOT}/out/logs",
        ]
        or runtime_plan.get("source_gate_before_artifact_sync") is not True
        or runtime_plan.get("require_hub_wrapper_exit_code_after_each_session")
        is not True
        or runtime_plan.get("hub_wrapper_exit_code_path") != "logs/<job>/exit_code"
        or runtime_plan.get("checkpoint_presence_gate_before_cpu_dispatch") is not True
        or runtime_plan.get("branch_must_remain_frozen_while_b2_running") is not True
    ):
        raise QueueError(
            "launch refused: manifest runtime plan is not sequential and fail-closed",
            BLOCKED_EXIT,
        )
    if cpu_runtime_smoke and not all(
        run["run_id"].startswith("cpu-smoke-") for run in manifest["runs"]
    ):
        raise QueueError(
            "launch refused: CPU runtime smoke accepts only cpu-smoke run IDs",
            BLOCKED_EXIT,
        )
    for run in manifest["runs"]:
        expected_a100_ttl = a100_ttls.get(run["system"])
        if run["a100_wrapper_ttl"] != expected_a100_ttl or run[
            "cpu_wrapper_ttl"
        ] != runtime_plan.get("cpu_wrapper_ttl"):
            raise QueueError(
                f"launch refused: run {run['run_id']} TTLs differ from the runtime plan",
                BLOCKED_EXIT,
            )

    try:
        manifest_path.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise QueueError("manifest is outside the supervisor's repository") from error

    manifest_arg = os.path.relpath(manifest_path.resolve(), REPOSITORY_ROOT)

    return pinned_commit, [*runner, "--manifest", manifest_arg]


def run_queue(
    manifest: QueueManifest,
    pinned_commit: str,
    runner_command: list[str] | None,
    dry_run: bool,
) -> None:
    emit(
        {
            "event": "queue_start",
            "dataset": manifest.get("dataset"),
            "mode": "dry_run" if dry_run else "launch",
            "pinned_source_commit": pinned_commit,
            "run_count": len(manifest["runs"]),
        }
    )

    for run in manifest["runs"]:
        source_gate(run, pinned_commit, require_clean=not dry_run)
        if dry_run:
            emit({"event": "run_planned", **run})
            continue

        if runner_command is None:
            raise QueueError(
                "launch refused: runner command is not configured", BLOCKED_EXIT
            )

        environment = os.environ.copy()
        environment.update(
            {
                "CANVAS_VAE_S5_RUN_ID": run["run_id"],
                "CANVAS_VAE_S5_SYSTEM": run["system"],
                "CANVAS_VAE_S5_SEED": str(run["seed"]),
                "CANVAS_VAE_S5_SESSION": run["session_name"],
                "CANVAS_VAE_S5_CPU_SESSION": run["cpu_session_name"],
                "CANVAS_VAE_S5_ARTIFACT_PREFIX": run["artifact_prefix"],
                "CANVAS_VAE_S5_SOURCE_COMMIT": pinned_commit,
            }
        )
        emit({"event": "run_dispatch", "run_id": run["run_id"]})
        result = subprocess.run(
            [*runner_command, "--run-id", run["run_id"]],
            cwd=REPOSITORY_ROOT,
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            emit(
                {
                    "event": "run_failed",
                    "run_id": run["run_id"],
                    "exit_code": result.returncode,
                }
            )
            raise QueueError(
                f"runner failed for {run['run_id']}; queue stopped",
                result.returncode if result.returncode > 0 else 1,
            )
        emit({"event": "run_completed", "run_id": run["run_id"]})

    emit({"event": "queue_complete", "run_count": len(manifest["runs"])})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--test-source-mismatch",
        action="store_true",
        help="dry-run only: pin an intentionally incorrect commit and prove fail-closed behavior",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test_source_mismatch and not args.dry_run:
        print("--test-source-mismatch requires --dry-run", file=sys.stderr)
        return 2

    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        current_commit, clean = current_source_state()
        if args.launch and not clean:
            raise QueueError("source gate requires a clean worktree")

        if args.launch:
            pinned_commit, runner_command = launch_preconditions(
                manifest, manifest_path
            )
            if current_commit != pinned_commit:
                raise QueueError(
                    "launch refused: current HEAD differs from source_commit",
                    BLOCKED_EXIT,
                )
        else:
            manifest_commit = manifest.get("source_commit")
            pinned_commit = current_commit
            if isinstance(manifest_commit, str) and COMMIT_PATTERN.fullmatch(
                manifest_commit
            ):
                pinned_commit = manifest_commit
            runner_command = None
            if args.test_source_mismatch:
                pinned_commit = "0" * 40

        run_queue(manifest, pinned_commit, runner_command, args.dry_run)
    except QueueError as error:
        print(f"queue stopped: {error}", file=sys.stderr)
        return error.exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
