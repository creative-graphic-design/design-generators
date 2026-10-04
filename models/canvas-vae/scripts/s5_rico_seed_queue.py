"""Plan or supervise the CanvasVAE RICO S5 seed queue.

The queue pins one clean source commit and checks it before every run. Its
per-run executor is intentionally supplied only after all S5 launch gates,
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
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "models/canvas-vae/configs/training/s5_rico_launch_manifest.template.json"
)
BLOCKED_EXIT = 78
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def source_gate(run: dict[str, Any], pinned_commit: str, require_clean: bool) -> None:
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


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QueueError(f"cannot read manifest {path}: {error}") from error

    if not isinstance(manifest, dict):
        raise QueueError("manifest root must be a JSON object")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise QueueError("manifest must define a non-empty runs list")
    for run in runs:
        if not isinstance(run, dict) or not all(
            isinstance(run.get(key), str) and run[key]
            for key in ("run_id", "system", "session_name", "artifact_prefix")
        ):
            raise QueueError("each run needs run_id, system, session_name, and artifact_prefix")

    return manifest


def launch_preconditions(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[str, list[str]]:
    pinned_commit = manifest.get("source_commit")
    if not isinstance(pinned_commit, str) or not COMMIT_PATTERN.fullmatch(pinned_commit):
        raise QueueError("launch requires a finalized 40-character source_commit")
    if manifest.get("queue_status") != "ready":
        raise QueueError("launch refused: manifest queue_status is not ready", BLOCKED_EXIT)

    parity = manifest.get("evaluation_path_parity")
    if not isinstance(parity, dict):
        raise QueueError("launch refused: evaluation_path_parity artifact is missing", BLOCKED_EXIT)
    parity_path = parity.get("path")
    parity_sha256 = parity.get("sha256")
    if not isinstance(parity_path, str) or not parity_path:
        raise QueueError("launch refused: evaluation_path_parity path is missing", BLOCKED_EXIT)
    if not isinstance(parity_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", parity_sha256):
        raise QueueError("launch refused: evaluation_path_parity SHA-256 is missing", BLOCKED_EXIT)
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
    if not isinstance(runner, list) or not runner or not all(
        isinstance(part, str) and part for part in runner
    ):
        raise QueueError("launch refused: runner_command is not configured", BLOCKED_EXIT)

    try:
        manifest_path.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise QueueError("manifest is outside the supervisor's repository") from error

    manifest_arg = os.path.relpath(manifest_path.resolve(), REPOSITORY_ROOT)

    return pinned_commit, [*runner, "--manifest", manifest_arg]


def run_queue(
    manifest: dict[str, Any],
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
            raise QueueError("launch refused: runner command is not configured", BLOCKED_EXIT)

        environment = os.environ.copy()
        environment.update(
            {
                "CANVAS_VAE_S5_RUN_ID": run["run_id"],
                "CANVAS_VAE_S5_SYSTEM": run["system"],
                "CANVAS_VAE_S5_SEED": str(run["seed"]),
                "CANVAS_VAE_S5_SESSION": run["session_name"],
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
            pinned_commit, runner_command = launch_preconditions(manifest, manifest_path)
            if current_commit != pinned_commit:
                raise QueueError(
                    "launch refused: current HEAD differs from source_commit",
                    BLOCKED_EXIT,
                )
        else:
            manifest_commit = manifest.get("source_commit")
            pinned_commit = current_commit
            if isinstance(manifest_commit, str) and COMMIT_PATTERN.fullmatch(manifest_commit):
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
