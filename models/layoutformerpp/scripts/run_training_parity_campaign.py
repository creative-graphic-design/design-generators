"""Run the approved six-process LayoutFormer++ staged campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.utils import parse_wheel_filename


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
TEST = (
    "models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py"
)
EXPECTED_FREEZE = SCRIPT_ROOT / "audit_runtime.freeze"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def redact(value: str, data_root: str) -> str:
    return value.replace(data_root, "<your data root>").replace(
        str(REPO_ROOT), "<repo root>"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_freeze(raw: str) -> str:
    """Return a stable freeze without machine-specific wheel or source paths."""
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("-e "):
            location = line[3:].strip()
            if location.startswith("file://<repo root>/"):
                line = f"-e {location.removeprefix('file://<repo root>/')}"
            elif location.startswith("file://"):
                parsed = urlparse(location)
                path = Path(unquote(parsed.path)).resolve()
                try:
                    relative = path.relative_to(REPO_ROOT)
                except ValueError as error:
                    raise RuntimeError(
                        f"editable install is outside the repository: {path}"
                    ) from error
                line = f"-e {relative.as_posix()}"
            else:
                line = line.replace(str(REPO_ROOT), "<repo root>")
        elif " @ file://" in line:
            # The evidence runtime installs local torch wheels; normalize their file URLs.
            distribution, _, location = line.partition(" @ ")
            wheel_path = Path(unquote(urlparse(location).path))
            try:
                wheel_distribution, wheel_version, _, _ = parse_wheel_filename(
                    wheel_path.name
                )
            except Exception as error:
                raise RuntimeError(f"cannot normalize wheel URL: {line}") from error
            if distribution != str(wheel_distribution):
                raise RuntimeError(
                    f"wheel distribution mismatch: {distribution} != {wheel_distribution}"
                )
            line = f"{wheel_distribution}=={wheel_version}"
        else:
            line = line.replace(str(REPO_ROOT), "<repo root>")
        lines.append(line)
    return "\n".join(sorted(lines, key=str.casefold)) + "\n"


def _runtime_paths() -> tuple[str, Path]:
    relative = os.environ.get("LAYOUTFORMERPP_AUDIT_VENV")
    if not relative:
        raise SystemExit("LAYOUTFORMERPP_AUDIT_VENV must point to the audited runtime")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit("LAYOUTFORMERPP_AUDIT_VENV must be repository-relative")
    runtime = REPO_ROOT / path
    return relative.rstrip("/"), runtime / "bin/python"


def _runtime_freeze(runtime_python: Path) -> str:
    raw = subprocess.check_output(
        ["uv", "pip", "freeze", "--no-config", "--python", str(runtime_python)],
        cwd=REPO_ROOT,
        text=True,
    )
    return normalized_freeze(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not args.attempt or not args.attempt.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("--attempt must be a simple directory name")
    data_root = os.environ.get("LAYOUTFORMERPP_PARITY_DATA_ROOT")
    if not data_root or not Path(data_root).is_dir():
        raise SystemExit(
            "LAYOUTFORMERPP_PARITY_DATA_ROOT must name the approved processed-data root"
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set")
    runtime_relative, runtime_python = _runtime_paths()
    if not runtime_python.is_file():
        raise SystemExit("audited runtime is missing its Python interpreter")
    expected_freeze = EXPECTED_FREEZE.read_text(encoding="utf-8")
    actual_freeze = _runtime_freeze(runtime_python)
    if actual_freeze != expected_freeze:
        raise SystemExit(
            "audited runtime freeze differs from committed expected freeze"
        )
    tracked_changes = (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
    )
    if tracked_changes:
        raise SystemExit("commit tracked changes before launching the campaign")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    runtime_freeze_sha256 = sha256(EXPECTED_FREEZE)
    run_root = REPO_ROOT / ".cache/layoutformerpp/post-merge" / args.attempt
    if run_root.exists():
        raise SystemExit(f"refusing to reuse existing campaign directory: {run_root}")
    run_root.mkdir(parents=True)
    preflight_environment = os.environ.copy()
    smoke = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "import torch; assert torch.cuda.is_available(); device=torch.cuda.current_device(); x=torch.ones((2,2), device='cuda'); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(device), (x @ x).tolist())",
        ],
        cwd=REPO_ROOT,
        env=preflight_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    preflight = {
        "runtime_interpreter": f"{runtime_relative}/bin/python",
        "runtime_freeze": "models/layoutformerpp/scripts/audit_runtime.freeze",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "cuda_visible_devices": visible_devices,
        "command": "import torch; assert torch.cuda.is_available(); tiny matmul on the current CUDA device",
        "returncode": smoke.returncode,
        "stdout": redact(smoke.stdout, data_root),
        "stderr": redact(smoke.stderr, data_root),
    }
    write_new(
        run_root / "cuda-smoke.json",
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
    )
    if smoke.returncode != 0:
        raise SystemExit(
            "audited runtime failed the CUDA smoke; no parity process launched"
        )

    runs = [
        ("s0", 1, ["-k", "s0"]),
        ("s1", 1, ["-k", "s1"]),
        ("s2", 1, ["-k", "s2"]),
        ("s3", 1, ["-k", "s3"]),
        ("s3", 2, ["-k", "s3"]),
        ("s4", 1, ["-k", "s4"]),
    ]
    manifest = {
        "manifest_version": 2,
        "source_commit": source_commit,
        "created_at": now(),
        "runtime_interpreter": f"{runtime_relative}/bin/python",
        "runtime_freeze": "models/layoutformerpp/scripts/audit_runtime.freeze",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "cuda_visible_devices": visible_devices,
        "data_root": "<your data root>",
        "runner": "models/layoutformerpp/scripts/run_training_parity_campaign.py",
        "record_plugin": "models/layoutformerpp/scripts/training_parity_record_plugin.py",
        "s3_gate": {
            "comparison": "fixed tolerance",
            "rtol": 1e-4,
            "atol": 1e-5,
            "repeat_two": "separate-process reproducibility evidence, not an envelope gate",
        },
        "runs": [
            {
                "stage": stage,
                "repeat": repeat,
                "record": f".cache/layoutformerpp/post-merge/{args.attempt}/{stage}{'-repeat-2' if repeat == 2 else ''}/record.json",
                "freeze": f".cache/layoutformerpp/post-merge/{args.attempt}/{stage}{'-repeat-2' if repeat == 2 else ''}/record.freeze",
                "runtime_freeze_sha256": runtime_freeze_sha256,
            }
            for stage, repeat, _ in runs
        ],
    }
    write_new(
        run_root / "launch_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    statuses: list[int] = []
    for stage, repeat, selector in runs:
        name = stage if repeat == 1 else f"{stage}-repeat-2"
        output_dir = run_root / name
        output_dir.mkdir()
        started_at = now()
        command = [
            str(runtime_python),
            "-m",
            "pytest",
            "-p",
            "training_parity_record_plugin",
            TEST,
            "-m",
            "vendor_parity and training",
            *selector,
            "-rs",
            "-q",
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "PARITY_REQUIRE": "1",
            "LAYOUTFORMERPP_AUDIT_VENV": runtime_relative,
            "PYTHONPATH": f"{SCRIPT_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "LAYOUTFORMERPP_RECORD_DIR": str(output_dir),
            "LAYOUTFORMERPP_RUN_STAGE": stage,
            "LAYOUTFORMERPP_RUN_REPEAT": str(repeat),
            "LAYOUTFORMERPP_RUN_STARTED_AT": started_at,
            "LAYOUTFORMERPP_SOURCE_COMMIT": source_commit,
            "LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256": runtime_freeze_sha256,
            "LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE": "models/layoutformerpp/scripts/audit_runtime.freeze",
            "LAYOUTFORMERPP_REPORT_DATA_ROOT": data_root,
            "LAYOUTFORMERPP_REPORT_REPO_ROOT": str(REPO_ROOT),
        }
        command_record = {
            "stage": stage,
            "repeat": repeat,
            "started_at": started_at,
            "command": [
                '"$LAYOUTFORMERPP_AUDIT_VENV/bin/python"',
                *command[1:],
            ],
            "data_root": "<your data root>",
            "source_commit": source_commit,
            "runtime_interpreter": f"{runtime_relative}/bin/python",
            "runtime_freeze": "models/layoutformerpp/scripts/audit_runtime.freeze",
            "runtime_freeze_sha256": runtime_freeze_sha256,
        }
        write_new(
            output_dir / "run_started.json",
            json.dumps(command_record, indent=2, sort_keys=True) + "\n",
        )
        with tempfile.NamedTemporaryFile(
            prefix="layoutformerpp-", suffix=".log", delete=False
        ) as raw_stream:
            raw_log = Path(raw_stream.name)
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=raw_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log_path = output_dir / "pytest.log"
        write_new(
            log_path,
            redact(raw_log.read_text(encoding="utf-8", errors="replace"), data_root),
        )
        raw_log.unlink(missing_ok=True)
        write_new(
            log_path.with_name("pytest.log.sha256"),
            f"sha256:{sha256(log_path)} pytest.log\n",
        )
        statuses.append(completed.returncode)
        print(
            f"{stage} repeat={repeat} exit={completed.returncode} record={output_dir / 'record.json'}",
            flush=True,
        )
    return max(statuses, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
