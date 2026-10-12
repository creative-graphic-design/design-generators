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


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
TEST = (
    "models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py"
)
DEFAULT_RUNTIME = ".cache/layoutformerpp/runtime/cu126-py311"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_once(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


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


def _runtime_paths() -> tuple[str, Path, Path]:
    relative = os.environ.get("LAYOUTFORMERPP_AUDIT_VENV", DEFAULT_RUNTIME)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit("LAYOUTFORMERPP_AUDIT_VENV must be repository-relative")
    runtime = REPO_ROOT / path
    return relative.rstrip("/"), runtime / "bin/python", runtime / "pip-freeze.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt-3")
    args = parser.parse_args()
    if args.attempt != "attempt-3":
        raise SystemExit("this runner is reserved for the approved attempt-3 campaign")
    data_root = os.environ.get("LAYOUTFORMERPP_PARITY_DATA_ROOT")
    if not data_root or not Path(data_root).is_dir():
        raise SystemExit(
            "LAYOUTFORMERPP_PARITY_DATA_ROOT must name the approved processed-data root"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise SystemExit("attempt-3 requires CUDA_VISIBLE_DEVICES=4")
    runtime_relative, runtime_python, runtime_freeze = _runtime_paths()
    if not runtime_python.is_file() or not runtime_freeze.is_file():
        raise SystemExit("audited runtime or its freeze file is missing")
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
    runtime_freeze_sha256 = sha256(runtime_freeze)
    run_root = REPO_ROOT / ".cache/layoutformerpp/post-merge" / args.attempt
    preflight_environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "4"}
    smoke = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "import torch; assert torch.cuda.is_available(); x=torch.ones((2,2), device='cuda:0'); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x @ x).tolist())",
        ],
        cwd=REPO_ROOT,
        env=preflight_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    preflight = {
        "runtime_freeze": f"{runtime_relative}/pip-freeze.txt",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "cuda_visible_devices": "4",
        "command": "import torch; assert torch.cuda.is_available(); tiny matmul on cuda:0",
        "returncode": smoke.returncode,
        "stdout": redact(smoke.stdout, data_root),
        "stderr": redact(smoke.stderr, data_root),
    }
    write_once(
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
        "runtime": f"{runtime_relative}/pip-freeze.txt",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "cuda_visible_devices": "4",
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
    write_once(
        run_root / "launch_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    statuses: list[int] = []
    for stage, repeat, selector in runs:
        name = stage if repeat == 1 else f"{stage}-repeat-2"
        output_dir = run_root / name
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
            "CUDA_VISIBLE_DEVICES": "4",
            "PARITY_REQUIRE": "1",
            "LAYOUTFORMERPP_AUDIT_VENV": runtime_relative,
            "PYTHONPATH": f"{SCRIPT_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "LAYOUTFORMERPP_RECORD_DIR": str(output_dir),
            "LAYOUTFORMERPP_RUN_STAGE": stage,
            "LAYOUTFORMERPP_RUN_REPEAT": str(repeat),
            "LAYOUTFORMERPP_RUN_STARTED_AT": started_at,
            "LAYOUTFORMERPP_SOURCE_COMMIT": source_commit,
            "LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256": runtime_freeze_sha256,
            "LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE": f"{runtime_relative}/pip-freeze.txt",
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
            "runtime_freeze": f"{runtime_relative}/pip-freeze.txt",
            "runtime_freeze_sha256": runtime_freeze_sha256,
        }
        write_once(
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
        write_once(
            log_path,
            redact(raw_log.read_text(encoding="utf-8", errors="replace"), data_root),
        )
        raw_log.unlink(missing_ok=True)
        write_once(
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
