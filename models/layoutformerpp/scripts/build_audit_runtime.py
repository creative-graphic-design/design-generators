#!/usr/bin/env -S uv run --package layoutformerpp
"""Build the reproducible LayoutFormer++ parity runtime."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPSPEED_PIN = "0.5.10"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _normalized_freeze(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.replace(str(REPO_ROOT), "<repo root>")
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-dir",
        default=".cache/layoutformerpp/runtime/cu126-py311",
        help="Repository-relative directory for the audited Python runtime.",
    )
    args = parser.parse_args()
    runtime_dir = Path(args.runtime_dir)
    if runtime_dir.is_absolute() or ".." in runtime_dir.parts:
        raise SystemExit("--runtime-dir must be repository-relative")
    runtime_dir = REPO_ROOT / runtime_dir
    python = runtime_dir / "bin/python"
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "uv",
            "venv",
            "--no-project",
            "--allow-existing",
            "--python",
            "3.11",
            str(runtime_dir),
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--torch-backend",
            "cu126",
            "-e",
            "lib/laygen",
            "-e",
            "lib/traingen",
            "-e",
            "lib/traingen-parity",
            "-e",
            "models/layoutformerpp[training,vendor]",
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-build-isolation",
            f"deepspeed=={DEEPSPEED_PIN}",
        ]
    )
    _run(
        [
            str(python),
            "-c",
            "import importlib.metadata; assert importlib.metadata.version('deepspeed') == '0.5.10'",
        ]
    )
    freeze = subprocess.check_output(
        ["uv", "pip", "freeze", "--no-config", "--python", str(python)],
        cwd=REPO_ROOT,
        text=True,
    )
    normalized = _normalized_freeze(freeze)
    freeze_path = runtime_dir / "pip-freeze.txt"
    freeze_path.write_text(normalized, encoding="utf-8")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    (runtime_dir / "pip-freeze.sha256").write_text(
        f"{digest}  pip-freeze.txt\n", encoding="utf-8"
    )
    print(f"runtime freeze: {freeze_path.relative_to(REPO_ROOT)}")
    print(f"runtime freeze sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
