#!/usr/bin/env -S uv run --package layoutformerpp
"""Build the reproducible LayoutFormer++ parity runtime."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.utils import parse_wheel_filename


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = (
    REPO_ROOT / "models/layoutformerpp/scripts/audit_runtime_requirements.txt"
)
EXPECTED_FREEZE = REPO_ROOT / "models/layoutformerpp/scripts/audit_runtime.freeze"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


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


def normalized_runtime_freeze(python: Path) -> str:
    raw = subprocess.check_output(
        ["uv", "pip", "freeze", "--no-config", "--python", str(python)],
        cwd=REPO_ROOT,
        text=True,
    )
    return normalized_freeze(raw)


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
    if runtime_dir.exists():
        raise SystemExit(f"refusing to reuse existing runtime directory: {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "uv",
            "venv",
            "--no-project",
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
            "-r",
            str(REQUIREMENTS),
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "--extra-index-url",
            "https://pypi.org/simple",
        ]
    )
    normalized = normalized_runtime_freeze(python)
    expected = EXPECTED_FREEZE.read_text(encoding="utf-8")
    if normalized != expected:
        raise SystemExit(
            "fresh runtime freeze differs from models/layoutformerpp/scripts/audit_runtime.freeze"
        )
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
