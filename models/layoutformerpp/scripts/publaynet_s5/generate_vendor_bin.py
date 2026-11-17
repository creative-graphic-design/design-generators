"""Generate the environment-configured Python shim used by vendor launch scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


SHIM = """#!/usr/bin/env bash
set -euo pipefail
runtime_python="${PACKAGE_ENV:?}"
runtime_python="$runtime_python/bin/python"
bootstrap="${LAYOUTFORMERPP_VENDOR_BOOTSTRAP:?}"
if [[ "${1:-}" == "main.py" ]]; then
    shift
    exec "$runtime_python" "$bootstrap" "$@"
fi
exec "$runtime_python" "$@"
"""


def generate(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "python"
    shim.write_text(SHIM, encoding="utf-8")
    shim.chmod(0o755)
    return shim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    generate(args.directory)
    print("vendor Python shim generated")


if __name__ == "__main__":
    main()
