#!/usr/bin/env bash

# @file models/lace/scripts/rebuild_audit_runtime.sh
# @brief Rebuild the audited LACE runtime from a recorded freeze.
# @description
#   Generates the pinned dependency inputs, installs the hash-checked CUDA
#   wheels and LACE workspace editables, and records a canonical distribution
#   diff against the cited freeze.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s <recorded-freeze> <wheel-root> <venv> <output-root>\n' "$0" >&2
  exit 2
fi

recorded_freeze=$1
wheel_root=$2
venv=$3
output_root=$4
runtime_requirements="$output_root/recorded-runtime-requirements.txt"
wheel_requirements="$output_root/wheel-hash-requirements.txt"
rebuilt_freeze="$output_root/pip-freeze.txt"
distribution_diff="$output_root/distribution-diff.txt"

mkdir -p "$output_root"

python - "$recorded_freeze" "$runtime_requirements" "$wheel_requirements" "$wheel_root" <<'PY'
from pathlib import Path
import sys

recorded_freeze, runtime_requirements, wheel_requirements, wheel_root = map(
    Path, sys.argv[1:]
)

runtime_lines = []
for raw_line in recorded_freeze.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("-e ") or line.startswith("torch @ "):
        continue
    if line.startswith("torchvision @ "):
        continue
    if line.startswith("--"):
        continue
    runtime_lines.append(line)

wheel_specs = (
    (
        "torch",
        "torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed",
    ),
    (
        "torchvision",
        "torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb",
    ),
)
wheel_lines = ["--no-index"]
for distribution, filename, digest in wheel_specs:
    wheel_path = wheel_root / filename
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)
    wheel_lines.extend(
        [
            f"{distribution} @ {wheel_path.as_uri()} \\",
            f"    --hash=sha256:{digest}",
        ]
    )

runtime_requirements.write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")
wheel_requirements.write_text("\n".join(wheel_lines) + "\n", encoding="utf-8")
PY

UV_NO_CONFIG=1 UV_FROZEN=1 uv venv "$venv" --python 3.11 --clear
UV_NO_CONFIG=1 UV_FROZEN=1 uv pip sync --python "$venv/bin/python" "$runtime_requirements"
UV_NO_CONFIG=1 UV_FROZEN=1 uv pip install --python "$venv/bin/python" --require-hashes --no-deps -r "$wheel_requirements"
UV_NO_CONFIG=1 UV_FROZEN=1 uv pip install --python "$venv/bin/python" --no-deps -e models/lace -e lib/laygen -e lib/traingen -e lib/traingen-parity
UV_NO_CONFIG=1 UV_FROZEN=1 uv pip freeze --python "$venv/bin/python" > "$rebuilt_freeze"

python - "$recorded_freeze" "$rebuilt_freeze" "$distribution_diff" <<'PY'
from pathlib import Path
import re
import sys

recorded_freeze, rebuilt_freeze, distribution_diff = map(Path, sys.argv[1:])


def canonical(line: str) -> str:
    line = line.strip()
    if line.startswith("-e file://"):
        path = line.removeprefix("-e file://")
        if path.endswith("/models/lace"):
            return "lace (editable)"
        if path.endswith("/lib/laygen"):
            return "laygen (editable)"
        if path.endswith("/lib/traingen"):
            return "traingen (editable)"
        if path.endswith("/lib/traingen-parity"):
            return "traingen-parity (editable)"
        if path.endswith("/models/ds-gan"):
            return "ds_gan (editable)"
        if path.endswith("/lib/posgen"):
            return "posgen (editable)"
        return re.sub(r"^.*?/([^/]+)$", r"\1 (editable)", path)
    if line.startswith("torch @ "):
        return "torch==2.8.0+cu128"
    if line.startswith("torchvision @ "):
        return "torchvision==0.23.0+cu128"
    return line


def distributions(path: Path) -> set[str]:
    return {canonical(line) for line in path.read_text(encoding="utf-8").splitlines() if line}


recorded = distributions(recorded_freeze)
rebuilt = distributions(rebuilt_freeze)
only_recorded = sorted(recorded - rebuilt)
only_rebuilt = sorted(rebuilt - recorded)
lines = ["Only in cited freeze:", *[f"- {line}" for line in only_recorded]]
lines.extend(["Only in rebuilt freeze:", *[f"+ {line}" for line in only_rebuilt]])
distribution_diff.write_text("\n".join(lines) + "\n", encoding="utf-8")
if only_recorded != ["ds_gan (editable)", "posgen (editable)"] or only_rebuilt:
    raise SystemExit(f"unexpected distribution diff: {lines}")
PY

printf 'recorded freeze sha256: '
sha256sum "$recorded_freeze"
printf 'rebuilt freeze: %s\n' "$rebuilt_freeze"
printf 'rebuilt freeze sha256: '
sha256sum "$rebuilt_freeze"
printf 'distribution diff: %s\n' "$distribution_diff"
cat "$distribution_diff"
