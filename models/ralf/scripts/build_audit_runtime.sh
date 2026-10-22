#!/usr/bin/env bash

# @file models/ralf/scripts/build_audit_runtime.sh
# @brief Build the RALF cu128 audit runtime from the frozen workspace inputs.
# @description
#   Keeps PyTorch and its CUDA wheels on the matching cu128 index, installs the
#   package editables, and records a path-independent freeze for the parity
#   runner.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
runtime_root=${1:-"$repo_root/.cache/ralf/runtime/audit-cu128-cu12"}
output_root=${2:-"${runtime_root}-records"}
source_lock_commit=fc0379bb49e58143f8718711bbf6f701bbfe9ab3
pytorch_index_url=https://download.pytorch.org/whl/cu128
torch_wheel="$pytorch_index_url/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
torchvision_wheel="$pytorch_index_url/torchvision-0.23.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
triton_wheel="$pytorch_index_url/../triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

mkdir -p "$output_root"
output_root=$(cd -- "$output_root" && pwd)
requirements="$output_root/workspace-requirements.txt"
raw_requirements="$output_root/workspace-requirements-without-cuda.txt"
raw_freeze="$output_root/runtime-freeze-raw.txt"
normalized_freeze="$output_root/runtime-freeze.txt"
if [[ "$(realpath -m "$runtime_root")" == "$(realpath -m "$output_root")" ]]; then
    printf 'runtime root and output root must be different directories\n' >&2
    exit 2
fi
if ! git -C "$repo_root" cat-file -e "$source_lock_commit^{commit}"; then
    printf 'source lock commit is unavailable: %s\n' "$source_lock_commit" >&2
    exit 2
fi

lock_source_root=$(mktemp -d "$output_root/lock-source.XXXXXX")
rmdir "$lock_source_root"
git -C "$repo_root" worktree add --detach "$lock_source_root" "$source_lock_commit"
trap 'git -C "$repo_root" worktree remove "$lock_source_root" >/dev/null' EXIT

run_uv() {
    local attempt
    for attempt in 1 2 3; do
        if UV_NO_CONFIG=1 UV_FROZEN=1 uv "$@"; then
            return 0
        fi
        if (( attempt < 3 )); then
            sleep $((attempt * 5))
        fi
    done
    return 1
}

run_uv --directory "$lock_source_root" export \
  --frozen \
  --package ralf \
  --extra training \
  --extra vendor \
  --format requirements-txt \
  --output-file "$requirements" > /dev/null
printf 'dependency lock commit: %s\n' "$source_lock_commit"

python - "$requirements" "$raw_requirements" <<'PY'
from pathlib import Path
import re
import sys

source, target = map(Path, sys.argv[1:])
excluded = {"torch", "torchvision", "torchaudio", "triton"}
output: list[str] = []
skip = False
for raw_line in source.read_text(encoding="utf-8").splitlines(keepends=True):
    stripped = raw_line.strip()
    if stripped.startswith("-e "):
        skip = True
        continue
    if (
        stripped
        and not stripped.startswith("#")
        and not stripped.startswith("--")
        and not raw_line[0].isspace()
    ):
        name = re.split(r"[<=>!~;\s\[]", stripped, maxsplit=1)[0].lower()
        skip = name in excluded or name.startswith("nvidia-")
    if not skip:
        output.append(raw_line)

target.write_text("".join(output), encoding="utf-8")
PY

run_uv venv "$runtime_root" --python 3.11 --clear --force
run_uv pip install \
  --python "$runtime_root/bin/python" \
  --index-url "$pytorch_index_url" \
  "$torch_wheel" \
  "$torchvision_wheel" \
  "$triton_wheel"
run_uv pip install \
  --python "$runtime_root/bin/python" \
  --require-hashes \
  --no-deps \
  -r "$raw_requirements"
run_uv pip install \
  --python "$runtime_root/bin/python" \
  pytest==9.1.1 \
  pluggy==1.6.0
run_uv pip install \
  --python "$runtime_root/bin/python" \
  --no-deps \
  --editable "$repo_root/lib/laygen" \
  --editable "$repo_root/lib/posgen" \
  --editable "$repo_root/lib/traingen" \
  --editable "$repo_root/lib/traingen-parity" \
  --editable "$repo_root/models/ralf"

run_uv pip freeze --python "$runtime_root/bin/python" > "$raw_freeze"

python - "$raw_freeze" "$normalized_freeze" "$repo_root" <<'PY'
from pathlib import Path
from urllib.parse import unquote, urlparse
import re
import sys

raw_path, normalized_path, repo_root = map(Path, sys.argv[1:])
repo_root = repo_root.resolve()
lines: list[str] = []
for raw_line in raw_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith("-e "):
        value = line[3:].strip()
        path = Path(unquote(urlparse(value).path)) if value.startswith("file://") else Path(value)
        path = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = "<external>"
        lines.append(f"-e <repo root>/{relative}")
        continue
    match = re.match(r"^(torch(?:vision)?) @ (.+)$", line)
    if match:
        name, reference = match.groups()
        wheel_name = Path(unquote(urlparse(reference).path)).name
        version_match = re.match(rf"{name}-(.+?)-cp", wheel_name)
        if version_match is None:
            raise SystemExit(f"cannot normalize {line!r}")
        lines.append(f"{name}=={version_match.group(1)}")
        continue
    lines.append(line)

normalized_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

"$runtime_root/bin/python" - <<'PY'
from importlib import metadata

expected_torch = "2.8.0+cu128"
if metadata.version("torch") != expected_torch:
    raise SystemExit(f"unexpected torch version: {metadata.version('torch')}")
if metadata.version("torchvision") != "0.23.0+cu128":
    raise SystemExit(f"unexpected torchvision version: {metadata.version('torchvision')}")

packages = {
    (dist.metadata.get("Name") or "").lower(): dist.version
    for dist in metadata.distributions()
    if (dist.metadata.get("Name") or "").lower().startswith("nvidia-")
}
if "nvidia-cudnn-cu12" not in packages:
    raise SystemExit("nvidia-cudnn-cu12 is missing")
bad = [
    f"{name}=={version}"
    for name, version in packages.items()
    if not name.endswith("-cu12") or "-cu13" in name or (name == "nvidia-cuda-runtime" and version.startswith("13."))
]
if bad:
    raise SystemExit(f"CUDA package family mismatch: {sorted(bad)}")
PY

printf 'normalized freeze: %s\n' "$normalized_freeze"
printf 'normalized freeze sha256: '
sha256sum "$normalized_freeze"
printf 'raw freeze: %s\n' "$raw_freeze"
