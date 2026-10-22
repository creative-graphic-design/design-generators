#!/usr/bin/env bash

# @file scripts/build_layout_corrector_audit_runtime.sh
# @brief Rebuild the pinned Layout-Corrector audit environment for this checkout.
# @description
#   Reads the unchanged source environment, redirects its editable installs to
#   this worktree, and verifies the frozen distributions and retained CUDA wheels.
#   Requires LAYOUT_CORRECTOR_SOURCE_RUNTIME and LAYOUT_CORRECTOR_TORCHVISION_WHEEL.
# @stdout The verified freeze SHA-256 and runtime path.
# @example
#   LAYOUT_CORRECTOR_SOURCE_RUNTIME=<source-runtime> \
#     LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<retained-wheel> \
#     scripts/build_layout_corrector_audit_runtime.sh

set -euo pipefail

if [[ -z "${LAYOUT_CORRECTOR_SOURCE_RUNTIME:-}" || -z "${LAYOUT_CORRECTOR_TORCHVISION_WHEEL:-}" ]]; then
  echo "LAYOUT_CORRECTOR_SOURCE_RUNTIME and LAYOUT_CORRECTOR_TORCHVISION_WHEEL are required" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_python="$LAYOUT_CORRECTOR_SOURCE_RUNTIME/bin/python"
torchvision_wheel="$LAYOUT_CORRECTOR_TORCHVISION_WHEEL"
cache_root="$repo_root/.cache/layout-corrector"
target="$cache_root/runtime"

if [[ ! -x "$source_python" ]]; then
  echo "source runtime Python is missing: $source_python" >&2
  exit 1
fi
if [[ ! -f "$torchvision_wheel" ]]; then
  echo "retained torchvision wheel is missing: $torchvision_wheel" >&2
  exit 1
fi
torchvision_wheel="$("$source_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$torchvision_wheel")"

mkdir -p "$cache_root"
exec {lock_fd}>"$cache_root/.runtime-build.lock"
if ! flock -n "$lock_fd"; then
  echo "another audit runtime build is already running" >&2
  exit 1
fi
if [[ -e "$target" ]]; then
  echo "target runtime already exists; refusing to replace it: $target" >&2
  exit 1
fi

source_version="$("$source_python" --version 2>&1)"
if [[ "$source_version" != "Python 3.11.15" ]]; then
  echo "expected source Python 3.11.15, found $source_version" >&2
  exit 1
fi

build_inputs="$(mktemp -d "$cache_root/.runtime-build.XXXXXX")"
target_created=0
cleanup() {
  if [[ "$target_created" == "1" ]]; then
    rm -rf "$target"
  fi
  rm -rf "$build_inputs"
}
trap cleanup EXIT
source_freeze="$build_inputs/source-uv-pip-freeze.txt"
expected_freeze="$build_inputs/expected-uv-pip-freeze.txt"
base_requirements="$build_inputs/base-requirements.txt"
torchvision_wheelhouse="$build_inputs/torchvision-wheelhouse"
normalized_target_freeze="$build_inputs/normalized-target-uv-pip-freeze.txt"

mkdir -p "$torchvision_wheelhouse"
ln -s "$torchvision_wheel" "$torchvision_wheelhouse/$(basename "$torchvision_wheel")"

uv pip freeze --python "$source_python" > "$source_freeze"
source_freeze_sha256="$(sha256sum "$source_freeze" | awk '{print $1}')"
if [[ "$source_freeze_sha256" != "3fdc59949e80293c858d8eba525ad4cd46713dd8e4ca31e8c9fab428e46cefdf" ]]; then
  echo "source freeze mismatch: expected 3fdc59949e80293c858d8eba525ad4cd46713dd8e4ca31e8c9fab428e46cefdf, found $source_freeze_sha256" >&2
  exit 1
fi

torch_requirement="$(awk '$1 == "torch" && $2 == "@" { print; exit }' "$source_freeze")"
torchvision_requirement="$(awk '/^torchvision==/ { print; exit }' "$source_freeze")"
if [[ -z "$torch_requirement" || -z "$torchvision_requirement" ]]; then
  echo "source freeze must pin torch by retained wheel and torchvision by version" >&2
  exit 1
fi
torch_uri="${torch_requirement#torch @ }"
torch_path="$("$source_python" -c 'from pathlib import Path; from urllib.parse import unquote, urlparse; import sys; uri = urlparse(sys.argv[1]); assert uri.scheme == "file"; print(unquote(uri.path))' "$torch_uri")"

check_wheel() {
  local package="$1"
  local wheel="$2"
  local expected="$3"
  local observed

  observed="$(sha256sum "$wheel" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "$package wheel SHA-256 mismatch: expected $expected, found $observed" >&2
    exit 1
  fi
}

check_wheel torch "$torch_path" 039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed
check_wheel torchvision "$torchvision_wheel" 93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb

editable_paths=(
  tools/devharness
  models/lace
  lib/laygen
  models/layout-corrector
  models/layout-dm
  lib/traingen
  lib/traingen-parity
)
editable_count=0
: > "$expected_freeze"
: > "$base_requirements"
while IFS= read -r requirement || [[ -n "$requirement" ]]; do
  if [[ "$requirement" == -e\ file://* ]]; then
    old_uri="${requirement#-e }"
    old_path="$("$source_python" -c 'from urllib.parse import unquote, urlparse; import sys; uri = urlparse(sys.argv[1]); assert uri.scheme == "file"; print(unquote(uri.path))' "$old_uri")"
    matched_path=""
    for relative_path in "${editable_paths[@]}"; do
      if [[ "$old_path" == */"$relative_path" ]]; then
        matched_path="$relative_path"
        break
      fi
    done
    if [[ -z "$matched_path" || ! -f "$repo_root/$matched_path/pyproject.toml" ]]; then
      echo "unexpected editable source in frozen runtime: $requirement" >&2
      exit 1
    fi
    editable_count=$((editable_count + 1))
    new_uri="$("$source_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).as_uri())' "$repo_root/$matched_path")"
    printf -- '-e %s\n' "$new_uri" >> "$expected_freeze"
    continue
  fi

  printf '%s\n' "$requirement" >> "$expected_freeze"
  if [[ "$requirement" != "torch @ "* && "$requirement" != torchvision==* ]]; then
    printf '%s\n' "$requirement" >> "$base_requirements"
  fi
done < "$source_freeze"

if [[ "$editable_count" -ne "${#editable_paths[@]}" ]]; then
  echo "expected ${#editable_paths[@]} editable packages, found $editable_count" >&2
  exit 1
fi

target_created=1
uv venv --no-project --python "$source_python" "$target"
target_python="$target/bin/python"
uv pip install --python "$target_python" --no-deps --requirements "$base_requirements"
uv pip install --python "$target_python" --no-deps "$torch_uri"
uv pip install --python "$target_python" --no-deps --no-index --find-links "$torchvision_wheelhouse" "$torchvision_requirement"
for relative_path in "${editable_paths[@]}"; do
  uv pip install --python "$target_python" --no-deps --editable "$repo_root/$relative_path"
done

target_version="$("$target_python" --version 2>&1)"
if [[ "$target_version" != "$source_version" ]]; then
  echo "runtime Python mismatch: expected $source_version, found $target_version" >&2
  exit 1
fi

uv pip freeze --python "$target_python" > "$target/uv-pip-freeze.txt"
if [[ "$torchvision_requirement" != "torchvision==0.23.0" ]]; then
  echo "expected frozen torchvision requirement torchvision==0.23.0, found $torchvision_requirement" >&2
  exit 1
fi
if ! awk '
  $0 == "torchvision==0.23.0+cu128" {
    print "torchvision==0.23.0"
    replacements += 1
    next
  }
  { print }
  END { if (replacements != 1) exit 1 }
' "$target/uv-pip-freeze.txt" > "$normalized_target_freeze"; then
  echo "new runtime freeze must contain exactly one torchvision==0.23.0+cu128 line" >&2
  exit 1
fi
if ! cmp -s "$expected_freeze" "$normalized_target_freeze"; then
  diff -u "$expected_freeze" "$normalized_target_freeze" || true
  echo "new runtime freeze differs beyond editable worktree paths and the torchvision local-version label" >&2
  exit 1
fi

"$target_python" -c 'import torch, torchvision; assert torch.__version__ == "2.8.0+cu128", torch.__version__; assert torchvision.__version__ == "0.23.0+cu128", torchvision.__version__'
printf 'source_freeze_sha256=%s\n' "$source_freeze_sha256"
printf 'runtime_freeze_sha256=%s\n' "$(sha256sum "$target/uv-pip-freeze.txt" | awk '{print $1}')"
printf 'torch_wheel_sha256=%s\n' "$(sha256sum "$torch_path" | awk '{print $1}')"
printf 'torchvision_wheel_sha256=%s\n' "$(sha256sum "$torchvision_wheel" | awk '{print $1}')"
printf 'runtime=%s\neditable_packages=%s\nfreeze_lines_match_except_editables_and_torchvision_local_version=true\n' "$target" "$editable_count"
target_created=0
