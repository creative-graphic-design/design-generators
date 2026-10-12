#!/usr/bin/env bash

# @file models/pixel-vae/scripts/run_cpu_parity_acceptance.sh
# @brief Preflight PixelVAE input selections, then run the distinct-input CPU acceptance sequence.
# @description Runs selector tests, member coverage tests, and the pinned archive audit, then records every real input selection before report-only numerical diagnostics, calibration, limit freezing, and held-out parity.
# @example
#   PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<sha> bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
cd "$repo_root"

export PIXELVAE_PARITY_DIR="${PIXELVAE_PARITY_DIR:-.cache/pixel-vae/parity}"
export PARITY_REQUIRE=1
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0
export CRELLO_V1_AUDIT_PATH="$PIXELVAE_PARITY_DIR/archive-dimensions.json"

archive_path=.cache/pixel-vae/crello-dataset-v1.zip
tfrecord_dir=.cache/pixel-vae/crello-v1-tfrecords
if [[ -e "$PIXELVAE_PARITY_DIR" || -e "$archive_path" || -e "$tfrecord_dir" ]]; then
  printf 'Refusing to reuse existing run outputs or private inputs: %s, %s, or %s\n' \
    "$PIXELVAE_PARITY_DIR" "$archive_path" "$tfrecord_dir" >&2
  exit 2
fi

selection_test_command=(
  uv run --package pixel-vae --extra vendor --extra parity --with pytest
  pytest models/pixel-vae/tests/vendor_parity -m vendor_parity -q -k 'selection or plan_mode'
)
"${selection_test_command[@]}"

scripts/run_member_tests.sh models/pixel-vae
scripts/run_member_tests.sh lib/traingen

mkdir -p "$PIXELVAE_PARITY_DIR" "$(dirname "$archive_path")"
curl --fail --location --retry 3 \
  --output "$archive_path" \
  https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip

uv run --package pixel-vae --extra vendor --extra parity \
  models/pixel-vae/scripts/audit_crello_png_dimensions.py \
  --archive "$archive_path" \
  --output "$CRELLO_V1_AUDIT_PATH" \
  --extract-dir "$tfrecord_dir"

uv run --no-project python - "$CRELLO_V1_AUDIT_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
invalid = {
    "non_256_examples": report["non_256_examples"],
    "invalid_png_counts": report["invalid_png_counts"],
}
if invalid["non_256_examples"] or invalid["invalid_png_counts"]:
    print(json.dumps(invalid, indent=2, sort_keys=True))
    raise SystemExit("Crello v1 contains document PNGs outside the 256x256 RGBA contract")
PY

parity_command=(
  uv run --package pixel-vae --extra vendor --extra parity --with pytest
  pytest models/pixel-vae/tests/vendor_parity -m vendor_parity -q
)

PIXELVAE_PARITY_MODE=plan "${parity_command[@]}" -k test_pixel_vae_cpu_stages

PIXELVAE_PARITY_MODE=diagnostic "${parity_command[@]}"

for repeat in 1 2 3; do
  PIXELVAE_PARITY_MODE=calibration PIXELVAE_PARITY_REPEAT="$repeat" \
    "${parity_command[@]}"
done

uv run --package pixel-vae --extra parity \
  models/pixel-vae/scripts/calibrate_cpu_limits.py \
  --input-dir "$PIXELVAE_PARITY_DIR/calibration" \
  --output "$PIXELVAE_PARITY_DIR/calibration/limits.json" \
  --lower-bounds models/pixel-vae/parity-limit-floors.json

PIXELVAE_PARITY_MODE=heldout "${parity_command[@]}"
