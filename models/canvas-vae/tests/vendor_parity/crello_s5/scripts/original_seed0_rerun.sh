#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
OUT="${CANVAS_VAE_ORIGINAL_SEED0_RERUN_OUTPUT:?set CANVAS_VAE_ORIGINAL_SEED0_RERUN_OUTPUT}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
NEED_MIB=9831
MARGIN_MIB=4096
ENV="${CANVAS_VAE_ORIGINAL_ENV:?set CANVAS_VAE_ORIGINAL_ENV}"
TF_SITE="${CANVASVAE_TF_GPU_SITE:?set CANVASVAE_TF_GPU_SITE}"
INPUTS="${CANVAS_VAE_S5_INPUTS:-$ROOT/.cache/canvas-vae/s5-path-a/inputs}"
GUARD="$ROOT/models/canvas-vae/scripts/source_tree_guard.py"
GUARD_MANIFEST="$RUN_ROOT/guards/original-seed-0.json"
TRAIN_HELPER="$TOOL_ROOT/helpers/train_original_best.py"
EVAL_HELPER="$TOOL_ROOT/helpers/evaluate_original_best.py"

cd "$ROOT"
unset LD_LIBRARY_PATH PYTHONPATH CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
export CUDA_VISIBLE_DEVICES="$GPU" NVIDIA_TF32_OVERRIDE=0
export TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$TF_SITE:$ROOT/models/canvas-vae/scripts:$ROOT/models/canvas-vae/src:$ROOT/lib/laygen/src:$ROOT/lib/traingen/src:$ROOT/lib/traingen-parity/src:$ROOT/vendor/canvas-vae/src/canvas-vae"
export CANVASVAE_TF_GPU_SITE="$TF_SITE" CANVAS_VAE_S4_TRAIN_STREAM=1
export CANVAS_VAE_PIP_FREEZE_FILE="$RUN_ROOT/stack/tensorflow-pip-freeze.txt"

for required in "$ENV/bin/python" "$INPUTS/original-data" "$GUARD" "$TRAIN_HELPER" "$EVAL_HELPER"; do
  [[ -e "$required" ]] || { printf 'missing required input or executable: %s\n' "$required" >&2; exit 1; }
done
[[ "$OUT" == "$RUN_ROOT/original/"* ]] || { printf 'rerun output must be inside the run directory\n' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { printf 'refusing to reuse an existing seed-0 rerun output\n' >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { printf 'source commit changed before original seed-0 rerun\n' >&2; exit 1; }
mkdir -p "$OUT"
trap 'rc=$?; printf "exit_code=%s\n" "$rc" > "$OUT/exit.status"' EXIT

exec 9>"$RUN_ROOT/seed-start.lock"
flock -x 9
while true; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d '[:space:]')"
  if (( free >= NEED_MIB + MARGIN_MIB )); then
    printf '%s original seed=0-rerun free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
    break
  fi
  printf '%s original seed=0-rerun waiting free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
  sleep 10
done
"$ENV/bin/python" "$GUARD" verify --source-root "$ROOT" --manifest "$GUARD_MANIFEST" --seed 1 > "$OUT/source-tree-guard.json"
flock -u 9
sha256sum "$EVAL_HELPER" > "$OUT/evaluator.sha256"
"$ENV/bin/python" - "$OUT/rerun.json" "$OUT/source-tree-guard.json" "$ROOT" "$EVAL_HELPER" "$TOOL_ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, guard_result_path, root_path, evaluator_path, tool_root = map(
    Path, sys.argv[1:]
)
guard = json.loads(guard_result_path.read_text())
evaluator_sha = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
commit = subprocess.check_output(
    ["git", "-C", str(root_path), "rev-parse", "HEAD"], text=True
).strip()
manifest_path.write_text(
    json.dumps(
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "training_seed": 0,
            "source_commit": commit,
            "source_tree_sha256": guard["source_tree_sha256"],
            "guard_result": str(guard_result_path.relative_to(root_path)),
            "launch_wrapper_sha256": hashlib.sha256(
                (tool_root / "scripts" / "original_seed0_rerun.sh").read_bytes()
            ).hexdigest(),
            "training_wrapper_sha256": hashlib.sha256(
                (tool_root / "helpers" / "train_original_best.py").read_bytes()
            ).hexdigest(),
            "evaluator_sha256": evaluator_sha,
            "training_command": "train_original_best.py --data-dir original-data --job-dir training --seed 0",
            "evaluation_command": "evaluate_original_best.py --checkpoint training/checkpoints/best.ckpt --output-dir evaluation --seeds 0 1 2",
        },
        indent=2,
    )
    + "\n"
)
PY
"$ENV/bin/python" "$TRAIN_HELPER" --data-dir "$INPUTS/original-data" --job-dir "$OUT/training" --seed 0 > "$OUT/training.log" 2>&1
unset CUDA_VISIBLE_DEVICES
"$ENV/bin/python" "$EVAL_HELPER" --data-dir "$INPUTS/original-data" --checkpoint "$OUT/training/checkpoints/best.ckpt" --output-dir "$OUT/evaluation" --seeds 0 1 2 > "$OUT/evaluation.log" 2>&1
printf '%s original seed=0-rerun training_and_evaluation_complete\n' "$(date -Is)" >> "$RUN_ROOT/progress-reruns.log"
