#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
NEED_MIB=9831
MARGIN_MIB=4096
ENV="${CANVAS_VAE_ORIGINAL_ENV:?set CANVAS_VAE_ORIGINAL_ENV}"
TF_SITE="${CANVASVAE_TF_GPU_SITE:?set CANVASVAE_TF_GPU_SITE}"
INPUTS="${CANVAS_VAE_S5_INPUTS:-$ROOT/.cache/canvas-vae/s5-path-a/inputs}"
ORIGINAL_DATA="${CANVAS_VAE_ORIGINAL_DATA:-$INPUTS/original-data}"
GUARD="$ROOT/models/canvas-vae/scripts/source_tree_guard.py"
GUARD_MANIFEST="$RUN_ROOT/guards/original-seed-0.json"
unset LD_LIBRARY_PATH PYTHONPATH CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
export CUDA_VISIBLE_DEVICES="$GPU" NVIDIA_TF32_OVERRIDE=0
export TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$TF_SITE:$ROOT/models/canvas-vae/scripts:$ROOT/models/canvas-vae/src:$ROOT/lib/laygen/src:$ROOT/lib/traingen/src:$ROOT/lib/traingen-parity/src:$ROOT/vendor/canvas-vae/src/canvas-vae"
export CANVASVAE_TF_GPU_SITE="$TF_SITE" CANVAS_VAE_S4_TRAIN_STREAM=1
export CANVAS_VAE_PIP_FREEZE_FILE="$RUN_ROOT/stack/tensorflow-pip-freeze.txt"
exec 9>"$RUN_ROOT/seed-start.lock"

wait_for_capacity() {
  local seed="$1" free
  flock -x 9
  while true; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d '[:space:]')"
    if (( free >= NEED_MIB + MARGIN_MIB )); then
      printf '%s original seed=%s free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$seed" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
      return
    fi
    printf '%s original seed=%s waiting free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$seed" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
    sleep 10
  done
}

for required in "$ENV/bin/python" "$ORIGINAL_DATA" "$GUARD" "$TOOL_ROOT/helpers/train_original_best.py" "$TOOL_ROOT/helpers/evaluate_original_best.py"; do
  [[ -e "$required" ]] || { printf 'missing required input or executable: %s\n' "$required" >&2; exit 1; }
done
for SEED in 0 1 2; do
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { printf 'source commit changed before original seed %s\n' "$SEED" >&2; exit 1; }
  wait_for_capacity "$SEED"
  OUT="$RUN_ROOT/original/seed-$SEED"
  mkdir -p "$OUT"
  if [[ "$SEED" == 0 ]]; then
    "$ENV/bin/python" "$GUARD" record --source-root "$ROOT" --manifest "$GUARD_MANIFEST" --seed 0 > "$OUT/source-tree-guard.json"
  else
    "$ENV/bin/python" "$GUARD" verify --source-root "$ROOT" --manifest "$GUARD_MANIFEST" --seed "$SEED" > "$OUT/source-tree-guard.json"
  fi
  flock -u 9
  "$ENV/bin/python" "$TOOL_ROOT/helpers/train_original_best.py" --data-dir "$ORIGINAL_DATA" --job-dir "$OUT/training" --seed "$SEED" > "$OUT/training.log" 2>&1
  unset CUDA_VISIBLE_DEVICES
  "$ENV/bin/python" "$TOOL_ROOT/helpers/evaluate_original_best.py" --data-dir "$ORIGINAL_DATA" --checkpoint "$OUT/training/checkpoints/best.ckpt" --output-dir "$OUT/evaluation" --seeds 0 1 2 > "$OUT/evaluation.log" 2>&1
  export CUDA_VISIBLE_DEVICES="$GPU"
  printf '%s original seed=%s training_and_evaluation_complete\n' "$(date -Is)" "$SEED" >> "$RUN_ROOT/progress.log"
done
