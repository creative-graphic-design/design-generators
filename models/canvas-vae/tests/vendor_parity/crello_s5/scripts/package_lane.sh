#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
NEED_MIB=8090
MARGIN_MIB=4096
ENV="${CANVAS_VAE_PACKAGE_ENV:?set CANVAS_VAE_PACKAGE_ENV}"
INPUTS="${CANVAS_VAE_S5_INPUTS:-$ROOT/.cache/canvas-vae/s5-path-a/inputs}"
DATA="${CANVAS_VAE_PACKAGE_DATA:-$INPUTS/package-data}"
CACHE="${CANVAS_VAE_EMBEDDING_CACHE:-$INPUTS/cache}"
GUARD="$ROOT/models/canvas-vae/scripts/source_tree_guard.py"
GUARD_MANIFEST="$RUN_ROOT/guards/package-seed-0.json"
unset LD_LIBRARY_PATH PYTHONPATH CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
export CUDA_VISIBLE_DEVICES="$GPU" NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/models/canvas-vae/src:$ROOT/lib/laygen/src:$ROOT/lib/traingen/src:$ROOT/lib/traingen-parity/src"
export CANVAS_VAE_PIP_FREEZE_FILE="$RUN_ROOT/stack/package-pip-freeze.txt"
exec 9>"$RUN_ROOT/seed-start.lock"

wait_for_capacity() {
  local seed="$1" free
  flock -x 9
  while true; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d '[:space:]')"
    if (( free >= NEED_MIB + MARGIN_MIB )); then
      printf '%s package seed=%s free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$seed" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
      return
    fi
    printf '%s package seed=%s waiting free=%s MiB threshold=%s MiB\n' "$(date -Is)" "$seed" "$free" "$((NEED_MIB + MARGIN_MIB))" >> "$RUN_ROOT/memory-gates.log"
    sleep 10
  done
}

for required in "$ENV/bin/traingen" "$ENV/bin/python" "$DATA" "$CACHE" "$GUARD" "$ROOT/models/canvas-vae/configs/training/canvas_vae_crello.yaml"; do
  [[ -e "$required" ]] || { printf 'missing required input or executable: %s\n' "$required" >&2; exit 1; }
done
SEEDS=("$@")
if ((${#SEEDS[@]} == 0)); then
  SEEDS=(0 1 2)
fi
for SEED in "${SEEDS[@]}"; do
  [[ "$SEED" == 0 || "$SEED" == 1 || "$SEED" == 2 ]] || { printf 'unsupported package seed: %s\n' "$SEED" >&2; exit 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { printf 'source commit changed before package seed %s\n' "$SEED" >&2; exit 1; }
  wait_for_capacity "$SEED"
  OUT="$RUN_ROOT/package/seed-$SEED"
  mkdir -p "$OUT"
  if [[ "$SEED" == 0 ]]; then
    "$ENV/bin/python" "$GUARD" record --source-root "$ROOT" --manifest "$GUARD_MANIFEST" --seed 0 > "$OUT/source-tree-guard.json"
  else
    "$ENV/bin/python" "$GUARD" verify --source-root "$ROOT" --manifest "$GUARD_MANIFEST" --seed "$SEED" > "$OUT/source-tree-guard.json"
  fi
  flock -u 9
  unset CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
  "$ENV/bin/traingen" fit --config models/canvas-vae/configs/training/canvas_vae_crello.yaml \
    --seed_everything="$SEED" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=32-true \
    --trainer.max_epochs=500 --trainer.log_every_n_steps=50 --trainer.enable_progress_bar=false \
    --trainer.default_root_dir="$OUT/training" \
    --model.init_args.data_dir="$DATA" --model.init_args.fixture_dir="$CACHE" \
    --data.init_args.data_dir="$DATA" --data.init_args.fixture_dir="$CACHE" \
    --data.init_args.use_s4_train_stream=true --data.init_args.num_workers=8 > "$OUT/training.log" 2>&1
  CHECKPOINT="$OUT/training/lightning_logs/version_0/checkpoints/best.ckpt"
  [[ -f "$CHECKPOINT" ]] || { printf 'package top-k checkpoint missing for seed %s: %s\n' "$SEED" "$CHECKPOINT" >&2; exit 1; }
  unset CUDA_VISIBLE_DEVICES
  "$ENV/bin/python" "$ROOT/models/canvas-vae/scripts/convert_original_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --vocabulary "$DATA/vocabulary.json" --dataset crello \
    --output-dir "$OUT/evaluated-model" > "$OUT/conversion.log" 2>&1
  "$ENV/bin/python" "$ROOT/models/canvas-vae/scripts/evaluate_crello.py" \
    --checkpoint "$OUT/evaluated-model" --data-dir "$DATA" --fixture-dir "$CACHE" \
    --seeds 0 1 2 --batch-size 1024 --device cpu --output "$OUT/evaluation.json" \
    --expected-test-documents 2278 > "$OUT/evaluation.log" 2>&1
  export CUDA_VISIBLE_DEVICES="$GPU"
  printf '%s package seed=%s training_and_evaluation_complete\n' "$(date -Is)" "$SEED" >> "$RUN_ROOT/progress.log"
done
