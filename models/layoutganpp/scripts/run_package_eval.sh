#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATASET="$1"
SEED="$2"
GPU="$3"
RUN_ID="$DATASET/package-seed-$SEED"
RUN_DIR="$ROOT/.cache/layoutganpp/package-rerun-295e2c6-20261026/$DATASET/package-seed-$SEED"
OUT_ROOT="${4:-.cache/layoutganpp/package-rerun-295e2c6-20261026/evaluation}"
if [[ "$OUT_ROOT" = /* || "/$OUT_ROOT/" == *"/../"* ]]; then
  echo "evaluation root must be repository-relative and cannot contain .."
  exit 2
fi
OUT_DIR="$ROOT/$OUT_ROOT/$DATASET/package-seed-$SEED"
PYTHON="$ROOT/.cache/layoutganpp/runtime/audit-venv/bin/python"
case "$DATASET" in
  magazine) VENDOR_DATASET=magazine ;;
  rico13) VENDOR_DATASET=rico ;;
  publaynet) VENDOR_DATASET=publaynet ;;
  *) echo "unsupported dataset: $DATASET"; exit 2 ;;
esac
if [[ -e "$OUT_DIR" ]]; then
  echo "evaluation output already exists: $OUT_DIR"
  exit 125
fi
SHM_FREE="$(df -B1 --output=avail /dev/shm | tail -n 1 | tr -d ' ')"
GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU + 1))p" | tr -d ' ')"
if [[ "$SHM_FREE" -lt 34359738368 || "$GPU_FREE_MIB" -lt 20480 ]]; then
  echo "resource precondition failed: shm=$SHM_FREE GPU$GPU=${GPU_FREE_MIB}MiB"
  exit 125
fi
mkdir -p "$OUT_DIR"
STARTED_AT="$(TZ=Asia/Tokyo date --iso-8601=seconds)"
python - "$OUT_DIR/launch.json" "$RUN_ID" "$GPU" "$SHM_FREE" "$GPU_FREE_MIB" "$STARTED_AT" <<'PY'
import json, sys
from pathlib import Path
path, run_id, gpu, shm, gpu_free, started = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": int(gpu),
    "shm_free_bytes_at_launch": int(shm),
    "gpu_free_mib_at_launch": int(gpu_free),
    "started_at_jst": started,
    "evaluation_seed": 42005,
    "split": "test",
}, indent=2, sort_keys=True) + "\n")
PY
ulimit -n "$(ulimit -Hn)" || exit 125
export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$ROOT/.cache/layoutganpp/vendor-work" || exit 125
  "$PYTHON" "$ROOT/models/layoutganpp/scripts/generate_package_predictions.py" --dataset "$DATASET" --run-dir "$RUN_DIR" --output "$OUT_DIR/predictions.pkl" --device cuda:0 > "$OUT_DIR/generate.log" 2>&1
RESULT=$?
if [[ "$RESULT" -eq 0 ]]; then
  "$PYTHON" "$ROOT/vendor/const-layout/eval.py" "$VENDOR_DATASET" "$OUT_DIR/predictions.pkl" --batch_size 64 > "$OUT_DIR/eval.log" 2>&1
  RESULT=$?
fi
printf '%s\n' "$RESULT" > "$OUT_DIR/exit-code"
ENDED_AT="$(TZ=Asia/Tokyo date --iso-8601=seconds)"
python - "$OUT_DIR/launch.json" "$ENDED_AT" "$RESULT" <<'PY'
import json, sys
from pathlib import Path
path, ended, result = sys.argv[1:]
record=json.loads(Path(path).read_text())
record["ended_at_jst"] = ended
record["exit_code"] = int(result)
Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
echo "EVALUATION_EXIT $RUN_ID code=$RESULT started=$STARTED_AT ended=$ENDED_AT shm=$SHM_FREE GPU$GPU=${GPU_FREE_MIB}MiB"
exit "$RESULT"
