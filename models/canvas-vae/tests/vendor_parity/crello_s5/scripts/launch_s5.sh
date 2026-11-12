#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
ORIGINAL_NEED_MIB=9831
PACKAGE_NEED_MIB=8090
MARGIN_PER_LANE_MIB=4096
COMBINED_THRESHOLD_MIB=$((ORIGINAL_NEED_MIB + PACKAGE_NEED_MIB + 2 * MARGIN_PER_LANE_MIB))
UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d '[:space:]')"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { printf 'refusing launch: source commit differs\n' >&2; exit 1; }
git -C "$ROOT" diff --quiet
git -C "$ROOT" diff --cached --quiet
FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d '[:space:]')"
APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F, -v uuid="$UUID" '$1 ~ uuid {print}')"
if (( FREE < COMBINED_THRESHOLD_MIB )); then
  printf 'refusing concurrent launch: GPU 7 free=%s MiB, combined threshold=%s MiB\n' "$FREE" "$COMBINED_THRESHOLD_MIB" >&2
  exit 1
fi
if [[ -n "$APPS" ]]; then
  printf 'refusing launch: GPU 7 already has compute apps\n' >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids" "$RUN_ROOT/stack" "$RUN_ROOT/guards"
[[ -f "$RUN_ROOT/preflight/package.passed" && -f "$RUN_ROOT/preflight/original.passed" ]] || { printf 'both isolated lane preflights must pass before launch\n' >&2; exit 1; }
[[ ! -e "$RUN_ROOT/manifest.json" ]] || { printf 'refusing to reuse an existing full-run manifest\n' >&2; exit 1; }
sha256sum --check "$RUN_ROOT/preflight/package-freeze.sha256" "$RUN_ROOT/preflight/original-freeze.sha256"
python3 "$TOOL_ROOT/scripts/write_manifest.py"
STARTED_AT="$(date -Is)"
printf '{"started_at":"%s","source_commit":"%s","gpu_index":%s,"gpu_uuid":"%s","free_mib_before":%s,"combined_threshold_mib":%s,"launch_mode":"concurrent-two-lanes","package_memory_estimate_mib":%s,"original_memory_estimate_mib":%s}\n' \
  "$STARTED_AT" "$EXPECTED_COMMIT" "$GPU" "$UUID" "$FREE" "$COMBINED_THRESHOLD_MIB" "$PACKAGE_NEED_MIB" "$ORIGINAL_NEED_MIB" > "$RUN_ROOT/launch.json"
setsid bash -c 'printf "%s\n" "$$" > "$1"; exec bash "$2"' launch "$RUN_ROOT/pids/original-lane.pid" "$TOOL_ROOT/scripts/original_lane.sh" > "$RUN_ROOT/logs/original-lane.log" 2>&1 < /dev/null &
setsid bash -c 'printf "%s\n" "$$" > "$1"; exec bash "$2"' launch "$RUN_ROOT/pids/package-lane.pid" "$TOOL_ROOT/scripts/package_lane.sh" > "$RUN_ROOT/logs/package-lane.log" 2>&1 < /dev/null &
for _ in $(seq 1 50); do
  [[ -s "$RUN_ROOT/pids/original-lane.pid" && -s "$RUN_ROOT/pids/package-lane.pid" ]] && break
  sleep 0.1
done
[[ -s "$RUN_ROOT/pids/original-lane.pid" && -s "$RUN_ROOT/pids/package-lane.pid" ]] || { printf 'launcher did not record both lane PIDs\n' >&2; exit 1; }
ORIGINAL_PID="$(cat "$RUN_ROOT/pids/original-lane.pid")"
PACKAGE_PID="$(cat "$RUN_ROOT/pids/package-lane.pid")"
sleep 15
if ! kill -0 "$ORIGINAL_PID" 2>/dev/null; then
  kill -TERM -- "-$PACKAGE_PID" 2>/dev/null || true
  printf 'original lane failed startup; stopped only package lane group\n' >&2
  exit 1
fi
if ! kill -0 "$PACKAGE_PID" 2>/dev/null; then
  kill -TERM -- "-$ORIGINAL_PID" 2>/dev/null || true
  printf 'package lane failed startup; stopped only original lane group\n' >&2
  exit 1
fi
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader -i "$GPU" > "$RUN_ROOT/startup-gpu-state.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F, -v uuid="$UUID" '$1 ~ uuid {print}' > "$RUN_ROOT/startup-compute-apps.csv"
python3 - "$RUN_ROOT" "$ORIGINAL_PID" "$PACKAGE_PID" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
manifest_path = run_root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "running"
manifest["startup"] = {
    "original_lane_pid": int(sys.argv[2]),
    "package_lane_pid": int(sys.argv[3]),
    "gpu_state": (run_root / "startup-gpu-state.csv").read_text(encoding="utf-8").strip(),
    "compute_apps_by_pid": (run_root / "startup-compute-apps.csv").read_text(encoding="utf-8").splitlines(),
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

launch_path = run_root / "launch.json"
launch = json.loads(launch_path.read_text(encoding="utf-8"))
launch["original_lane_pid"] = int(sys.argv[2])
launch["package_lane_pid"] = int(sys.argv[3])
launch["gpu_state_after_start"] = manifest["startup"]["gpu_state"]
launch["compute_apps_by_pid"] = manifest["startup"]["compute_apps_by_pid"]
launch_path.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
setsid bash "$TOOL_ROOT/scripts/telemetry.sh" > "$RUN_ROOT/logs/telemetry.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_ROOT/pids/telemetry.pid"
printf 'STARTUP_SURVIVED original_pid=%s package_pid=%s gpu_free_mib=%s combined_threshold_mib=%s\n' \
  "$ORIGINAL_PID" "$PACKAGE_PID" \
  "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d '[:space:]')" \
  "$COMBINED_THRESHOLD_MIB"
