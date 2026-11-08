#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d '[:space:]')"
while kill -0 "$(cat "$RUN_ROOT/pids/original-lane.pid")" 2>/dev/null || kill -0 "$(cat "$RUN_ROOT/pids/package-lane.pid")" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw,power.limit --format=csv,noheader -i "$GPU" >> "$RUN_ROOT/gpu-telemetry.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F, -v uuid="$UUID" '$1 ~ uuid {gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); printf "%s,%s,%s\n", strftime("%Y-%m-%dT%H:%M:%S%z"), $2, $3}' >> "$RUN_ROOT/gpu-process-memory.csv"
  for lane in original package; do
    lane_pid="$(cat "$RUN_ROOT/pids/$lane-lane.pid")"
    ps -eo pid=,pgid=,pcpu= | awk -v stamp="$(date -Is)" -v lane="$lane" -v pgid="$lane_pid" '$2 == pgid {printf "%s,%s,%s,%s\n", stamp, lane, $1, $3}' >> "$RUN_ROOT/process-cpu.csv"
  done
  sleep 1
done
