#!/usr/bin/env bash

# @file scripts/pick_free_gpus.sh
# @brief Print the least-loaded GPU indices by used memory.
# @description
#   Select GPUs for multi-job training or evaluation runs before launching
#   detached workers. The picker sorts `nvidia-smi` results by `memory.used`
#   ascending and prints the first requested GPU indices, excluding any reserved
#   indices passed as a comma-separated list.
# @arg $1 count Number of GPU indices to print.
# @arg $2 exclude_csv Optional comma-separated GPU indices to skip.
# @stdout One GPU index per line, least-loaded first.
# @example
#   mapfile -t gpus < <(scripts/pick_free_gpus.sh 6 "3,7")
#   CUDA_VISIBLE_DEVICES="${gpus[0]}" setsid ./train-one-seed.sh &

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: scripts/pick_free_gpus.sh <count> [exclude_csv]" >&2
  exit 2
fi

count="$1"
exclude_csv="${2:-}"

if [[ ! "${count}" =~ ^[0-9]+$ ]] || [ "${count}" -lt 1 ]; then
  echo "count must be a positive integer: ${count}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to pick free GPUs" >&2
  exit 127
fi

exclude=",${exclude_csv//[[:space:]]/},"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F',' -v exclude="${exclude}" '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        if (index(exclude, "," $1 ",") == 0) {
          print $2 " " $1
        }
      }
    ' \
  | sort -n -k1,1 -k2,2n \
  | head -n "${count}" \
  | awk '{ print $2 }'
