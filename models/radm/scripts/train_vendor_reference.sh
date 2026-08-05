#!/usr/bin/env bash

# @file models/radm/scripts/train_vendor_reference.sh
# @brief Prepare and launch RADM vendor reference training.
# @description
#   This entrypoint keeps generated datasets and checkpoints below RADM_CACHE_ROOT,
#   accepts explicit GPU and seed arguments, and delegates Detectron2 execution
#   to models/radm/scripts/run_vendor_training.py.

set -euo pipefail

cache_root="${RADM_CACHE_ROOT:-$HOME/.cache/radm}"
gpu="0"
num_gpus="1"
seed="0"
max_iter="20"
source_root="${cache_root}/datasets/cgl-dataset-v2"
prepared_root="${cache_root}/vendor-data/cgl-v2"
run_root="${cache_root}/vendor-runs"
run_id="radm-cgl-v2-vendor-dry-run"
max_train_samples="8"
max_val_samples="4"
plan_only="false"
overwrite="false"

usage() {
    cat <<'USAGE'
Usage: models/radm/scripts/train_vendor_reference.sh [options]

Options:
  --gpu ID                 CUDA_VISIBLE_DEVICES value for the launch.
  --num-gpus N             Number of visible GPUs used by Detectron2.
  --seed N                 Fixed random seed.
  --max-iter N             Training iterations; use 250000 for S5.
  --source-root PATH       CGL-v2 ralf-style Parquet root.
  --prepared-root PATH     Materialized vendor dataset root.
  --run-root PATH          Vendor run root.
  --run-id NAME            Run directory name below --run-root.
  --max-train-samples N    Limit train samples during materialization.
  --max-val-samples N      Limit validation samples during materialization.
  --full-data              Materialize all available train/validation rows.
  --overwrite              Recreate the prepared vendor dataset.
  --plan-only              Validate paths and write launch_plan.json only.
  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --num-gpus)
            num_gpus="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --max-iter)
            max_iter="$2"
            shift 2
            ;;
        --source-root)
            source_root="$2"
            shift 2
            ;;
        --prepared-root)
            prepared_root="$2"
            shift 2
            ;;
        --run-root)
            run_root="$2"
            shift 2
            ;;
        --run-id)
            run_id="$2"
            shift 2
            ;;
        --max-train-samples)
            max_train_samples="$2"
            shift 2
            ;;
        --max-val-samples)
            max_val_samples="$2"
            shift 2
            ;;
        --full-data)
            max_train_samples=""
            max_val_samples=""
            shift
            ;;
        --overwrite)
            overwrite="true"
            shift
            ;;
        --plan-only)
            plan_only="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

prepare_args=(
    --source-root "$source_root"
    --output-root "$prepared_root"
)
if [[ -n "$max_train_samples" ]]; then
    prepare_args+=(--max-train-samples "$max_train_samples")
fi
if [[ -n "$max_val_samples" ]]; then
    prepare_args+=(--max-val-samples "$max_val_samples")
fi
if [[ "$overwrite" == "true" ]]; then
    prepare_args+=(--overwrite)
fi

uv run --package radm --extra vendor --extra training python \
    models/radm/scripts/prepare_vendor_cgl_v2_dataset.py \
    "${prepare_args[@]}"

run_args=(
    --prepared-data-root "$prepared_root"
    --run-root "$run_root"
    --run-id "$run_id"
    --seed "$seed"
    --num-gpus "$num_gpus"
    --max-iter "$max_iter"
)
if [[ "$plan_only" == "true" ]]; then
    run_args+=(--plan-only)
fi

CUDA_VISIBLE_DEVICES="$gpu" uv run --package radm --extra vendor python \
    models/radm/scripts/run_vendor_training.py \
    "${run_args[@]}"
