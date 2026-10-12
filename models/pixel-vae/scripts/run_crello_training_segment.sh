#!/usr/bin/env bash

# @file models/pixel-vae/scripts/run_crello_training_segment.sh
# @brief Run one resumable PixelVAE Crello training segment.
# @description
#   Production runs are invoked through colab-gpu-run and use epoch boundaries
#   10, 60, 110, 160, 210, and 250. The same script supports a one-epoch CPU
#   rehearsal when PIXELVAE_CPU_REHEARSAL=1.
# @option --hub-input NAME=URI Durable input; source, data_tar, resume, and best are supported.
# @example
#   bash run_crello_training_segment.sh --hub-input source=https://github.com/creative-graphic-design/design-generators/archive/<sha>.tar.gz --hub-input data_tar=hf://datasets/<owner>/<repo>/<file>.tar.gz

set -Eeuo pipefail

source_uri=''
data_uri=''
resume_uri=''
best_uri=''
while (($#)); do
    case "$1" in
        --hub-input)
            (($# >= 2)) || { printf 'missing Hub input value\n' >&2; exit 2; }
            name="${2%%=*}"
            uri="${2#*=}"
            [[ "$uri" != "$2" && -n "$uri" ]] || { printf 'invalid Hub input\n' >&2; exit 2; }
            case "$name" in
                source) source_uri="$uri" ;;
                data_tar) data_uri="$uri" ;;
                resume) resume_uri="$uri" ;;
                best) best_uri="$uri" ;;
                *) printf 'unsupported Hub input name: %s\n' "$name" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --help)
            sed -n '1,12p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) printf 'unsupported argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ "$source_uri" =~ ^https://github\.com/creative-graphic-design/design-generators/archive/([[:xdigit:]]{40})\.tar\.gz$ ]] || {
    printf 'source input must be a commit-pinned GitHub archive\n' >&2
    exit 2
}
source_sha="${BASH_REMATCH[1],,}"
work_root="${PIXELVAE_WORK_ROOT:-$PWD}"
cpu_rehearsal="${PIXELVAE_CPU_REHEARSAL:-0}"
end_epoch="${PIXELVAE_END_EPOCH:-}"
if [[ "$cpu_rehearsal" != 1 ]]; then
    shopt -s nullglob
    job_work_root="${COLAB_JOB_WORK_ROOT:-$work_root/colab-jobs}"
    wrapper_logs=("$job_work_root"/*/logs/job.log)
    if ((${#wrapper_logs[@]} != 1)); then
        printf 'production training must run inside the colab-gpu-run wrapper\n' >&2
        exit 2
    fi
    job_name="$(basename "$(dirname "$(dirname "${wrapper_logs[0]}")")")"
    if [[ "$job_name" =~ ^pixelvae-([[:xdigit:]]{40})-e([0-9]{3})-a[a-z0-9-]+$ ]]; then
        job_source_sha="${BASH_REMATCH[1],,}"
        job_end_epoch="${BASH_REMATCH[2]}"
        [[ "$job_source_sha" == "$source_sha" ]] || {
            printf 'job name and source archive commits do not match\n' >&2
            exit 2
        }
        end_epoch="$job_end_epoch"
    else
        printf 'job name must encode the source commit, epoch boundary, and unique attempt\n' >&2
        exit 2
    fi
fi
[[ "$end_epoch" =~ ^[0-9]+$ ]] || { printf 'set PIXELVAE_END_EPOCH\n' >&2; exit 2; }
end_epoch=$((10#$end_epoch))

repo_dir="${PIXELVAE_PROJECT_DIR:-$work_root/design-generators}"
data_dir="${PIXELVAE_DATA_DIR:-$work_root/crello-image}"
run_dir="${PIXELVAE_RUN_DIR:-$work_root/pixelvae-output}"
batch_size="${PIXELVAE_BATCH_SIZE:-64}"
if [[ "$cpu_rehearsal" == 1 ]]; then
    [[ "$batch_size" == 1 && ( "$end_epoch" == 1 || "$end_epoch" == 2 ) ]] || {
        printf 'CPU rehearsal uses batch size 1 and endpoint 1 or 2\n' >&2
        exit 2
    }
    [[ -n "${PIXELVAE_SOURCE_SHA:-}" && "${PIXELVAE_SOURCE_SHA,,}" == "$source_sha" ]] || {
        printf 'CPU rehearsal source commit is missing or mismatched\n' >&2
        exit 2
    }
else
    [[ "$batch_size" == 64 && "$end_epoch" -le 250 ]] || {
        printf 'production recipe requires batch size 64 and at most 250 epochs\n' >&2
        exit 2
    }
    [[ -n "$data_uri" ]] || { printf 'production run requires --hub-input data_tar=URI\n' >&2; exit 2; }
fi

if [[ -f "$repo_dir/.pixelvae-source-commit" ]]; then
    [[ "$(<"$repo_dir/.pixelvae-source-commit")" == "$source_sha" ]] || {
        printf 'existing source checkout has a different commit\n' >&2
        exit 2
    }
else
    [[ ! -e "$repo_dir" ]] || { printf 'source directory already exists without a commit marker\n' >&2; exit 2; }
    archive_path="$work_root/design-generators-${source_sha}.tar.gz"
    mkdir -p "$repo_dir"
    curl --fail --silent --show-error --location --retry 3 --output "$archive_path" "$source_uri"
    tar -xzf "$archive_path" --strip-components=1 -C "$repo_dir"
    printf '%s\n' "$source_sha" > "$repo_dir/.pixelvae-source-commit"
fi

cd "$repo_dir"
private_dir="$work_root/pixelvae-private/${job_name:-cpu-rehearsal-$$}"
mkdir -p "$private_dir"
private_log="$private_dir/job.log"
[[ ! -e "$private_log" ]] || { printf 'refusing to overwrite a private log\n' >&2; exit 2; }
: > "$private_log"
run_logged() {
    if "$@" >>"$private_log" 2>&1; then
        return 0
    else
        status=$?
    fi
    emit_scanned_log || true
    return "$status"
}
emit_scanned_log() {
    if grep -Eiq 'hf_[[:alnum:]]{20,}' "$private_log"; then
        printf 'log contains an HF token-shaped value; detailed output is withheld\n' >&2
        return 1
    fi
    cat "$private_log"
}
download_hub_file() {
    local uri="$1"
    local target="$2"
    local expected_repo_type="$3"
    run_logged uv run --locked --package pixel-vae --extra training python - "$uri" "$target" "$expected_repo_type" <<'PY'
import shutil
import sys
from urllib.parse import urlsplit

from huggingface_hub import hf_hub_download

uri, target, expected = sys.argv[1:]
parts = urlsplit(uri)
components = parts.path.strip("/").split("/")
if parts.scheme != "hf" or parts.netloc != expected or len(components) < 3:
    raise SystemExit(f"input must be an hf://{expected}/.../<file> URI")
downloaded = hf_hub_download(
    repo_id="/".join(components[:2]),
    filename="/".join(components[2:]),
    repo_type="dataset" if expected == "datasets" else "model",
    token=True,
)
shutil.copyfile(downloaded, target)
PY
}

uv_install_dir="$work_root/uv-bin"
if ! uv_version="$(uv --version 2>/dev/null)" || [[ ! "$uv_version" =~ ^uv[[:space:]]0\.12\.22([[:space:]]|$) ]]; then
    mkdir -p "$uv_install_dir"
    run_logged bash -o pipefail -c '
        curl --fail --silent --show-error --location https://astral.sh/uv/0.12.22/install.sh |
            env UV_UNMANAGED_INSTALL="$1" sh
    ' _ "$uv_install_dir"
fi
export PATH="$uv_install_dir:$PATH"
uv_version="$(uv --version 2>/dev/null)" || { printf 'uv 0.12.22 is required; uv is unavailable\n' >&2; exit 2; }
[[ "$uv_version" =~ ^uv[[:space:]]0\.12\.22([[:space:]]|$) ]] || {
    printf 'uv 0.12.22 is required, found: %s\n' "$uv_version" >&2
    exit 2
}
export UV_CACHE_DIR="$work_root/uv-cache"
export HF_HOME="$work_root/pixelvae-hf"
export HF_TOKEN_PATH="${PIXELVAE_HF_TOKEN_PATH:-$work_root/hf-job-token}"
export HF_HUB_VERBOSITY=error
run_logged uv sync --locked --package pixel-vae --extra training

if [[ -n "$data_uri" && ! -f "$data_dir/count.json" ]]; then
    [[ ! -e "$data_dir" ]] || { printf 'data directory exists without count.json\n' >&2; exit 2; }
    mkdir -p "$data_dir"
    download_hub_file "$data_uri" "$work_root/pixelvae-data.tar.gz" datasets
    run_logged tar -xzf "$work_root/pixelvae-data.tar.gz" --no-same-owner -C "$data_dir"
fi

if [[ "$cpu_rehearsal" == 1 && ! -f "$data_dir/count.json" ]]; then
    run_logged uv run --locked --package pixel-vae --extra training python - "$data_dir" <<'PY'
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from tfrecord.writer import TFRecordWriter

data_dir = Path(sys.argv[1])
data_dir.mkdir(parents=True)
(data_dir / "count.json").write_text(
    json.dumps({"train": 1, "val": 1, "test": 0}), encoding="utf-8"
)
for split, value in (("train", 64), ("val", 192)):
    image = Image.new("RGBA", (256, 256), (value, value, value, 255))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    writer = TFRecordWriter(str(data_dir / f"{split}-00000.tfrecord"))
    writer.write({"image": (encoded.getvalue(), "byte")}, None)
    writer.close()
PY
fi
[[ -f "$data_dir/count.json" ]] || { printf 'Crello data archive is missing count.json\n' >&2; exit 2; }

mkdir -p "$run_dir/checkpoints" "$run_dir/reports" "$run_dir/logs"
resume_path="${PIXELVAE_RESUME_FROM:-}"
best_path="${PIXELVAE_BEST_FROM:-}"
if [[ -n "$resume_uri" ]]; then
    resume_path="$run_dir/checkpoints/last.ckpt"
    download_hub_file "$resume_uri" "$resume_path" models
fi
if [[ -n "$best_uri" ]]; then
    best_path="$run_dir/checkpoints/best.ckpt"
    download_hub_file "$best_uri" "$best_path" models
fi
[[ -z "$resume_path" || -f "$resume_path" ]] || { printf 'resume checkpoint is missing\n' >&2; exit 2; }
[[ -z "$best_path" || -f "$best_path" ]] || { printf 'best checkpoint is missing\n' >&2; exit 2; }
[[ -z "$resume_path" && -z "$best_path" || -n "$resume_path" && -n "$best_path" ]] || {
    printf 'resume and best checkpoints must be supplied together\n' >&2
    exit 2
}

start_epoch=0
if [[ -n "$resume_path" ]]; then
    start_epoch="$(uv run --locked --package pixel-vae --extra training python - "$resume_path" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not checkpoint.get("optimizer_states"):
    raise SystemExit("resume checkpoint has no optimizer state")
print(int(checkpoint["epoch"]) + 1)
PY
)"
fi
if [[ "$cpu_rehearsal" == 1 ]]; then
    [[ "$start_epoch" == 0 && "$end_epoch" == 1 || "$start_epoch" == 1 && "$end_epoch" == 2 ]] || {
        printf 'CPU rehearsal must save epoch 1 and resume through epoch 2\n' >&2
        exit 2
    }
    export CUDA_VISIBLE_DEVICES=''
    run_logged uv run --locked --package pixel-vae --extra training python - <<'PY'
import torch

visible = torch.cuda.device_count()
if visible != 0:
    raise SystemExit(f"CPU rehearsal expected no visible CUDA devices, got {visible}")
PY
    accelerator=cpu
else
    [[ "$start_epoch" == 0 && "$end_epoch" == 10 || "$start_epoch" -gt 0 && "$end_epoch" -gt "$start_epoch" && "$end_epoch" -le 250 && $((end_epoch - start_epoch)) -le 50 ]] || {
        printf 'production segment must end at epoch 10 or the next 50-epoch boundary\n' >&2
        exit 2
    }
    if [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
    run_logged uv run --locked --package pixel-vae --extra training python - <<'PY'
import torch

visible = torch.cuda.device_count()
if visible != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {visible}")
PY
    accelerator=gpu
fi

[[ "$end_epoch" == 10 || "$end_epoch" == 60 || "$end_epoch" == 110 || "$end_epoch" == 160 || "$end_epoch" == 210 || "$end_epoch" == 250 || "$cpu_rehearsal" == 1 ]] || {
    printf 'unsupported production epoch boundary\n' >&2
    exit 2
}
if [[ -z "$resume_path" && -e "$run_dir/checkpoints/last.ckpt" ]]; then
    printf 'refusing to start without a resume checkpoint over an existing run\n' >&2
    exit 2
fi

gpu_samples="$run_dir/reports/gpu-samples.csv"
sampler_pid=''
if [[ "$cpu_rehearsal" != 1 && "$start_epoch" == 0 ]]; then
    run_logged nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
    nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits -l 1 > "$gpu_samples" 2>>"$private_log" &
    sampler_pid=$!
    sleep 2
    kill -0 "$sampler_pid" 2>/dev/null || { printf 'GPU sampling failed to start\n' >&2; exit 2; }
    [[ -s "$gpu_samples" ]] || { printf 'GPU sampling produced no measurements\n' >&2; exit 2; }
fi
stop_sampler() {
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=''
    fi
}
trap stop_sampler EXIT

time_file="$private_dir/train-time.txt"
fit_args=(
    --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
    "--trainer.accelerator=$accelerator"
    --trainer.devices=1
    "--trainer.max_epochs=$end_epoch"
    "--trainer.default_root_dir=$run_dir"
    --trainer.logger=false
    --trainer.enable_progress_bar=false
    "--data.init_args.data_dir=$data_dir"
    "--data.init_args.batch_size=$batch_size"
)
if [[ -n "$resume_path" ]]; then
    fit_args+=(--ckpt_path "$resume_path")
fi
TIMEFORMAT='%3R %P'
set +e
{ time uv run --locked --package pixel-vae --extra training traingen fit \
    "${fit_args[@]}" >>"$private_log" 2>&1; } 2>"$time_file"
train_status=$?
set -e
stop_sampler
if ((train_status != 0)); then
    emit_scanned_log || true
    printf 'training segment failed with exit code %s\n' "$train_status" >&2
    exit "$train_status"
fi

report_path="$run_dir/reports/segment-e$(printf '%03d' "$end_epoch").json"
run_logged uv run --locked --package pixel-vae --extra training python - \
    "$data_dir/count.json" "$run_dir/checkpoints/last.ckpt" \
    "$run_dir/checkpoints/best.ckpt" "$start_epoch" "$end_epoch" \
    "$batch_size" "$time_file" "$gpu_samples" "$report_path" "$source_sha" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

counts_path, last_path, best_path, start, end, batch, time_path, gpu_path, report_path, source = sys.argv[1:]
counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
completed = int(checkpoint["epoch"]) + 1
if completed != int(end) or not checkpoint.get("optimizer_states"):
    raise SystemExit("last checkpoint is not a full-state epoch-boundary checkpoint")
elapsed_text, cpu_text = Path(time_path).read_text(encoding="utf-8").split()
train_steps = math.ceil(counts["train"] / int(batch))
val_steps = math.ceil(counts["val"] / int(batch))
samples = []
if Path(gpu_path).is_file():
    with Path(gpu_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) >= 4:
                try:
                    samples.append(tuple(float(value.strip()) for value in row[1:4]))
                except ValueError:
                    continue
gpu = None
if samples:
    gpu = {
        "sample_count": len(samples),
        "mean_utilization_percent": sum(row[0] for row in samples) / len(samples),
        "mean_power_watts": sum(row[1] for row in samples) / len(samples),
        "mean_power_limit_watts": sum(row[2] for row in samples) / len(samples),
        "mean_power_limit_fraction": sum(row[1] / row[2] for row in samples) / len(samples),
    }
report = {
    "source_commit": source,
    "segment": {
        "start_epoch": int(start),
        "end_epoch": completed,
        "global_step": int(checkpoint["global_step"]),
    },
    "data": {
        "train_unique_images": counts["train"],
        "validation_unique_images": counts["val"],
        "batch_size": int(batch),
        "train_steps_per_epoch": train_steps,
        "validation_steps_per_epoch": val_steps,
    },
    "timing": {
        "training_wall_seconds": float(elapsed_text),
        "cpu_percent": cpu_text,
        "seconds_per_train_step_including_validation": float(elapsed_text)
        / max(1, (completed - int(start)) * train_steps),
        "projected_250_epoch_hours": float(elapsed_text)
        / max(1, completed - int(start))
        * 250
        / 3600,
    },
    "gpu_samples": gpu,
    "checkpoints": {
        "last_sha256": hashlib.sha256(Path(last_path).read_bytes()).hexdigest(),
        "best_sha256": hashlib.sha256(Path(best_path).read_bytes()).hexdigest(),
    },
}
Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, sort_keys=True))
PY

for output_log in "$private_log" "$gpu_samples" "$report_path"; do
    [[ -f "$output_log" ]] || continue
    if grep -Eiq 'hf_[[:alnum:]]{20,}' "$output_log"; then
        printf 'log or report contains an HF token-shaped value; detailed output is withheld\n' >&2
        exit 2
    fi
done
shopt -s nullglob
job_work_root="${COLAB_JOB_WORK_ROOT:-$work_root/colab-jobs}"
for wrapper_log in "$job_work_root"/*/logs/job.log; do
    if grep -Eiq 'hf_[[:alnum:]]{20,}' "$wrapper_log"; then
        printf 'wrapper log contains an HF token-shaped value; upload is withheld by operator review\n' >&2
        exit 2
    fi
done
cp "$private_log" "$run_dir/logs/training-e$(printf '%03d' "$end_epoch").log"
printf 'PixelVAE segment finished at epoch %s; report: %s\n' "$end_epoch" "$report_path"
cat "$report_path"
