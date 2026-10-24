#!/usr/bin/env bash

# @file models/pixel-vae/scripts/run_crello_training_segment.sh
# @brief Run one resumable PixelVAE Crello training segment.
# @description
#   Production runs are invoked through colab-gpu-run, use 10 epochs first,
#   then at most 68 epochs per segment. At the measured rate, 68 epochs plus
#   30 minutes of setup and a 10% reserve fit within 9.5 hours. The 300-step
#   lockstep runs on the first production segment only. After a training-path
#   change, update the pinned attribution-report SHA-256 and set
#   PIXELVAE_RUN_LOCKSTEP=1. Later segments verify the pinned report SHA and
#   skip the lockstep. The same script supports one-epoch CPU rehearsals from a
#   tiny tar, including resume from the previous process, when
#   PIXELVAE_CPU_REHEARSAL=1; rehearsals require data_tar.
#   PIXELVAE_CPU_FIXTURE_PREP=1 only prepares a tiny fixture before packaging.
# @option --hub-input NAME=URI Durable input; source, data_tar, attribution, resume, and best are supported.
# @example
#   bash run_crello_training_segment.sh --hub-input source=https://github.com/creative-graphic-design/design-generators/archive/<sha>.tar.gz --hub-input data_tar=hf://<owner>/<repo>/<file>.tar

set -Eeuo pipefail

unset UV_SYSTEM_PYTHON PYTHONPATH MPLBACKEND

source_uri=''
data_uri=''
attribution_uri=''
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
                attribution) attribution_uri="$uri" ;;
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
cpu_fixture_prep="${PIXELVAE_CPU_FIXTURE_PREP:-0}"
run_lockstep="${PIXELVAE_RUN_LOCKSTEP:-0}"
end_epoch="${PIXELVAE_END_EPOCH:-}"
[[ "$cpu_fixture_prep" == 0 || "$cpu_fixture_prep" == 1 ]] || {
    printf 'PIXELVAE_CPU_FIXTURE_PREP must be 0 or 1\n' >&2
    exit 2
}
[[ "$run_lockstep" == 0 || "$run_lockstep" == 1 ]] || {
    printf 'PIXELVAE_RUN_LOCKSTEP must be 0 or 1\n' >&2
    exit 2
}
[[ "$cpu_rehearsal" == 1 || "$cpu_fixture_prep" == 0 ]] || {
    printf 'fixture preparation requires PIXELVAE_CPU_REHEARSAL=1\n' >&2
    exit 2
}
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
attribution_sha256='99030e6a1b53303898851707cb1f31b67a681e59b295f20408635b89738d1b17'
if [[ "$cpu_rehearsal" == 1 ]]; then
    [[ "$batch_size" == 1 && "$end_epoch" -ge 1 && "$end_epoch" -le 250 ]] || {
        printf 'CPU rehearsal uses batch size 1 and endpoint 1 through 250\n' >&2
        exit 2
    }
    [[ -n "${PIXELVAE_SOURCE_SHA:-}" && "${PIXELVAE_SOURCE_SHA,,}" == "$source_sha" ]] || {
        printf 'CPU rehearsal source commit is missing or mismatched\n' >&2
        exit 2
    }
    if [[ "$cpu_fixture_prep" == 1 ]]; then
        [[ "$end_epoch" == 1 && -z "$data_uri" ]] || {
            printf 'CPU fixture preparation uses endpoint 1 without data_tar\n' >&2
            exit 2
        }
    else
        [[ -n "$data_uri" ]] || {
            printf 'CPU rehearsal requires --hub-input data_tar=URI\n' >&2
            exit 2
        }
        [[ -n "$attribution_uri" ]] || {
            printf 'CPU rehearsal requires --hub-input attribution=URI\n' >&2
            exit 2
        }
    fi
else
    [[ "$batch_size" == 64 && "$end_epoch" -le 250 ]] || {
        printf 'production recipe requires batch size 64 and at most 250 epochs\n' >&2
        exit 2
    }
    [[ -n "$data_uri" ]] || { printf 'production run requires --hub-input data_tar=URI\n' >&2; exit 2; }
    [[ -n "$attribution_uri" ]] || {
        printf 'production run requires --hub-input attribution=URI\n' >&2
        exit 2
    }
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
if parts.scheme != "hf" or expected not in {"datasets", "models"}:
    raise SystemExit("input must use hf:// and a supported Hub repository type")

if parts.netloc in {"datasets", "models"}:
    if parts.netloc != expected or len(components) < 3:
        raise SystemExit(f"typed Hub input must use hf://{expected}/<owner>/<repo>/<file>")

    repo_parts, file_parts = components[:2], components[2:]
else:
    if not parts.netloc or len(components) < 2:
        raise SystemExit("Hub input must identify an owner, repository, and file")

    repo_parts, file_parts = [parts.netloc, components[0]], components[1:]

downloaded = hf_hub_download(
    repo_id="/".join(repo_parts),
    filename="/".join(file_parts),
    repo_type="dataset" if expected == "datasets" else "model",
    token=True,
)
shutil.copyfile(downloaded, target)
PY
}

uv_install_dir="$work_root/uv-install"
if [[ ! -x "$uv_install_dir/uv" ]]; then
    run_logged env UV_INSTALL_DIR="$uv_install_dir" UV_NO_MODIFY_PATH=1 \
        bash -o pipefail -c \
        'curl --fail --silent --show-error --location --retry 3 --proto "=https" --tlsv1.2 https://releases.astral.sh/github/uv/releases/download/0.12.22/uv-installer.sh | sh'
fi
[[ -x "$uv_install_dir/uv" ]] || {
    printf 'pinned uv installer did not create %s/uv\n' "$uv_install_dir" >&2
    exit 2
}
export PATH="$uv_install_dir:$PATH"
uv_version="$(uv --version 2>/dev/null)" || { printf 'uv 0.12.22 is required; uv is unavailable\n' >&2; exit 2; }
[[ "$uv_version" =~ ^uv[[:space:]]0\.12\.22([[:space:]]|$) ]] || {
    printf 'uv 0.12.22 is required, found: %s\n' "$uv_version" >&2
    exit 2
}
export UV_CACHE_DIR="$work_root/uv-cache"
export UV_PROJECT_ENVIRONMENT="$work_root/pixelvae-uv-env"
export HF_HOME="$work_root/pixelvae-hf"
export HF_TOKEN_PATH="${PIXELVAE_HF_TOKEN_PATH:-$work_root/hf-job-token}"
export HF_HUB_VERBOSITY=error
run_logged uv sync --locked --package pixel-vae --extra training

if [[ "$cpu_fixture_prep" != 1 ]]; then
    attribution_path="$work_root/pixelvae-attribution.md"
    [[ ! -e "$attribution_path" ]] || {
        printf 'refusing to overwrite the attribution report\n' >&2
        exit 2
    }
    download_hub_file "$attribution_uri" "$attribution_path" models
    actual_attribution_sha256="$(sha256sum "$attribution_path" | cut -d ' ' -f 1)"
    [[ "$actual_attribution_sha256" == "$attribution_sha256" ]] || {
        printf 'attribution report SHA-256 does not match the approved report\n' >&2
        exit 2
    }
    printf 'verified attribution report SHA-256 %s\n' "$actual_attribution_sha256"
fi

if [[ -n "$data_uri" && ! -f "$data_dir/count.json" ]]; then
    [[ ! -e "$data_dir" ]] || { printf 'data directory exists without count.json\n' >&2; exit 2; }
    mkdir -p "$data_dir"
    download_hub_file "$data_uri" "$work_root/pixelvae-data.tar" models
    run_logged tar -xf "$work_root/pixelvae-data.tar" --no-same-owner --wildcards \
        --strip-components=2 -C "$data_dir" 'original-run-2/crello-image/*'
fi

if [[ "$cpu_rehearsal" == 1 && "$cpu_fixture_prep" == 1 && ! -f "$data_dir/count.json" ]]; then
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
    json.dumps({"train": 1, "val": 1, "test": 1}), encoding="utf-8"
)
for split, value in (("train", 64), ("val", 192), ("test", 128)):
    image = Image.new("RGBA", (256, 256), (value, value, value, 255))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    writer = TFRecordWriter(str(data_dir / f"{split}-00000.tfrecord"))
    writer.write({"image": (encoded.getvalue(), "byte")}, None)
    writer.close()
(data_dir / "vocabulary.json").write_text("{}\n", encoding="utf-8")
PY
fi
[[ -f "$data_dir/count.json" ]] || { printf 'Crello data archive is missing count.json\n' >&2; exit 2; }
if [[ "$cpu_fixture_prep" == 1 ]]; then
    printf 'CPU fixture data is ready at %s\n' "$data_dir"
    exit 0
fi

mkdir -p "$run_dir/checkpoints" "$run_dir/reports" "$run_dir/logs" \
    "$private_dir/checkpoint-inputs"
resume_path="${PIXELVAE_RESUME_FROM:-}"
best_path="${PIXELVAE_BEST_FROM:-}"
if [[ -n "$resume_uri" ]]; then
    resume_path="$private_dir/checkpoint-inputs/resume-last.ckpt"
    download_hub_file "$resume_uri" "$resume_path" models
fi
if [[ -n "$best_uri" ]]; then
    best_path="$private_dir/checkpoint-inputs/resume-best.ckpt"
    download_hub_file "$best_uri" "$best_path" models
fi
[[ -z "$resume_path" || -f "$resume_path" ]] || { printf 'resume checkpoint is missing\n' >&2; exit 2; }
[[ -z "$best_path" || -f "$best_path" ]] || { printf 'best checkpoint is missing\n' >&2; exit 2; }
[[ -z "$resume_path" && -z "$best_path" || -n "$resume_path" && -n "$best_path" ]] || {
    printf 'resume and best checkpoints must be supplied together\n' >&2
    exit 2
}
if [[ "$resume_path" == "$run_dir/checkpoints/last.ckpt" || \
    "$best_path" == "$run_dir/checkpoints/best.ckpt" ]]; then
    printf 'resume checkpoints must be staged outside the output directory\n' >&2
    exit 2
fi

training_state="$(uv run --locked --package pixel-vae --extra training python - \
    "$data_dir/count.json" "$resume_path" "$batch_size" "$cpu_rehearsal" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import torch

counts_path, resume_path, batch_text, cpu_rehearsal = sys.argv[1:]
counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
train_steps = math.ceil(counts["train"] / int(batch_text))
start_epoch = 0
start_global_step = 0
if resume_path:
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("optimizer_states"):
        raise SystemExit("resume checkpoint has no optimizer state")

    start_global_step = int(checkpoint["global_step"])
    if start_global_step % train_steps:
        raise SystemExit("resume checkpoint is not at a train-epoch boundary")
    if cpu_rehearsal == "1":
        cpu_start_epoch = os.environ.get("PIXELVAE_CPU_START_EPOCH")
        if cpu_start_epoch is None:
            raise SystemExit("CPU resume requires PIXELVAE_CPU_START_EPOCH")
        start_epoch = int(cpu_start_epoch)
    else:
        start_epoch = start_global_step // train_steps

print(start_epoch, start_global_step, train_steps)
PY
)"
read -r start_epoch start_global_step train_steps <<< "$training_state"
((end_epoch > start_epoch)) || {
    printf 'segment endpoint must follow the checkpoint epoch\n' >&2
    exit 2
}
target_global_step=$((start_global_step + (end_epoch - start_epoch) * train_steps))
if [[ "$cpu_rehearsal" != 1 && "$start_global_step" -ne "$((start_epoch * train_steps))" ]]; then
    printf 'checkpoint global step does not match its epoch boundary\n' >&2
    exit 2
fi
if [[ "$cpu_rehearsal" == 1 ]]; then
    [[ "$end_epoch" == "$((start_epoch + 1))" && "$end_epoch" -le 250 ]] || {
        printf 'CPU rehearsal must save one epoch after its loaded checkpoint\n' >&2
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
    expected_end=$((start_epoch == 0 ? 10 : start_epoch + 68))
    ((expected_end <= 250)) || expected_end=250
    [[ "$end_epoch" == "$expected_end" ]] || {
        printf 'production segment must end at the next boundary, epoch %s\n' "$expected_end" >&2
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

gpu_samples="$run_dir/reports/gpu-samples.csv"
runtime_path="$run_dir/reports/runtime-e$(printf '%03d' "$end_epoch").json"
epoch_timing_path="$run_dir/reports/epoch-times-e$(printf '%03d' "$end_epoch").jsonl"
lockstep_path="$run_dir/reports/lockstep-e$(printf '%03d' "$end_epoch").jsonl"
lockstep_summary_path="${lockstep_path%.jsonl}.summary.json"
sampler_pid=''
if [[ -e "$runtime_path" || -e "$epoch_timing_path" || -e "$gpu_samples" \
    || -e "$lockstep_path" || -e "$lockstep_summary_path" ]]; then
    printf 'refusing to overwrite existing segment telemetry\n' >&2
    exit 2
fi
run_logged uv run --locked --no-sync --package pixel-vae --extra training python - \
    "$runtime_path" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

runtime_path = Path(sys.argv[1])
nproc = int(subprocess.check_output(["nproc"], text=True).strip())
visible_devices = torch.cuda.device_count()
runtime = {
    "os_cpu_count": os.cpu_count(),
    "nproc": nproc,
    "cuda_visible_device_count": visible_devices,
}
if visible_devices == 1:
    runtime["gpu_name"] = torch.cuda.get_device_name(0)
runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
print(json.dumps(runtime, sort_keys=True))
PY

if [[ "$start_epoch" == 0 || "$run_lockstep" == 1 ]]; then
    lockstep_steps=300
    lockstep_device=cuda
    if [[ "$cpu_rehearsal" == 1 ]]; then
        lockstep_steps=1
        lockstep_device=cpu
    fi

    vendor_commit=bc1e2072ba3a253f1b099e8b0c604f6051e787da
    vendor_dir="$repo_dir/vendor/canvas-vae"
    vendor_model="$vendor_dir/src/pixel-vae/pixelvae/model.py"
    if [[ ! -f "$vendor_model" ]]; then
        if [[ -e "$vendor_dir" ]]; then
            rmdir "$vendor_dir" || {
                printf 'vendor CanvasVAE directory exists without its pinned model and is not empty\n' >&2
                exit 2
            }
        fi
        vendor_archive="$work_root/canvas-vae-${vendor_commit}.tar.gz"
        mkdir -p "$vendor_dir"
        run_logged curl --fail --silent --show-error --location --retry 3 \
            --output "$vendor_archive" \
            "https://github.com/CyberAgentAILab/canvas-vae/archive/${vendor_commit}.tar.gz"
        run_logged tar -xzf "$vendor_archive" --strip-components=1 -C "$vendor_dir"
    fi

    run_logged uv sync --locked --package pixel-vae --extra training --extra vendor --group dev
    export TF_CPP_MIN_LOG_LEVEL=2
    export TF_ENABLE_ONEDNN_OPTS=0
    run_logged env PYTHONPATH="$repo_dir/models/pixel-vae/tests" \
        uv run --locked --no-sync --package pixel-vae --extra training \
        --group dev python -m vendor_parity.run_real_scale_lockstep \
        --data-dir "$data_dir" \
        --batch-size "$batch_size" \
        --num-workers 6 \
        --steps "$lockstep_steps" \
        --device "$lockstep_device" \
        --record-only \
        --output "$lockstep_path"
    [[ -f "$lockstep_path" && -f "$lockstep_summary_path" ]] || {
        printf 'lockstep completed without its evidence files\n' >&2
        exit 2
    }
fi

best_output_path="$run_dir/checkpoints/best.ckpt"
if [[ -n "$best_path" && ! -e "$best_output_path" && ! -L "$best_output_path" ]]; then
    cp -- "$best_path" "$best_output_path"
fi
if [[ -n "$best_path" && ( ! -f "$best_output_path" || -L "$best_output_path" ) ]]; then
    printf 'resume best checkpoint was not staged as a regular output file\n' >&2
    exit 2
fi

export PIXELVAE_EPOCH_TIMING_FILE="$epoch_timing_path"
stop_sampler() {
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=''
    fi
}
trap stop_sampler EXIT
if [[ "$cpu_rehearsal" != 1 ]]; then
    run_logged nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
    nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits -l 1 > "$gpu_samples" 2>>"$private_log" &
    sampler_pid=$!
    sleep 2
    kill -0 "$sampler_pid" 2>/dev/null || { printf 'GPU sampling failed to start\n' >&2; exit 2; }
    [[ -s "$gpu_samples" ]] || { printf 'GPU sampling produced no measurements\n' >&2; exit 2; }
fi

if [[ -z "$resume_path" && -e "$run_dir/checkpoints/last.ckpt" ]]; then
    printf 'refusing to start without a resume checkpoint over an existing run\n' >&2
    exit 2
fi
time_file="$private_dir/train-time.txt"
fit_args=(
    --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
    "--trainer.accelerator=$accelerator"
    --trainer.devices=1
    # Lightning resumes an epoch-boundary checkpoint before max_steps advances.
    # Keep one epoch of headroom; max_steps still bounds the requested work.
    "--trainer.max_epochs=$((end_epoch + 1))"
    "--trainer.max_steps=$target_global_step"
    "--trainer.default_root_dir=$run_dir"
    --trainer.logger=false
    --trainer.enable_progress_bar=false
    "--data.init_args.data_dir=$data_dir"
    "--data.init_args.batch_size=$batch_size"
    "--data.init_args.sampler_start_epoch=$start_epoch"
)
if [[ -n "$resume_path" ]]; then
    fit_args+=(--ckpt_path "$resume_path")
fi
TIMEFORMAT='%3R %P'
set +e
{ time uv run --locked --no-sync --package pixel-vae --extra training traingen fit \
    "${fit_args[@]}" >>"$private_log" 2>&1; } 2>"$time_file"
train_status=$?
set -e
stop_sampler
if ((train_status != 0)); then
    emit_scanned_log || true
    printf 'training segment failed with exit code %s\n' "$train_status" >&2
    exit "$train_status"
fi
if [[ -f "$run_dir/config.yaml" ]]; then
    config_archive="$run_dir/reports/config-e$(printf '%03d' "$end_epoch").yaml"
    [[ ! -e "$config_archive" ]] || { printf 'segment config archive already exists\n' >&2; exit 2; }
    mv "$run_dir/config.yaml" "$config_archive"
fi

report_path="$run_dir/reports/segment-e$(printf '%03d' "$end_epoch").json"
[[ -f "$run_dir/checkpoints/last.ckpt" && ! -L "$run_dir/checkpoints/last.ckpt" ]] || {
    printf 'last checkpoint must be a regular file before it can be reported\n' >&2
    exit 2
}
run_logged uv run --locked --no-sync --package pixel-vae --extra training python - \
    "$data_dir/count.json" "$run_dir/checkpoints/last.ckpt" \
    "$run_dir/checkpoints/best.ckpt" "$start_epoch" "$end_epoch" \
    "$batch_size" "$time_file" "$gpu_samples" "$report_path" "$source_sha" \
    "$runtime_path" "$epoch_timing_path" "$start_global_step" "$train_steps" \
    "$target_global_step" <<'PY'
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

(
    counts_path,
    last_path,
    best_path,
    start,
    end,
    batch,
    time_path,
    gpu_path,
    report_path,
    source,
    runtime_path,
    epoch_timing_path,
    start_global_step,
    train_steps,
    target_global_step,
) = sys.argv[1:]
counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
checkpoint_epoch = int(checkpoint["epoch"])
completed = int(end)
actual_global_step = int(checkpoint["global_step"])
if actual_global_step != int(target_global_step):
    raise SystemExit(
        f"last checkpoint global step is {actual_global_step}; "
        f"expected {target_global_step} from {start_global_step} plus "
        f"{int(end) - int(start)} epochs of {train_steps} steps"
    )
optimizer_states = checkpoint.get("optimizer_states")
if not isinstance(optimizer_states, list) or not optimizer_states:
    keys = ",".join(sorted(checkpoint))
    raise SystemExit(f"last checkpoint has no optimizer state; keys={keys}")
checkpoint_callbacks = checkpoint.get("callbacks", {})
best_scores = [
    state["best_model_score"]
    for state in checkpoint_callbacks.values()
    if state.get("monitor") == "val_reconstruction"
    and state.get("best_model_score") is not None
]
if len(best_scores) != 1:
    raise SystemExit("last checkpoint has no unique val_reconstruction best score")
best_validation = best_scores[0]
if isinstance(best_validation, torch.Tensor):
    best_validation = best_validation.item()
best_validation = float(best_validation)
if not math.isfinite(best_validation):
    raise SystemExit("best val_reconstruction score is not finite")
elapsed_text, cpu_text = Path(time_path).read_text(encoding="utf-8").split()
train_steps = math.ceil(counts["train"] / int(batch))
val_steps = math.ceil(counts["val"] / int(batch))
samples = []
if Path(gpu_path).is_file():
    with Path(gpu_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) >= 4:
                try:
                    samples.append(
                        (
                            row[0].strip(),
                            *(float(value.strip()) for value in row[1:4]),
                        )
                    )
                except ValueError:
                    continue
gpu = None
if samples:
    gpu = {
        "sample_count": len(samples),
        "mean_utilization_percent": sum(row[1] for row in samples) / len(samples),
        "mean_power_watts": sum(row[2] for row in samples) / len(samples),
        "mean_power_limit_watts": sum(row[3] for row in samples) / len(samples),
        "mean_power_limit_fraction": sum(row[2] / row[3] for row in samples) / len(samples),
    }
runtime = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
epoch_times = [
    json.loads(line)
    for line in Path(epoch_timing_path).read_text(encoding="utf-8").splitlines()
    if line
]
expected_sampler_epochs = list(range(int(start), int(end)))
actual_sampler_epochs = [int(epoch["sampler_epoch"]) for epoch in epoch_times]
actual_epochs = [int(epoch["epoch"]) for epoch in epoch_times]
if actual_sampler_epochs != expected_sampler_epochs:
    raise SystemExit(
        f"sampler epoch sequence is {actual_sampler_epochs}; "
        f"expected {expected_sampler_epochs}"
    )
if actual_epochs != list(range(int(start) + 1, int(end) + 1)):
    raise SystemExit(f"logical epoch sequence is {actual_epochs}")
for epoch in epoch_times:
    start_time = datetime.fromisoformat(epoch["started_at_utc"])
    end_time = datetime.fromisoformat(epoch["finished_at_utc"])
    epoch_samples = []
    for row in samples:
        try:
            sample_time = datetime.strptime(
                row[0], "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            sample_time = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        if start_time <= sample_time <= end_time:
            epoch_samples.append(row[1])
    epoch["gpu_utilization_sample_count"] = len(epoch_samples)
    epoch["mean_gpu_utilization_percent"] = (
        sum(epoch_samples) / len(epoch_samples) if epoch_samples else None
    )
report = {
    "source_commit": source,
    "segment": {
        "start_epoch": int(start),
        "end_epoch": int(end),
        "checkpoint_epoch": checkpoint_epoch,
        "start_global_step": int(start_global_step),
        "global_step": actual_global_step,
    },
    "data": {
        "train_unique_images": counts["train"],
        "validation_unique_images": counts["val"],
        "batch_size": int(batch),
        "train_steps_per_epoch": train_steps,
        "validation_steps_per_epoch": val_steps,
    },
    "validation": {"best_val_reconstruction": best_validation},
    "runtime": runtime,
    "epoch_telemetry": epoch_times,
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

for output_log in \
    "$private_log" "$gpu_samples" "$report_path" "$runtime_path" \
    "$epoch_timing_path" "$lockstep_path" "$lockstep_summary_path"; do
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
