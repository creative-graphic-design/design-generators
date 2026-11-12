#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
ENV="${CANVAS_VAE_ORIGINAL_ENV:?set CANVAS_VAE_ORIGINAL_ENV}"
TF_SITE="${CANVASVAE_TF_GPU_SITE:?set CANVASVAE_TF_GPU_SITE}"
INPUTS="${CANVAS_VAE_ORIGINAL_DATA:-${CANVAS_VAE_S5_INPUTS:-$ROOT/.cache/canvas-vae/s5-path-a/inputs}/original-data}"
GUARD="$ROOT/models/canvas-vae/scripts/source_tree_guard.py"
unset LD_LIBRARY_PATH PYTHONPATH CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
export CUDA_VISIBLE_DEVICES="$GPU" NVIDIA_TF32_OVERRIDE=0 TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$TF_SITE:$ROOT/models/canvas-vae/scripts:$ROOT/models/canvas-vae/src:$ROOT/lib/laygen/src:$ROOT/lib/traingen/src:$ROOT/lib/traingen-parity/src:$ROOT/vendor/canvas-vae/src/canvas-vae"
export CANVASVAE_TF_GPU_SITE="$TF_SITE" CANVAS_VAE_S4_TRAIN_STREAM=1
export CANVAS_VAE_PREFLIGHT_STACK="$RUN_ROOT/preflight/original-stack.json"
export CANVAS_VAE_PREFLIGHT_METADATA="$RUN_ROOT/preflight/original-run-metadata.json"
for required in "$ENV/bin/python" "$TF_SITE" "$INPUTS/train-00000-of-00001.tfrecord" "$INPUTS/val-00000-of-00001.tfrecord" "$INPUTS/test-00000-of-00001.tfrecord" "$INPUTS/vocabulary.json" "$GUARD" "$TOOL_ROOT/helpers/train_original_best.py" "$TOOL_ROOT/helpers/evaluate_original_best.py"; do
  [[ -e "$required" ]] || { printf 'missing original preflight input: %s\n' "$required" >&2; exit 1; }
done
[[ -n "$CANVASVAE_TF_GPU_SITE" && -d "$CANVASVAE_TF_GPU_SITE" ]]
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
uv pip freeze --python "$ENV/bin/python" > "$RUN_ROOT/preflight/original-pip-freeze-base.txt"
"$ENV/bin/python" - <<'PY' > "$RUN_ROOT/preflight/original-pip-freeze-live.txt"
import importlib.metadata
import re

packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        packages.setdefault(canonical, set()).add(distribution.version)
conflicts = {name: sorted(versions) for name, versions in packages.items() if len(versions) != 1}
if conflicts:
    raise SystemExit(f"duplicate package versions in the TensorFlow overlay: {conflicts}")
for name in sorted(packages):
    print(f"{name}=={next(iter(packages[name]))}")
PY
python3 "$TOOL_ROOT/helpers/check_live_freeze.py" --expected "${CANVAS_VAE_ORIGINAL_FREEZE_REFERENCE:?set CANVAS_VAE_ORIGINAL_FREEZE_REFERENCE}" --actual "$RUN_ROOT/preflight/original-pip-freeze-live.txt" --copy-to "$RUN_ROOT/stack/tensorflow-pip-freeze.txt"
export CANVAS_VAE_PIP_FREEZE_FILE="$RUN_ROOT/stack/tensorflow-pip-freeze.txt"
"$ENV/bin/python" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
from generate_vendor_reference import disable_tensorflow_tf32, import_original

tf, spec, _ = import_original()
disable_tensorflow_tf32(tf)
gpu_devices = tf.config.list_physical_devices("GPU")
assert len(gpu_devices) == 1, gpu_devices
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from canvasvae.train import train, evaluate
from canvas_vae.update_trace import install_tensorflow_s4_train_stream, install_tensorflow_training_trace
from canvas_vae.run_metadata import write_run_metadata
with tf.device("/GPU:0"):
    value = tf.ones((128, 128), dtype=tf.float32)
    result = tf.matmul(value, value)
assert float(result[0, 0].numpy()) == 128.0
record = {
    "framework": f"tensorflow {tf.__version__}",
    "cuda_runtime": tf.sysconfig.get_build_info().get("cuda_version"),
    "cudnn": tf.sysconfig.get_build_info().get("cudnn_version"),
    "nccl": importlib.metadata.version("nvidia-nccl-cu12"),
    "numpy": importlib.metadata.version("numpy"),
    "gpu": tf.config.experimental.get_device_details(gpu_devices[0]).get("device_name"),
    "vendor_data_spec": spec.DataSpec.__name__,
    "trainer": train.__name__,
    "evaluator": evaluate.__name__,
    "gpu_matmul": float(result[0, 0].numpy()),
}
Path(os.environ["CANVAS_VAE_PREFLIGHT_STACK"]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
write_run_metadata(os.environ["CANVAS_VAE_PREFLIGHT_METADATA"], "tensorflow")
print(json.dumps(record, sort_keys=True), flush=True)
PY
sha256sum "$RUN_ROOT/stack/tensorflow-pip-freeze.txt" > "$RUN_ROOT/preflight/original-freeze.sha256"
printf 'PASS original preflight\n' > "$RUN_ROOT/preflight/original.passed"
