#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${CANVAS_VAE_S5_RUN_ROOT:-$ROOT/.cache/canvas-vae/full-run/crello/path-a}"
GPU="${CANVAS_VAE_GPU:?set CANVAS_VAE_GPU}"
EXPECTED_COMMIT="${CANVAS_VAE_S5_SOURCE_COMMIT:-58f9338e40cacb448cb45e400d600e8a5b616c8a}"
ENV="${CANVAS_VAE_PACKAGE_ENV:?set CANVAS_VAE_PACKAGE_ENV}"
INPUTS="${CANVAS_VAE_S5_INPUTS:-$ROOT/.cache/canvas-vae/s5-path-a/inputs}"
DATA="${CANVAS_VAE_PACKAGE_DATA:-$INPUTS/package-data}"
CACHE="${CANVAS_VAE_EMBEDDING_CACHE:-$INPUTS/cache}"
GUARD="$ROOT/models/canvas-vae/scripts/source_tree_guard.py"
CONFIG="$ROOT/models/canvas-vae/configs/training/canvas_vae_crello.yaml"
unset LD_LIBRARY_PATH PYTHONPATH CANVAS_VAE_UPDATE_TRACE CANVAS_VAE_UPDATE_TRACE_SEED
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="$ROOT/models/canvas-vae/src:$ROOT/lib/laygen/src:$ROOT/lib/traingen/src:$ROOT/lib/traingen-parity/src"
export CANVAS_VAE_PREFLIGHT_STACK="$RUN_ROOT/preflight/package-stack.json"
export CANVAS_VAE_PREFLIGHT_METADATA="$RUN_ROOT/preflight/package-run-metadata.json"
for required in "$ENV/bin/python" "$ENV/bin/traingen" "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/test.jsonl" "$DATA/vocabulary.json" "$CACHE/manifest.json" "$CACHE/posterior_means.npy" "$GUARD" "$CONFIG"; do
  [[ -f "$required" || -x "$required" ]] || { printf 'missing package preflight input: %s\n' "$required" >&2; exit 1; }
done
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
uv pip freeze --python "$ENV/bin/python" > "$RUN_ROOT/preflight/package-pip-freeze-live.txt"
python3 "$TOOL_ROOT/helpers/check_live_freeze.py" --expected "${CANVAS_VAE_PACKAGE_FREEZE_REFERENCE:?set CANVAS_VAE_PACKAGE_FREEZE_REFERENCE}" --actual "$RUN_ROOT/preflight/package-pip-freeze-live.txt" --copy-to "$RUN_ROOT/stack/package-pip-freeze.txt"
export CANVAS_VAE_PIP_FREEZE_FILE="$RUN_ROOT/stack/package-pip-freeze.txt"
"$ENV/bin/traingen" fit --help > "$RUN_ROOT/preflight/package-cli-help.txt"
"$ENV/bin/python" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
import torch

assert importlib.metadata.version("torch") == "2.13.0+cu126"
assert np.__version__ == "2.4.6", np.__version__
assert torch.cuda.is_available(), "Torch found no CUDA device"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
from canvas_vae.training import CanvasVAECrelloDataModule, CanvasVAECrelloTrainingModule
from canvas_vae.run_metadata import write_run_metadata
value = torch.ones((256, 256), device="cuda")
assert torch.equal(value @ value.T, torch.full((256, 256), 256.0, device="cuda"))
torch.cuda.synchronize()
record = {
    "framework": f"torch {torch.__version__}",
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "nccl": torch.cuda.nccl.version(),
    "numpy": np.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "training_module": CanvasVAECrelloTrainingModule.__name__,
    "data_module": CanvasVAECrelloDataModule.__name__,
}
Path(os.environ["CANVAS_VAE_PREFLIGHT_STACK"]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
write_run_metadata(os.environ["CANVAS_VAE_PREFLIGHT_METADATA"], "torch")
print(json.dumps(record, sort_keys=True), flush=True)
PY
sha256sum "$RUN_ROOT/stack/package-pip-freeze.txt" > "$RUN_ROOT/preflight/package-freeze.sha256"
printf 'PASS package preflight\n' > "$RUN_ROOT/preflight/package.passed"
