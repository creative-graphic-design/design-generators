# Reproducing Flex-DM Conversion

This walkthrough creates original-implementation references, converts the released Crello and RICO checkpoints, compares the package against each reference, and runs the required vendor-parity suite. Run every command from the repository root.

The walkthrough stores downloads, extracted data, references, converted checkpoints, and layer probes under `.cache/flex-dm`. The Crello data archive is 3,661,085,903 bytes before extraction, so check available disk space before downloading it. Choose one free CUDA device with `scripts/pick_free_gpus.sh 1`; set `GPU_ID` to a specific available index before running the commands to select that device explicitly.

Workflow order: download assets, generate original-implementation references, convert checkpoints, compare package outputs, run parity tests, and smoke-test local `from_pretrained` loading.

The required environment uses the `data` and `vendor-tf` extras and the `dev` dependency group. The `dev` group installs pytest. In the verified CUDA 13 PyTorch and TensorFlow 2.15.1 stack, TensorFlow installs `nvidia-nccl-cu12`, whose library path conflicts with PyTorch's CUDA 13 NCCL. Remove only `nvidia-nccl-cu12`, reinstall `nvidia-nccl-cu13`, and preserve TensorFlow's other CUDA 12 packages. The later `uv run --no-sync` commands keep uv from restoring the conflicting package.

```bash
GPU_ID="${GPU_ID:-$(scripts/pick_free_gpus.sh 1)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NVIDIA_TF32_OVERRIDE=0

git submodule update --init vendor/flex-dm
uv sync --locked --package flex-dm --group dev --extra data --extra vendor-tf
flex_dm_python="$(uv run --locked --no-sync --package flex-dm python -c 'import sys; print(sys.executable)')"
uv pip uninstall --python "$flex_dm_python" nvidia-nccl-cu12
uv pip install --python "$flex_dm_python" --reinstall nvidia-nccl-cu13==2.29.7

# Download weights and confirm space before the multi-gigabyte Crello archive.
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
df -h .
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset crello --assets data
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset rico --assets data

# Inspect the original TensorFlow checkpoints.
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/crello/crello/ours-exp-ft --output-json .cache/flex-dm/reports/crello-ours-exp-ft-tf-vars.json
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/rico/rico/ours-exp-ft --output-json .cache/flex-dm/reports/rico-ours-exp-ft-tf-vars.json

# Generate original-implementation forward references with TensorFlow TF32 disabled.
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/generate_reference_outputs.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/crello-ours-exp-ft --seed 0 --tasks elem,pos,attr,img,txt --num-iter 1,4 --batch-size 1 --max-steps 1 --disable-tf32
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/generate_reference_outputs.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/rico-ours-exp-ft --seed 0 --tasks elem,pos,attr --num-iter 1,4 --batch-size 1 --max-steps 1 --disable-tf32

# Convert both released checkpoints and verify local loading.
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/convert_original_checkpoint.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-rico
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-crello
uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-rico

# Export original-implementation layer probes, then compare them with PyTorch.
for task in elem pos attr img txt; do uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/probe_layer_parity.py --mode export-vendor --dataset crello --task "$task" --probe-dir ".cache/flex-dm/probes/crello-$task" --disable-tf32; done
for task in elem pos attr; do uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/probe_layer_parity.py --mode export-vendor --dataset rico --task "$task" --probe-dir ".cache/flex-dm/probes/rico-$task" --disable-tf32; done
for task in elem pos attr img txt; do uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/probe_layer_parity.py --mode compare-torch --dataset crello --task "$task" --probe-dir ".cache/flex-dm/probes/crello-$task"; done
for task in elem pos attr; do uv run --locked --no-sync --package flex-dm models/flex-dm/scripts/probe_layer_parity.py --mode compare-torch --dataset rico --task "$task" --probe-dir ".cache/flex-dm/probes/rico-$task"; done

# Run all 17 checks and fail if any local parity asset is missing.
PARITY_REQUIRE=1 uv run --package flex-dm --locked --no-sync --group dev pytest models/flex-dm/tests/vendor_parity -m vendor_parity
```

The forward reference command writes `reference_results.json` and per-case arrays under `forward_cases/`. The required pytest run executes every saved forward step and layer probe; all 17 checks must pass without skips.
