# Reproducing RALF Parity

This guide reproduces the original-implementation agreement checks for the RALF package.

Workflow order: download assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/ralf` before generating vendor references. The regular package path and converter do not import Hydra or OmegaConf; the `vendor` optional extra installs the original implementation dependencies for reference generation and parity only. The command examples keep local artifacts under `.cache/ralf/` and never modify `vendor/ralf`.

Step 1 downloads and unpacks the original cache bundle. Generated metadata is `.cache/ralf/cache_manifest.json`.

```bash
uv run --package ralf --extra download models/ralf/scripts/download_original_assets.py \
  --cache-dir .cache/ralf/cache \
  --zip-path .cache/ralf/cache.zip \
  --manifest .cache/ralf/cache_manifest.json \
  --download \
  --unzip
```

Step 2 generates golden vendor reference metadata and, when the vendor environment is available, runs vendor inference on one selected GPU.

```bash
CUDA_VISIBLE_DEVICES="${RALF_GPU:?set RALF_GPU}" uv run --package ralf --extra vendor models/ralf/scripts/generate_reference_outputs.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --cache-dir .cache/ralf/cache \
  --output-dir .cache/ralf/references \
  --seed 0 \
  --split test \
  --gpu "${RALF_GPU:?set RALF_GPU}" \
  --condition-type uncond \
  --batch-size 1 \
  --run-vendor
```

Step 3 converts the CGL and PKU checkpoints to local [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style directories. The PKU original wrapper calls the label-size task `chw`; the converter normalizes both `chw` and `cwh` to canonical `label_size`.

```bash
uv run --package ralf --extra vendor models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --checkpoint .cache/ralf/cache/training_logs/ralf_uncond_cgl/gen_final_model.pt \
  --dataset cgl \
  --task unconditional \
  --vocabulary-json .cache/ralf/cache/dataset/cgl/vocabulary.json \
  --output-dir .cache/ralf/converted/ralf-cgl-unconditional-strict

uv run --package ralf --extra vendor models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_pku10 \
  --checkpoint .cache/ralf/cache/training_logs/ralf_uncond_pku10/gen_final_model.pt \
  --dataset pku \
  --task unconditional \
  --output-dir .cache/ralf/converted/ralf-pku-unconditional-strict
```

Convert the remaining task checkpoints with the same strict-load converter.

```bash
declare -A RALF_CGL_TASKS=(
  [label]=c
  [label_size]=chw
  [completion]=partial
  [refinement]=refinement
  [relation]=relation
)

for task in label label_size completion refinement relation; do
  job="ralf_${RALF_CGL_TASKS[$task]}_cgl"
  uv run --package ralf --extra vendor models/ralf/scripts/convert_original_checkpoint.py \
    --job-dir ".cache/ralf/cache/training_logs/${job}" \
    --checkpoint ".cache/ralf/cache/training_logs/${job}/gen_final_model.pt" \
    --dataset cgl \
    --task "${task}" \
    --vocabulary-json .cache/ralf/cache/dataset/cgl/vocabulary.json \
    --output-dir ".cache/ralf/converted/ralf-cgl-${task//_/-}-strict"
done

declare -A RALF_PKU_TASKS=(
  [label]=c
  [label_size]=chw
  [completion]=partial
  [refinement]=refinement
  [relation]=relation
)

for task in label label_size completion refinement relation; do
  job="ralf_${RALF_PKU_TASKS[$task]}_pku10"
  uv run --package ralf --extra vendor models/ralf/scripts/convert_original_checkpoint.py \
    --job-dir ".cache/ralf/cache/training_logs/${job}" \
    --checkpoint ".cache/ralf/cache/training_logs/${job}/gen_final_model.pt" \
    --dataset pku \
    --task "${task}" \
    --output-dir ".cache/ralf/converted/ralf-pku-${task//_/-}-strict"
done
```

Step 4 runs the gated parity tests against the generated references and converted checkpoints.

```bash
CUDA_VISIBLE_DEVICES="${RALF_GPU:?set RALF_GPU}" uv run --package ralf --extra vendor pytest models/ralf/tests/vendor_parity -m vendor_parity -q
```

Expected unconditional parity result from the existing reference set:

```text
6 passed
```

## Evaluation-path parity runner

Run `models/ralf/tests/vendor_parity/run_condition_parity.py` to compare one condition's package pipeline and vendor evaluator on the full TEST split. Set `RALF_DATA_ROOT` to the directory containing the selected `cgl/` or `pku/` dataset, `RALF_CAMPAIGN_ROOT` to that condition's S5 campaign root, `RALF_PARITY_OUTPUT_ROOT` to a new scratch directory, and `RALF_GPU` to the selected GPU. Keep these machine-specific values in environment variables. For CGL unconditional, also set `RALF_CGL_UNCONDITIONAL_VENDOR_JOB` to the released vendor job directory; other conditions use the vendor job under the selected campaign root.

Set `RALF_DATASET` and `RALF_CONDITION` from this table before running the command. Non-relation runs omit `RALF_PARITY_SEED` and use the runner's default inference seeds `0`, `1`, and `2` with cuDNN disabled. Run relation once per seed, setting `RALF_PARITY_SEED` to `0`, `1`, or `2`; relation runs keep cuDNN enabled.

| Dataset | Condition     | `RALF_DATASET` | `RALF_CONDITION` | Additional arguments                                                            |
| ------- | ------------- | -------------- | ---------------- | ------------------------------------------------------------------------------- |
| CGL     | unconditional | `cgl`          | `unconditional`  | `--vendor-job-path "$RALF_CGL_UNCONDITIONAL_VENDOR_JOB"`; non-relation defaults |
| CGL     | label         | `cgl`          | `label`          | non-relation defaults                                                           |
| CGL     | label-size    | `cgl`          | `label-size`     | non-relation defaults                                                           |
| CGL     | completion    | `cgl`          | `completion`     | non-relation defaults                                                           |
| CGL     | refinement    | `cgl`          | `refinement`     | non-relation defaults                                                           |
| CGL     | relation      | `cgl`          | `relation`       | `--seed 0`, `--seed 1`, and `--seed 2` in separate runs                         |
| PKU     | unconditional | `pku`          | `unconditional`  | non-relation defaults                                                           |
| PKU     | label         | `pku`          | `label`          | non-relation defaults                                                           |
| PKU     | label-size    | `pku`          | `label-size`     | non-relation defaults                                                           |
| PKU     | completion    | `pku`          | `completion`     | non-relation defaults                                                           |
| PKU     | refinement    | `pku`          | `refinement`     | non-relation defaults                                                           |
| PKU     | relation      | `pku`          | `relation`       | `--seed 0`, `--seed 1`, and `--seed 2` in separate runs                         |

Use a fresh output directory for each invocation. Relation seed runs write to separate `seed-<N>` subdirectories under that output root.

```bash
set -e

dataset="${RALF_DATASET:?set RALF_DATASET from the table}"
condition="${RALF_CONDITION:?set RALF_CONDITION from the table}"
dataset_root="${RALF_DATA_ROOT:?set RALF_DATA_ROOT}"
campaign_root="${RALF_CAMPAIGN_ROOT:?set RALF_CAMPAIGN_ROOT for this condition}"
output_root="${RALF_PARITY_OUTPUT_ROOT:?set RALF_PARITY_OUTPUT_ROOT}/$dataset/$condition"
gpu="${RALF_GPU:?set RALF_GPU}"
condition_args=(--disable-cudnn)
vendor_args=()

if [[ "$condition" == relation ]]; then
  seed="${RALF_PARITY_SEED:?set RALF_PARITY_SEED to 0, 1, or 2}"
  condition_args=(--seed "$seed")
  output_root="$output_root/seed-$seed"
fi

if [[ "$dataset" == cgl && "$condition" == unconditional ]]; then
  vendor_args=(--vendor-job-path "${RALF_CGL_UNCONDITIONAL_VENDOR_JOB:?set RALF_CGL_UNCONDITIONAL_VENDOR_JOB}")
fi

CUDA_VISIBLE_DEVICES="$gpu" .cache/ralf/runtime/audit-cu128-cu12/bin/python \
  models/ralf/tests/vendor_parity/run_condition_parity.py \
  --dataset-root "$dataset_root" \
  --dataset "$dataset" \
  --condition "$condition" \
  --campaign-root "$campaign_root" \
  --output-root "$output_root" \
  --runtime-freeze-path .cache/ralf/runtime/audit-cu128-cu12-records/runtime-freeze.txt \
  --gpu "$gpu" \
  "${vendor_args[@]}" \
  "${condition_args[@]}"
```

Step 5 runs a `from_pretrained` smoke check for a converted checkpoint.

```bash
uv run --package ralf python -c 'from ralf import RalfPipeline; pipe = RalfPipeline.from_pretrained(".cache/ralf/converted/ralf-cgl-unconditional-strict"); out = pipe(condition_type="unconditional", seed=0, top_k=5); print(pipe.config.vocab_size, out.bbox.shape, out.labels.shape, out.mask.shape)'
```
