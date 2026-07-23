# Reproducing RALF Parity

This guide reproduces the original-implementation agreement checks for the RALF package.

Workflow order: download assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/ralf` before generating vendor references. The regular package path and converter do not import Hydra or OmegaConf; the `vendor` optional extra installs the original implementation dependencies for reference generation and parity only. The command examples keep local artifacts under `.cache/ralf/` and never modify `vendor/ralf`.

Step 1 downloads and unpacks the original cache bundle. Generated metadata is `.cache/ralf/cache_manifest.json`.

```bash
uv run --package ralf --extra download python models/ralf/scripts/download_original_assets.py \
  --cache-dir .cache/ralf/cache \
  --zip-path .cache/ralf/cache.zip \
  --manifest .cache/ralf/cache_manifest.json \
  --download \
  --unzip
```

Step 2 generates golden vendor reference metadata and, when the vendor environment is available, runs vendor inference on GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package ralf --extra vendor python models/ralf/scripts/generate_reference_outputs.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --cache-dir .cache/ralf/cache \
  --output-dir .cache/ralf/references \
  --seed 0 \
  --split test \
  --gpu 0 \
  --condition-type uncond \
  --batch-size 1 \
  --run-vendor
```

Step 3 converts the CGL and PKU checkpoints to local [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style directories. The PKU original wrapper calls the label-size task `chw`; the converter normalizes both `chw` and `cwh` to canonical `label_size`.

```bash
uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --checkpoint .cache/ralf/cache/training_logs/ralf_uncond_cgl/gen_final_model.pt \
  --dataset cgl \
  --task unconditional \
  --vocabulary-json .cache/ralf/cache/dataset/cgl/vocabulary.json \
  --output-dir .cache/ralf/converted/ralf-cgl-unconditional-strict

uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
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
  uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
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
  uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
    --job-dir ".cache/ralf/cache/training_logs/${job}" \
    --checkpoint ".cache/ralf/cache/training_logs/${job}/gen_final_model.pt" \
    --dataset pku \
    --task "${task}" \
    --output-dir ".cache/ralf/converted/ralf-pku-${task//_/-}-strict"
done
```

Step 4 runs the gated parity tests against the generated references and converted checkpoints.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package ralf --extra vendor pytest models/ralf/tests/vendor_parity -m vendor_parity -q
```

Expected unconditional parity result from the existing reference set:

```text
6 passed
```

Step 5 runs a `from_pretrained` smoke check for a converted checkpoint.

```bash
uv run --package ralf python -c 'from ralf import RalfPipeline; pipe = RalfPipeline.from_pretrained(".cache/ralf/converted/ralf-cgl-unconditional-strict"); out = pipe(condition_type="unconditional", seed=0, top_k=5); print(pipe.config.vocab_size, out.bbox.shape, out.labels.shape, out.mask.shape)'
```
