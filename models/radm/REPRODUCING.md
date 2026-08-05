# Reproducing RADM

Workflow order: download or otherwise obtain local assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

## Prerequisites

- A local RADM checkpoint supplied by the user or maintainer.
- A local CGL dataset root and matching text-feature root.
- A local original RADM source checkout from
  [JD-GenX/RADM](https://github.com/JD-GenX/RADM) with `train_net.py` for gated
  parity.
- One explicitly selected GPU for the original Detectron2 reference path.

The checked RADM README publishes dataset and testing-feature assets only; no
released checkpoint is published. Use a locally trained or otherwise
user-supplied checkpoint for conversion and heavyweight parity.

Detectron2 is required for original-code RADM instantiation and is included in
the `vendor` extra. The RADM extra pins the upstream `v0.6` source tag; install
the vendor extra in the original-code reference environment before running
denoiser architecture parity:

```bash
uv sync --package radm --extra vendor
```

If the local CUDA toolkit does not match the installed PyTorch CUDA build, use
`CUDA_VISIBLE_DEVICES="" uv sync --package radm --extra vendor` to build
Detectron2 without CUDA extensions for the architecture-parity check.

## Regenerate the Vendor Reference Checkpoint

The helper below materializes CGL-v2 ralf-style Parquet shards into the COCO
tree expected by the checked RADM training script, then launches the original
Detectron2 trainer with fixed seed and explicit GPU selection. It writes only
one rolling checkpoint plus `radm_cgl_vendor_final.pth` below
`$RADM_CACHE_ROOT/vendor-runs`; when `RADM_CACHE_ROOT` is unset, the scripts use
`$HOME/.cache/radm`.

For a launch-plan check on a tiny materialized subset:

```bash
models/radm/scripts/train_vendor_reference.sh \
  --gpu 0 \
  --num-gpus 1 \
  --seed 0 \
  --max-iter 20 \
  --plan-only \
  --overwrite
```

For the full CGL-v2 S5-aligned vendor reference run:

```bash
models/radm/scripts/train_vendor_reference.sh \
  --gpu 0,1,2,3 \
  --num-gpus 4 \
  --seed 0 \
  --max-iter 250000 \
  --run-id radm-cgl-v2-vendor-s5 \
  --full-data \
  --overwrite
```

After the run completes, use the final checkpoint as the original checkpoint for
reference generation and conversion:

```bash
mkdir -p .cache/radm/original
cp "${RADM_CACHE_ROOT:-$HOME/.cache/radm}/vendor-runs/radm-cgl-v2-vendor-s5/radm_cgl_vendor_final.pth" \
  .cache/radm/original/radm_cgl.pth
```

The package-side train-ourselves entrypoint uses the `traingen` console script:

```bash
uv run --package radm --extra training traingen fit \
  --config models/radm/configs/training/s5_cgl_v2.yaml
```

## Inspect Local Assets

```bash
uv run --package radm python models/radm/scripts/inspect_original_checkpoint.py \
  --checkpoint .cache/radm/original/radm_cgl.pth
```

## Generate Reference Outputs

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package radm python models/radm/scripts/generate_reference_outputs.py \
  --vendor-root ./vendor/radm \
  --checkpoint .cache/radm/original/radm_cgl.pth \
  --dataset-root .cache/radm/data/cgl \
  --text-feature-root .cache/radm/text_features/cgl \
  --output-dir .cache/radm/reference/cgl \
  --seed 1 \
  --device cuda:0
```

The generated reference directory should contain metadata and future golden tensors outside git. Do not commit generated references, weights, or images.

## Convert Checkpoints

```bash
uv run --package radm python models/radm/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/radm/original/radm_cgl.pth \
  --dataset-name cgl \
  --output-dir .cache/radm/converted/cgl
```

## Run Parity Checks

Run the source-level parity checks that do not require released weights:

```bash
RADM_VENDOR_ROOT=./vendor/radm \
uv run --package radm pytest models/radm/tests/vendor_parity -m vendor_parity -rs
```

These checks compare the cosine beta schedule, forward diffusion `q_sample`,
DDIM coefficients, and focal-style postprocessing order against the checked
RADM source. The denoiser architecture check requires a local Detectron2 build
because the checked `RADM/head.py` DynamicHead depends on Detectron2 RoI pooling;
with `PARITY_REQUIRE=1`, that missing dependency fails loudly instead of being
reported as completed parity.

```bash
PARITY_REQUIRE=1 \
RADM_ORIGINAL_CHECKPOINT=.cache/radm/original/radm_cgl.pth \
RADM_REFERENCE_DIR=.cache/radm/reference/cgl \
RADM_VENDOR_ROOT=./vendor/radm \
uv run --package radm pytest models/radm/tests/vendor_parity -m vendor_parity
```

Without `PARITY_REQUIRE=1`, missing local assets skip cleanly:

```bash
uv run --package radm pytest models/radm/tests/vendor_parity -m vendor_parity
```

## Smoke-Test Local Loading

```bash
uv run --package radm python models/radm/scripts/smoke_from_pretrained.py \
  --path .cache/radm/converted/cgl
```

## CI-Safe Unit Tests

```bash
uv run --package radm pytest models/radm/tests -m "not vendor_parity and not integration" \
  --cov=radm --cov-report=term-missing
```
