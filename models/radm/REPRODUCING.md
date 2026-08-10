# Reproducing RADM

Workflow order: download or otherwise obtain local assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

## Prerequisites

- A local RADM checkpoint supplied by the user or maintainer.
- A local CGL dataset root and matching text-feature root.
- A local original RADM source checkout from
  [JD-GenX/RADM](https://github.com/JD-GenX/RADM) with `train_net.py` for the
  original model setup used by gated parity.
- One explicitly selected GPU for the original Detectron2 reference path.

The checked RADM README publishes dataset and testing-feature assets only; no
released checkpoint is published. Use a user-supplied local checkpoint for
conversion and heavyweight parity.

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
