# Reproducing RADM

Workflow order: download or otherwise obtain local assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

## Prerequisites

- A local RADM checkpoint supplied by the user or maintainer.
- A local CGL dataset root and matching text-feature root.
- A local original RADM source checkout with `train_net.py` for gated parity.
- One explicitly selected GPU for the original Detectron2 reference path.

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
