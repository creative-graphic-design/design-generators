# Reproducing House-GAN

This walkthrough reproduces House-GAN agreement checks against the original implementation using local Dropbox assets, fixed seeds, and generated external parity artifacts.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

## 1. Validate Original Assets

```bash
uv run --package housegan --extra download \
  python models/housegan/scripts/download_original_assets.py \
    --assets-dir .cache/housegan/original \
    --output-dir .cache/housegan/original
```

## 2. Generate Vendor References

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package housegan --extra vendor \
  python models/housegan/scripts/generate_reference_outputs.py \
    --vendor-dir ./vendor/housegan \
    --assets-dir .cache/housegan/original \
    --checkpoint .cache/housegan/original/checkpoints/exp_with_graph_global_new_D_200000.pth \
    --target-set D \
    --indices 0 6 42 \
    --seed 0 \
    --output-dir .cache/housegan/parity/housegan-floorplan-d
```

## 3. Run Vendor-Parity Tests

```bash
CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 uv run --package housegan \
  pytest models/housegan/tests -m vendor_parity
```

## 4. Convert Checkpoints

```bash
uv run --package housegan \
  python models/housegan/scripts/convert_original_checkpoint.py \
    --checkpoint .cache/housegan/original/checkpoints/exp_with_graph_global_new_D_200000.pth \
    --target-set D \
    --output-dir .cache/housegan/converted/housegan-floorplan-d
```

## 5. Smoke Test `from_pretrained`

```bash
uv run --package housegan \
  python models/housegan/scripts/smoke_from_pretrained.py \
    --checkpoint-dir .cache/housegan/converted/housegan-floorplan-d
```
