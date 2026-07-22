# Reproducing LayoutAction

These commands reproduce the agreement checks against the original LayoutAction implementation.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

## 1. Download Original Assets

```bash
uv run --package layout-action --extra download \
  python models/layout-action/scripts/download_original_assets.py \
    --output-dir .cache/layout-action/original \
    --download
```

For pre-downloaded assets:

```bash
uv run --package layout-action --extra download \
  python models/layout-action/scripts/download_original_assets.py \
    --source-dir /path/to/LayoutActionResources \
    --output-dir .cache/layout-action/original
```

## 2. Generate Vendor References

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-action --extra vendor \
  python models/layout-action/scripts/generate_reference_outputs.py \
    --dataset rico \
    --asset-dir .cache/layout-action/original \
    --output-dir .cache/layout-action/references \
    --seed 42 \
    --eval-command all \
    --num-batches 2
```

Repeat with `--dataset publaynet` to cover both released public checkpoints.

## 3. Run Vendor Parity

```bash
LAYOUT_ACTION_VENDOR_ROOT=vendor/layout-action/LayoutAction \
LAYOUT_ACTION_ASSET_DIR=.cache/layout-action/original \
LAYOUT_ACTION_REFERENCE_DIR=.cache/layout-action/references \
CUDA_VISIBLE_DEVICES=0 uv run --package layout-action pytest \
  models/layout-action/tests -m vendor_parity
```

## 4. Convert Checkpoints

```bash
uv run --package layout-action --extra convert \
  python models/layout-action/scripts/convert_original_checkpoint.py \
    --dataset rico \
    --checkpoint .cache/layout-action/original/pretrained_model_resources/Ours/rico.pth \
    --output-dir .cache/layout-action/converted/layout-action-rico13
```

For PubLayNet:

```bash
uv run --package layout-action --extra convert \
  python models/layout-action/scripts/convert_original_checkpoint.py \
    --dataset publaynet \
    --checkpoint .cache/layout-action/original/pretrained_model_resources/Ours/publaynet.pth \
    --output-dir .cache/layout-action/converted/layout-action-publaynet
```

## 5. Run `from_pretrained` Smoke

```bash
uv run --package layout-action \
  python models/layout-action/scripts/smoke_from_pretrained.py \
    .cache/layout-action/converted/layout-action-rico13
```
