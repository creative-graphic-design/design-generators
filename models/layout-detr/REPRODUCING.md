# Reproducing LayoutDETR

These commands reproduce LayoutDETR agreement checks against the original implementation by downloading the released assets, generating vendor references, running parity tests, converting the pickle, and smoke-testing local loading with `from_pretrained`.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

## Download Original Assets

```bash
uv run --package layout-detr --extra download python models/layout-detr/scripts/download_original_assets.py \
  --output-dir .cache/layout-detr/original \
  --include model
```

To include the Up-DETR preinit weights and the vendor Ad Banner dataset, use:

```bash
uv run --package layout-detr --extra download python models/layout-detr/scripts/download_original_assets.py \
  --output-dir .cache/layout-detr/original \
  --include all
```

## Generate Vendor References

Reference tensors are written outside git under `.cache/layout-detr/reference`.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package layout-detr --extra vendor python models/layout-detr/scripts/generate_reference_outputs.py \
  --vendor-root vendor/layout-detr \
  --checkpoint .cache/layout-detr/original/checkpoints/layoutdetr_ad_banner.pkl \
  --background "vendor/layout-detr/examples/Lumber 2 [header]EVERYTHING 10% OFF[body text]Friends & Family Savings Event[button]SHOP NOW[disclaimer]CODE FRIEND10.jpg" \
  --texts "EVERYTHING 10% OFF|Friends & Family Savings Event|SHOP NOW|CODE FRIEND10" \
  --labels "header|body text|button|disclaimer / footnote" \
  --seed 0 \
  --output-dir .cache/layout-detr/reference/lumber2
```

## Run Vendor Parity

```bash
CUDA_VISIBLE_DEVICES=1 LAYOUT_DETR_VENDOR_ROOT=vendor/layout-detr \
  LAYOUT_DETR_CHECKPOINT=.cache/layout-detr/original/checkpoints/layoutdetr_ad_banner.pkl \
  LAYOUT_DETR_REFERENCE_DIR=.cache/layout-detr/reference/lumber2 \
  uv run --package layout-detr --extra vendor --with pytest --with pytest-cov pytest models/layout-detr/tests/vendor_parity -m vendor_parity --no-cov
```

## Convert Checkpoint

```bash
uv run --package layout-detr --extra convert python models/layout-detr/scripts/convert_original_checkpoint.py \
  --vendor-root vendor/layout-detr \
  --checkpoint .cache/layout-detr/original/checkpoints/layoutdetr_ad_banner.pkl \
  --output-dir .cache/layout-detr/converted/layout-detr-ad-banner \
  --allow-partial
```

## Run From-Pretrained Smoke

```bash
uv run --package layout-detr python models/layout-detr/scripts/smoke_from_pretrained.py \
  --model-dir .cache/layout-detr/converted/layout-detr-ad-banner
```
