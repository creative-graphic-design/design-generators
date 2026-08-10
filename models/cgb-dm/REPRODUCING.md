# Reproducing CGB-DM Checks

The staged parity workflow defined in [TRAINING.md](models/cgb-dm/TRAINING.md) does not download the 13 GB dataset during tests.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

### 1. Download Original Assets

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/download_original_assets.py
```

### 2. Generate Golden Reference Metadata

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package cgb-dm python models/cgb-dm/scripts/generate_reference_outputs.py \
  --dataset pku_posterlayout \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --manifest-output .cache/cgb-dm/reference/pku_posterlayout_train_manifest.json
```

### 3. Run Vendor Parity Tests

```bash
PARITY_REQUIRE=1 \
CGB_DM_DATA_ROOT=.cache/cgb-dm/datasets/pku/split \
CGB_DM_VENDOR_ORDER_MANIFEST=.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json \
CUDA_VISIBLE_DEVICES=0 \
uv run --package cgb-dm --extra vendor --with pytz \
  pytest models/cgb-dm/tests/vendor_parity -m vendor_parity
```

### 4. Convert Checkpoints And Smoke Test from_pretrained

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/convert_training_checkpoint.py --checkpoint .cache/cgb-dm/checkpoints/example.ckpt --output-dir .cache/cgb-dm/converted/pku
uv run --package cgb-dm python models/cgb-dm/scripts/smoke_from_pretrained.py --path .cache/cgb-dm/converted/pku
```

Full PKU and CGL S5 generated-metric results are recorded in
[TRAINING.md](models/cgb-dm/TRAINING.md).
That guide includes the full-training checkpoints and the standalone S5
evaluation commands for reloading package and reference checkpoints.
