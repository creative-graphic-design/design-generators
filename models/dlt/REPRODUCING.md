# Reproducing DLT Checks

This walkthrough reproduces DLT agreement checks against the original implementation once local trained checkpoints are available.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

## Commands

```bash
mkdir -p .cache/dlt/reference .cache/dlt/converted
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package dlt --extra vendor \
  python models/dlt/scripts/generate_vendor_reference.py \
  --config vendor/dlt/dlt/configs/remote/dlt_publaynet_config.py \
  --workdir dlt-publaynet \
  --epoch 799 \
  --condition all \
  --output-metadata .cache/dlt/reference/publaynet-all.json
```

```bash
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=0 uv run --package dlt --extra vendor \
  pytest models/dlt/tests/vendor_parity -m vendor_parity
```

```bash
uv run --package dlt \
  python models/dlt/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --checkpoint-dir .cache/dlt/original/checkpoint-799 \
  --output-dir .cache/dlt/converted/publaynet
```

```bash
uv run --package dlt \
  python models/dlt/scripts/smoke_from_pretrained.py \
  --path .cache/dlt/converted/publaynet
```
