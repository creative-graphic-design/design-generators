# Reproducing Parse-Then-Place Parity

This guide reproduces the original-implementation agreement checks for the Parse-Then-Place package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Reproduce original-implementation agreement by downloading vendor assets,
generating fixed-seed references on one GPU, running marked parity tests, then
converting and smoke-loading the pipeline checkpoint.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/download_original_assets.py \
  --output-dir .cache/parse-then-place/assets

CUDA_VISIBLE_DEVICES=4 uv run --package parse-then-place python models/parse-then-place/scripts/generate_reference_outputs.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --seed 42 \
  --num-examples 1 \
  --output-dir .cache/parse-then-place/reference/rico-finetune

CUDA_VISIBLE_DEVICES=4 PARSE_THEN_PLACE_ORIGINAL_ROOT=.cache/parse-then-place/assets \
  PARSE_THEN_PLACE_REFERENCE_DIR=.cache/parse-then-place/reference/rico-finetune \
  uv run --package parse-then-place pytest models/parse-then-place/tests/vendor_parity -m vendor_parity

uv run --package parse-then-place python models/parse-then-place/scripts/convert_original_checkpoint.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --parser-state-path .cache/parse-then-place/assets/ckpt/rico/stage1/pytorch_model.bin \
  --output-dir .cache/parse-then-place/converted/rico-finetune

uv run --package parse-then-place python models/parse-then-place/scripts/smoke_from_pretrained.py \
  --path .cache/parse-then-place/converted/rico-finetune
```
