# Reproducing Parse-Then-Place Parity

This guide reproduces the original-implementation agreement checks for the Parse-Then-Place package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

Run commands from the repository root. Set `CUDA_VISIBLE_DEVICES=<gpu-index>` to the GPU you want to use for vendor reference generation and parity.

1. Download the original assets. This writes `.cache/parse-then-place/assets`, including the expected `ckpt/rico/stage1/pytorch_model.bin` parser state and stage-2 placement checkpoints.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/download_original_assets.py \
  --output-dir .cache/parse-then-place/assets
```

2. Generate fixed-seed vendor reference outputs for the RICO finetune placement model.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package parse-then-place python models/parse-then-place/scripts/generate_reference_outputs.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --seed 42 \
  --num-examples 1 \
  --output-dir .cache/parse-then-place/reference/rico-finetune
```

3. Run the marked vendor parity tests against the generated reference directory. The current parity suite covers the stage-2 placement model; stage-1 parser adapter flattening is checked by conversion and smoke loading.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARSE_THEN_PLACE_ORIGINAL_ROOT=.cache/parse-then-place/assets \
  PARSE_THEN_PLACE_REFERENCE_DIR=.cache/parse-then-place/reference/rico-finetune \
  uv run --package parse-then-place pytest models/parse-then-place/tests/vendor_parity -m vendor_parity
```

4. Convert the RICO finetune checkpoint into local [`🤗transformers`](https://huggingface.co/docs/transformers/index) pipeline format.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/convert_original_checkpoint.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --parser-state-path .cache/parse-then-place/assets/ckpt/rico/stage1/pytorch_model.bin \
  --output-dir .cache/parse-then-place/converted/rico-finetune
```

5. Smoke-test local `from_pretrained` loading.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/smoke_from_pretrained.py \
  --path .cache/parse-then-place/converted/rico-finetune
```
