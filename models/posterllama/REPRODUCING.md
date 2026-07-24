# Reproducing PosterLlama Checks

This page gives the mechanical local commands for PosterLlama prompt/parser
checks, original-code metadata capture, checkpoint conversion, and
`from_pretrained` smoke tests.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

Run the commands below from the repository root. Downloads and generated
artifacts are written under `.cache/posterllama`.

## Metadata Inspection

```bash
uv run --package posterllama python models/posterllama/scripts/download_original_assets.py
```

Add `--download` only when the raw checkpoint may be fetched into the local
cache:

```bash
uv run --package posterllama python models/posterllama/scripts/download_original_assets.py \
  --download \
  --cache-dir ./.cache/posterllama/original
```

## Unit Tests

```bash
uv run --package posterllama pytest models/posterllama/tests -m "not vendor_parity and not integration"
```

## Original Reference Metadata

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package posterllama --extra vendor \
  python models/posterllama/scripts/generate_reference_outputs.py \
    --vendor-root ./vendor/posterllama \
    --checkpoint-path ./.cache/posterllama/original/pytorch_model.bin \
    --base-llm-path ./models/codeLlama-7b-hf \
    --image-root ./.cache/posterllama/images \
    --jsonl ./.cache/posterllama/fixture.jsonl \
    --output-metadata ./.cache/posterllama/reference/metadata.json \
    --device cuda:0 \
    --seed 42
```

## Gated Parity

```bash
PARITY_REQUIRE=1 \
POSTERLLAMA_VENDOR_ROOT=./vendor/posterllama \
POSTERLLAMA_CHECKPOINT_PATH=./.cache/posterllama/original/pytorch_model.bin \
POSTERLLAMA_BASE_LLM_PATH=./models/codeLlama-7b-hf \
uv run --package posterllama pytest models/posterllama/tests/vendor_parity -m vendor_parity
```

## Conversion And Smoke

```bash
uv run --package posterllama python models/posterllama/scripts/convert_original_checkpoint.py \
  --checkpoint-path ./.cache/posterllama/original/pytorch_model.bin \
  --base-llm-path ./models/codeLlama-7b-hf \
  --output-dir ./.cache/posterllama/converted \
  --parser-smoke-text '<svg width="360" height="504"><rect data-category="text" x="10" y="20" width="100" height="40"/></svg>'
```

```bash
uv run --package posterllama python models/posterllama/scripts/smoke_from_pretrained.py \
  ./.cache/posterllama/converted
```
