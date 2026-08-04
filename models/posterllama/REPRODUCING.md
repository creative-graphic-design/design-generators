# Reproducing PosterLlama Checks

This page gives the mechanical local commands for PosterLlama prompt/parser
checks, original-code prompt/parser reference capture, checkpoint conversion, and
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

## Original Prompt And Parser References

```bash
uv run --package posterllama \
  python models/posterllama/scripts/generate_prompt_parser_references.py \
    --source-root ./vendor/posterllama \
    --output-json ./.cache/posterllama/reference/prompt_parser.json
```

## Gated Parity

```bash
PARITY_REQUIRE=1 \
POSTERLLAMA_VENDOR_ROOT=./vendor/posterllama \
uv run --package posterllama pytest models/posterllama/tests/vendor_parity -m vendor_parity
```

This prompt/parser parity path compares five original-source prompt templates
and one `html_to_ui.get_bbox()` parser result. Run full generation parity by
invoking `vendor/posterllama/generate.py` with the raw checkpoint, base LLM
path, image root, JSONL input, fixed seed, and one selected GPU.

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
