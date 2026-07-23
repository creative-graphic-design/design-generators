# Reproducing PosterO Checks

Workflow order: download step, generate reference metadata, run parity checks, save prompt configuration, then smoke-test local loading.

## Setup

```bash
uv sync --package postero
```

## Download

No external download is required for the first prompt-config checks. Real provider outputs and large assets should stay outside git.

## Reference Metadata

Regenerate the prompt golden metadata from deterministic in-memory records.

```bash
uv run --package postero python models/postero/scripts/generate_vendor_golden.py
```

## Parity

```bash
PARITY_REQUIRE=1 uv run --package postero pytest models/postero/tests/vendor_parity -m vendor_parity
```

## Prompt Configuration

```bash
uv run --package postero python models/postero/scripts/save_prompt_config.py --output-dir .cache/postero/prompt-config
```

## Smoke

```bash
uv run --package postero python models/postero/scripts/smoke_from_pretrained.py --work-dir .cache/postero/smoke
```
