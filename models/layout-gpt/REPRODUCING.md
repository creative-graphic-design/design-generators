# Reproducing LayoutGPT Parity

This guide reproduces the original-implementation agreement checks for the LayoutGPT package.

Workflow order: prepare prompt assets, generate references, run parity checks, save prompt configuration, then smoke-test local loading.

These steps reproduce the deterministic parity checks. The generated JSON is an inspectable artifact written to `.cache/layout-gpt/vendor-golden.json`; the parity test also regenerates its golden data in-process so the test cannot accidentally pass by reading a stale file. The scripts look for `vendor/layout-gpt` inside this repository; pass `--vendor-root` if your checkout is elsewhere.

### 1. Prepare Prompt Assets

LayoutGPT itself has no learned weights to download. Initialize the original prompt and NSR-1K data source before generating references.

```bash
git submodule update --init vendor/layout-gpt
```

### 2. Regenerate Vendor Golden Data

This writes an inspectable reference artifact.

```bash
uv run --package layout-gpt python models/layout-gpt/scripts/generate_vendor_golden.py \
  --output .cache/layout-gpt/vendor-golden.json
```

### 3. Run Parity Tests

```bash
uv run --package layout-gpt pytest models/layout-gpt/tests/vendor_parity -m vendor_parity
```

### 4. Persist Prompt Configuration

There is no learned-weight conversion step for LayoutGPT. `save_pretrained`
stores the prompt and parser configuration that the agent reloads at runtime.

```bash
uv run --package layout-gpt python models/layout-gpt/scripts/save_prompt_config.py \
  --path .cache/layout-gpt/prompt-config
```

### 5. Run Loading Smoke

This smoke verifies `save_pretrained` to `from_pretrained` reload and output conversion.

```bash
uv run --package layout-gpt python models/layout-gpt/scripts/smoke_from_pretrained.py \
  --path .cache/layout-gpt/prompt-config
```
