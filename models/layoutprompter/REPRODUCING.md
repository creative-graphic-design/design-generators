# Reproducing LayoutPrompter Parity

This guide reproduces the original-implementation agreement checks for the LayoutPrompter package.

Workflow order: prepare prompt assets, generate references, run parity checks, save prompt configuration, then smoke-test local loading.

Use these commands to regenerate deterministic reference data and run the vendor parity checks. The generated JSON is an inspectable artifact written under `.cache/layoutprompter/`; the parity test also regenerates its golden data in-process so the test cannot accidentally pass by reading a stale file.

### 1. Prepare Prompt Assets

LayoutPrompter has no learned weights to download. Initialize the original prompt code before generating references.

```bash
git submodule update --init vendor/ms-layout-generation
```

### 2. Generate Reference Golden

```bash
uv run --package layoutprompter python models/layoutprompter/scripts/generate_vendor_golden.py \
  --output .cache/layoutprompter/vendor-golden.json
```

Generated file:

```text
.cache/layoutprompter/vendor-golden.json
```

### 3. Run Parity Checks

```bash
uv run --package layoutprompter pytest models/layoutprompter/tests/vendor_parity -m vendor_parity
```

The test checks prompt bytes, selected exemplar ids, and parser output tensors.

See `Parity Results` for the measured agreement table.

### 4. Persist Prompt Configuration

There is no learned-weight conversion step for LayoutPrompter. The smoke command below calls `save_pretrained` before reloading the prompt configuration.

### 5. from_pretrained Smoke

`save_pretrained` stores prompt configuration, not weights.

```bash
uv run --package layoutprompter python models/layoutprompter/scripts/smoke_from_pretrained.py \
  --path .cache/layoutprompter/prompt-config
```

Smoke output:

```text
webui
```
