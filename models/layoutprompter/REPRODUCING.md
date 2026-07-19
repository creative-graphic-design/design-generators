# Reproducing LayoutPrompter Parity

This guide reproduces the original-implementation agreement checks for the LayoutPrompter package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Use these commands to regenerate deterministic reference data and run the
vendor parity checks. The generated JSON is written under `.cache/layoutprompter/`.

### 1. Weights

LayoutPrompter has no weights to download.

```bash
printf 'LayoutPrompter has no learned weights.\n'
```

### 2. Generate Reference Golden

Prerequisites:

```bash
git submodule update --init vendor/ms-layout-generation
```

```text
The script resolves vendor/ms-layout-generation through laygen.common.vendor_root()
and then imports LayoutPrompter/src from that checkout.

Set LAYOUTPROMPTER_VENDOR_SRC to another LayoutPrompter/src directory if needed.
```

Command:

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

There is no learned-weight conversion step for LayoutPrompter. `save_pretrained`
stores prompt configuration for reload.

```bash
printf 'LayoutPrompter stores prompt configuration rather than learned weights.\n'
```

### 5. from_pretrained Smoke

`save_pretrained` stores prompt configuration, not weights.

```bash
uv run --package layoutprompter python - <<'PY'
from tempfile import TemporaryDirectory

from pydantic_ai.models.test import TestModel

from layoutprompter import LayoutPrompter, LayoutPrompterConfig

model = TestModel(custom_output_args={"elements": []})
agent = LayoutPrompter(LayoutPrompterConfig(model=model, dataset="webui"))

with TemporaryDirectory() as tmpdir:
    agent.save_pretrained(tmpdir)
    loaded = LayoutPrompter.from_pretrained(tmpdir, model=model)
    print(loaded.config.dataset)
PY
```

Smoke output:

```text
webui
```
