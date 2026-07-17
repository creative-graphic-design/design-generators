# LayoutPrompter

LayoutPrompter is a Pydantic AI agent for few-shot layout generation. It builds prompts from layout exemplars, calls a provider-configured LLM, and returns normalized layout tensors through `LayoutGenerationOutput`.

The package is prompt-based and has no learned weights.

## Install

From the repository root:

```bash
uv sync --all-packages
```

Run package commands with `uv run --package layoutprompter ...`.

## Basic Usage

```python
import torch
from pydantic_ai.models.test import TestModel

from layoutprompter import LayoutPrompter, LayoutPrompterConfig

model = TestModel(
    custom_output_args={
        "elements": [
            {
                "label": "text",
                "bbox": {"left": 12, "top": 16, "width": 24, "height": 32},
            }
        ]
    }
)

train_data = [
    {
        "labels": torch.tensor([0, 2]),
        "bboxes": torch.tensor([[8, 10, 20, 15], [70, 80, 10, 12]]),
        "discrete_gold_bboxes": torch.tensor([[8, 10, 20, 15], [70, 80, 10, 12]]),
    }
]
test_data = {
    "labels": torch.tensor([0, 2]),
    "bboxes": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
    "discrete_gold_bboxes": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
}

agent = LayoutPrompter(
    LayoutPrompterConfig(dataset="webui", model=model, shuffle=False, num_prompt=1)
)
output = agent.run_sync(train_data, test_data)
print(output.labels)
print(output.bbox)
```

`output.bbox` is normalized center `xywh` with shape `(batch, elements, 4)`.

## Model Settings

Pass any Pydantic AI model object through `LayoutPrompterConfig.model`.

```python
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from layoutprompter import LayoutPrompterConfig

provider = OpenAIProvider(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
config = LayoutPrompterConfig(
    dataset="webui",
    model=OpenAIChatModel("gpt-4o-mini", provider=provider),
    shuffle=False,
    num_prompt=1,
)
```

If `model` is omitted, the agent checks `LAYOUTPROMPTER_MODEL`, then `PYDANTIC_AI_MODEL`, and finally falls back to `openai:gpt-4o-mini`.

## Supported Tasks

`condition_type` accepts the public names and common vendor aliases:

```text
label, label_size, relation, completion, refinement, text
cat_cond, gen_t, size_cond, gen_ts, gen_r, partial, refine
```

The current package includes prompt serialization and parsing for `seq` and `html` layouts. The demo script uses a tiny synthetic WebUI-style example.

```bash
uv run --package layoutprompter python models/layoutprompter/scripts/demo.py
```

Without `OPENAI_API_KEY`, the demo exits with a skip message.

## Reproducing Vendor Parity

The deterministic vendor parity test compares the port against golden data regenerated from the read-only LayoutPrompter vendor code. The generated JSON is written to a temporary path and is not committed.

### 1. Weights

LayoutPrompter has no weights to download.

```bash
printf 'LayoutPrompter has no learned weights.\n'
```

### 2. Generate Reference Golden

Prerequisites:

```text
The vendor source must exist at one of:
- vendor/ms-layout-generation/LayoutPrompter/src
- ../design-generators/vendor/ms-layout-generation/LayoutPrompter/src

Set LAYOUTPROMPTER_VENDOR_SRC to another src directory if needed.
CUDA_VISIBLE_DEVICES is accepted for consistency with model packages, but this script does not use CUDA.
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layoutprompter \
  python models/layoutprompter/scripts/generate_vendor_golden.py \
  --output /tmp/layoutprompter_vendor_golden.json
```

Generated file:

```text
/tmp/layoutprompter_vendor_golden.json
```

### 3. Run Vendor Parity

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layoutprompter \
  pytest models/layoutprompter/tests/vendor_parity -m vendor_parity
```

The test checks prompt bytes, selected exemplar ids, and parser output tensors.

### 4. Checkpoint Conversion

There is no checkpoint conversion step for LayoutPrompter.

```bash
printf 'No checkpoint conversion is required for LayoutPrompter.\n'
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

Expected output:

```text
webui
```

## Regular Checks

```bash
uv run --package layoutprompter pytest models/layoutprompter/tests -m "not vendor_parity and not integration"
uv run --package layoutprompter ruff check models/layoutprompter
uv run --package layoutprompter ty check models/layoutprompter
```
