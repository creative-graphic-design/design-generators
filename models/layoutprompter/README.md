# LayoutPrompter

LayoutPrompter is a Pydantic AI agent for few-shot layout generation. It builds prompts from layout exemplars, calls a provider-configured LLM, and returns normalized layout arrays through `NumpyLayoutGenerationOutput`.

The package is prompt-based and has no learned weights.

## Supported Checkpoints

LayoutPrompter does not publish learned checkpoints or Hub model repositories. `save_pretrained` stores prompt configuration so an agent can be reloaded with a caller-provided Pydantic AI model.

## Install

From the repository root:

```bash
uv sync --all-packages
```

Run package commands with `uv run --package layoutprompter ...`.

## Basic Usage

```python
import numpy as np
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
        "labels": np.asarray([0, 2]),
        "bboxes": np.asarray([[8, 10, 20, 15], [70, 80, 10, 12]]),
        "discrete_gold_bboxes": np.asarray([[8, 10, 20, 15], [70, 80, 10, 12]]),
    }
]
test_data = {
    "labels": np.asarray([0, 2]),
    "bboxes": np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]]),
    "discrete_gold_bboxes": np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]]),
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

## Datasets

Built-in label vocabularies are available for these dataset keys:

```text
publaynet
rico
posterlayout
webui
```

## Supported Tasks

`condition_type` accepts the public names and common vendor aliases:

```text
label, label_size, relation, completion, refinement, text
cat_cond, gen_t, size_cond, gen_ts, gen_r, partial, refine
```

The current package includes prompt serialization and parsing for `seq` and `html` layouts. Closed string modes such as dataset names, condition types, prompt formats, output type, and box format are normalized to enums at public boundaries while string inputs remain accepted.

The demo script uses a tiny synthetic WebUI-style example.

```bash
uv run --package layoutprompter python models/layoutprompter/scripts/demo.py
```

Without `OPENAI_API_KEY`, the demo exits with a skip message.

## Parity Results

The numpy-only implementation was checked against the original LayoutPrompter
vendor path and the previous torch-backed package output on the deterministic
WebUI fixture.

| Check | Cases | Result |
| --- | ---: | --- |
| Prompt byte equality | 1/1 prompt fixture | Exact byte match |
| Exemplar selection | 2/2 selected exemplar ids | Exact id match |
| Parser golden output | 2/2 arrays (`labels`, `bbox`) | Exact value match |
| Torch-to-numpy output equivalence | 1/1 public parser fixture | Exact JSON match for `bbox`, `labels`, `mask`, and `id2label` |

## Reproducibility

This section reproduces the parity verification against the original implementation.
Use these commands to regenerate deterministic reference data and run the
vendor parity checks. The generated JSON is written to `/tmp`.

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
Set CUDA_VISIBLE_DEVICES for the same command shape used by model packages; this script does not use CUDA.
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

### 3. Run Parity Checks

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layoutprompter \
  pytest models/layoutprompter/tests/vendor_parity -m vendor_parity
```

The test checks prompt bytes, selected exemplar ids, and parser output tensors.

See `Parity Results` for the measured agreement table.

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

Smoke output:

```text
webui
```

## Regular Checks

```bash
uv run --package layoutprompter pytest models/layoutprompter/tests -m "not vendor_parity and not integration"
uv run --package layoutprompter --with pytest-cov pytest models/layoutprompter/tests -m "not vendor_parity and not integration" --cov=layoutprompter --cov-report=term-missing
uv run --package layoutprompter ruff check models/layoutprompter
uv run --package layoutprompter ty check models/layoutprompter
```

The current package coverage under the command above is 99%.

## Original Implementation

The reference implementation is Microsoft LayoutGeneration, under `vendor/ms-layout-generation/LayoutPrompter` when the vendor source is available in this repository.

## License And Citation

Use this package together with the license terms of the original LayoutPrompter implementation and any dataset or LLM provider used at runtime. Cite the LayoutPrompter paper when publishing results based on this method.
