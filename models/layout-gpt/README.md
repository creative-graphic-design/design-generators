# LayoutGPT

LayoutGPT is a Pydantic AI agent for text-to-layout planning. It builds the
same in-context prompts as the original LayoutGPT 2D script, calls any
Pydantic AI compatible chat model, and parses CSS-style layout lines into typed
Python objects and the shared layout output schema.

The package does not ship model weights. Generation uses the model provider you
pass to Pydantic AI, such as an OpenAI-compatible endpoint.

## Install

Run commands from this package directory unless a command says otherwise.

```bash
(cd models/layout-gpt && uv sync --package layout-gpt)
```

## Basic Usage

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from layout_gpt import LayoutGPTAgent
from layout_gpt.exemplars import load_nsr_examples
from layout_gpt.schema import LayoutGPTConfig

model = OpenAIChatModel(
    "gpt-5-mini",
    provider=OpenAIProvider(
        base_url="https://example-openai-compatible-endpoint/v1",
        api_key="...",
    ),
)

examples = load_nsr_examples(
    "../../vendor/layout-gpt/dataset/NSR-1K/counting/counting.train.json",
    setting="counting",
)

agent = LayoutGPTAgent(
    model=model,
    config=LayoutGPTConfig(
        setting="counting",
        icl_type="fixed-random",
        k=8,
        canvas_size=256,
    ),
)

output = agent.run_sync(
    "there is one clock in the image",
    train_examples=examples,
)
layout_output = output.to_layout_generation_output()
```

`layout_output.bbox` is normalized center `xywh` in `[0, 1]`.
`layout_output.labels` are request-local integer ids, and
`layout_output.id2label` maps those ids back to object names.

## Model Settings

You can pass a Pydantic AI model object directly:

```python
from layout_gpt import LayoutGPTAgent

agent = LayoutGPTAgent(model=model)
```

Or set a provider name supported by Pydantic AI:

```bash
export LAYOUT_GPT_MODEL="openai:gpt-5-mini"
```

Then construct the agent without a model argument:

```python
from layout_gpt import LayoutGPTAgent

agent = LayoutGPTAgent()
```

The core generation settings are:

- `setting`: `counting` or `spatial`.
- `icl_type`: `fixed-random` or `k-similar`.
- `k`: number of exemplars to include before token-budget truncation.
- `canvas_size`: square canvas size in pixels for prompt serialization and parsing.
- `gpt_input_length_limit`: token budget used while adding exemplars.
- `temperature`, `top_p`: forwarded to the Pydantic AI model when supported.

For `k-similar`, pass a query embedding function and candidate embeddings to
`run_sync()` or `build_prompt()`. The package keeps CLIP optional.

## Demo

```bash
(
  cd models/layout-gpt
  uv run --package layout-gpt python scripts/demo_layout_gpt.py \
    --train-json ../../vendor/layout-gpt/dataset/NSR-1K/counting/counting.train.json \
    --prompt "there is one clock in the image" \
    --setting counting \
    --canvas-size 256 \
    --model "${LAYOUT_GPT_MODEL:-openai:gpt-5-mini}"
)
```

## Reproducing Vendor Parity

These steps reproduce the deterministic parity checks without committing golden
payloads. The scripts look for `vendor/layout-gpt` next to this worktree; pass
`--vendor-root` if your checkout is elsewhere.

### 1. Prepare Optional Weight Cache

LayoutGPT itself has no checkpoint weights. This command prepares a cache
directory for optional provider or downstream assets without downloading
anything.

```bash
(
  cd models/layout-gpt
  export LAYOUT_GPT_CACHE_DIR="${LAYOUT_GPT_CACHE_DIR:-/tmp/layout-gpt-cache}"
  mkdir -p "${LAYOUT_GPT_CACHE_DIR}"
  printf 'LayoutGPT has no package weights to download. Cache: %s\n' "${LAYOUT_GPT_CACHE_DIR}"
)
```

### 2. Regenerate Vendor Golden Data

The generated JSON is a temporary artifact. Do not commit it.

```bash
(
  cd models/layout-gpt
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  uv run --package layout-gpt python scripts/generate_vendor_golden.py \
    --output "${LAYOUT_GPT_CACHE_DIR:-/tmp/layout-gpt-cache}/vendor-golden.json"
)
```

### 3. Run Vendor Parity Tests

```bash
(
  cd models/layout-gpt
  uv run --package layout-gpt pytest tests/vendor_parity -m vendor_parity
)
```

### 4. Convert Checkpoints

There is no checkpoint conversion step for LayoutGPT. The package converts
provider text responses into typed layout outputs at runtime.

```bash
(
  cd models/layout-gpt
  uv run --package layout-gpt python - <<'PY'
print("LayoutGPT has no checkpoint conversion step.")
PY
)
```

### 5. Run Loading Smoke

There is no `from_pretrained` checkpoint for LayoutGPT. This smoke verifies the
installed package and output conversion path.

```bash
(
  cd models/layout-gpt
  uv run --package layout-gpt python - <<'PY'
from layout_gpt.schema import LayoutGPTOutput, LayoutItem2D

output = LayoutGPTOutput(
    prompt="there is one clock in the image",
    canvas_size=64,
    items=[LayoutItem2D(label="clock", left=0.25, top=0.25, width=0.5, height=0.5)],
    raw_text="clock {height: 32px; width: 32px; top: 16px; left: 16px; }",
    id2label={0: "clock"},
)
layout = output.to_layout_generation_output()
assert layout.bbox.shape == (1, 1, 4)
assert layout.id2label == {0: "clock"}
print("LayoutGPT loading smoke passed.")
PY
)
```

## Local Checks

```bash
(
  cd models/layout-gpt
  uv run --package layout-gpt pytest
  uv run --package layout-gpt ruff check .
  uv run --package layout-gpt ruff format --check .
  uv run --package layout-gpt ty check .
)
```
