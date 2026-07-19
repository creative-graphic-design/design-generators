# Reproducing LayoutGPT Parity

This guide reproduces the original-implementation agreement checks for the LayoutGPT package.

Workflow order: prepare prompt assets, generate references, run parity checks, save prompt configuration, then smoke-test local loading.

These steps reproduce the deterministic parity checks. The generated JSON is
written to `.cache/layout-gpt/vendor-golden.json`. The scripts look for
`vendor/layout-gpt` next to this worktree; pass `--vendor-root` if your checkout
is elsewhere.

### 1. Prepare Prompt Configuration Cache

LayoutGPT itself has no learned weights. This command prepares a cache
directory for prompt configuration and generated vendor-reference JSON without
downloading anything.

```bash
export LAYOUT_GPT_CACHE_DIR="${LAYOUT_GPT_CACHE_DIR:-.cache/layout-gpt}"
mkdir -p "${LAYOUT_GPT_CACHE_DIR}"
printf 'LayoutGPT has no learned weights to download. Cache: %s\n' "${LAYOUT_GPT_CACHE_DIR}"
```

### 2. Regenerate Vendor Golden Data

The generated JSON path is controlled by `LAYOUT_GPT_CACHE_DIR`.

```bash
uv run --package layout-gpt python models/layout-gpt/scripts/generate_vendor_golden.py \
  --output "${LAYOUT_GPT_CACHE_DIR:-.cache/layout-gpt}/vendor-golden.json"
```

### 3. Run Parity Tests

```bash
uv run --package layout-gpt pytest models/layout-gpt/tests/vendor_parity -m vendor_parity
```

### 4. Persist Prompt Configuration

There is no learned-weight conversion step for LayoutGPT. `save_pretrained`
stores the prompt and parser configuration that the agent reloads at runtime.

```bash
uv run --package layout-gpt python - <<'PY'
from layout_gpt import LayoutGPTAgent
from layout_gpt.schema import LayoutGPTConfig

agent = LayoutGPTAgent(config=LayoutGPTConfig(setting="counting", icl_type="fixed-random", k=1))
agent.save_pretrained(".cache/layout-gpt/prompt-config")
print(".cache/layout-gpt/prompt-config/layout_gpt_config.json")
PY
```

### 5. Run Loading Smoke

This smoke verifies `save_pretrained` to `from_pretrained` reload and output conversion.

```bash
uv run --package layout-gpt python - <<'PY'
from layout_gpt import LayoutGPTAgent
from layout_gpt.schema import LayoutGPTOutput, LayoutItem2D

loaded = LayoutGPTAgent.from_pretrained(".cache/layout-gpt/prompt-config")
assert loaded.config.setting == "counting"
assert loaded.config.icl_type == "fixed-random"
assert loaded.config.k == 1

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
```
