---
language:
  - en
license: "mit"
library_name: "pydantic-ai"
pipeline_tag: "other"
tags:
  - "layout-gpt"
  - "layout-generation"
  - "neurips-2023"
datasets:
  - "NSR-1K"
---

# Model Card for LayoutGPT

![paper](https://img.shields.io/static/v1?label=paper&message=LayoutGPT&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=NeurIPS%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=NSR-1K&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package provides [LayoutGPT](https://arxiv.org/abs/2305.15393) as a [Pydantic AI](https://ai.pydantic.dev/) agent that reproduces the prompt, exemplar-selection, and parser behavior of the original text-to-layout method.

## Model Details

### Model Description

LayoutGPT is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `pydantic-ai`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** English prompts for prompt-only operation.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** [LayoutGPT repository](https://github.com/UCSB-AI/LayoutGPT). The `vendor/layout-gpt` submodule tracks this repository.
- **Paper:** [arXiv 2305.15393](https://arxiv.org/abs/2305.15393)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| prompt configuration | n/a | no learned checkpoint; `save_pretrained` stores reloadable prompt configuration |

## Uses

### Direct Use

Use this package for research inference, prompt-serialization checks, and vendor-parity validation of generated layouts.

LayoutGPT builds the same in-context prompts as the original 2D script, calls any Pydantic AI compatible chat model, and parses CSS-style layout lines into typed Python objects and the shared layout output schema. The supported settings are `counting` and `spatial`; exemplar selection supports `fixed-random` and `k-similar`.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The packaged behavior follows the upstream prompt fixtures, exemplar metadata, parser rules, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing prompt configurations or comparing new results against the original implementation.

## How to Get Started with the Model

```bash
uv sync --package layout-gpt
```

```python
from pydantic_ai.models.test import TestModel
from layout_gpt import LayoutGPTAgent
from layout_gpt.exemplars import load_nsr_examples
from layout_gpt.schema import LayoutGPTConfig

model = TestModel(custom_output_args={"elements": []})
examples = load_nsr_examples(
    "vendor/layout-gpt/dataset/NSR-1K/counting/counting.train.json",
    setting="counting",
)
agent = LayoutGPTAgent(
    model=model,
    config=LayoutGPTConfig(
        setting="counting",
        icl_type="fixed-random",
        k=1,
        canvas_size=256,
    ),
    exemplars=examples,
)
output = agent.run_sync("there is one clock in the image")
layout_output = output.to_layout_generation_output()
print(layout_output.bbox)
print(layout_output.labels)
print(layout_output.id2label)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| NSR-1K | vendor/layout-gpt dataset metadata | not mirrored |

NSR-1K examples are loaded from the vendor dataset JSON files. `layout_output.bbox` is normalized center `xywh` in `[0, 1]`; `layout_output.labels` are request-local integer ids, and `layout_output.id2label` maps those ids back to object names.

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Inputs and outputs are normalized to the public layout schema at package boundaries. Vendor-specific boxes, tokens, prompts, or analog bits stay inside package adapters and parity fixtures.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training time and carbon measurements are not recorded in the current README.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated prompt fixtures and parser-reference JSON. Provider outputs, generated artifacts, and downloaded datasets are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or prompt fixture where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality, exact token or byte equality, or explicitly stated numeric tolerance against the vendor path.

### Results

The numeric agreement record is the `## Parity Results` table below. Rows marked as not recorded are documentation gaps rather than inferred measurements.

## Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| Fixed-random exemplar ids | 3 ids | exact list equality against vendor golden seed 42 | pass |
| Fixed-random chat prompt bytes | 1 prompt | exact UTF-8 byte equality | pass |
| Fixed-random completion prompt bytes | 1 prompt | exact UTF-8 byte equality | pass |
| K-similar exemplar ids | 3 ids | exact list equality against vendor golden seed 42 | pass |
| K-similar completion prompt bytes | 1 prompt | exact UTF-8 byte equality | pass |
| 2D parser output | golden parser lines | exact normalized vendor parse equality | pass |
| 3D parser output | golden parser lines | exact normalized vendor parse equality | pass |

## Reproducibility

This section reproduces the parity verification against the original implementation.
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
print("LayoutGPT stores prompt configuration rather than learned weights.")
PY
```

### 5. Run Loading Smoke

This smoke verifies the installed package and output conversion path.

```bash
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
```

## Model Examination

Interpretability and failure-analysis artifacts are not recorded in the current README.

## Environmental Impact

No new model training is performed by this prompt-only package. Parity costs depend on the selected prompt fixtures, provider configuration, and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream prompt construction, exemplar selection, and CSS-like output parser needed for inference while exposing the common layout schema. For `k-similar`, pass a query embedding function and candidate embeddings to `run_sync()` or `build_prompt()`; the package keeps CLIP optional.

### Compute Infrastructure

Vendor parity commands are deterministic CPU checks for prompt serialization, exemplar selection, and parser behavior.

#### Hardware

CPU is sufficient for import, smoke tests, and recorded vendor parity checks.

#### Software

Use `uv run --package layout-gpt ...` from the repository root so workspace dependency sources and extras resolve correctly.

The demo script accepts a configured Pydantic AI model:

```bash
uv run --package layout-gpt python models/layout-gpt/scripts/demo_layout_gpt.py \
  --train-json vendor/layout-gpt/dataset/NSR-1K/counting/counting.train.json \
  --prompt "there is one clock in the image" \
  --setting counting \
  --canvas-size 256 \
  --model "${LAYOUT_GPT_MODEL:-openai:gpt-5-mini}"
```

Regular package checks are:

```bash
uv run --package layout-gpt pytest models/layout-gpt/tests
uv run --package layout-gpt ruff check models/layout-gpt
uv run --package layout-gpt ruff format --check models/layout-gpt
uv run --package layout-gpt ty check models/layout-gpt
```

## License

Repository wrapper code is Apache-2.0. The original LayoutGPT repository is MIT licensed; keep upstream prompt assets, datasets, and provider outputs under their own terms.

## Citation

```bibtex
@article{feng2023layoutgpt,
  title = {LayoutGPT: Compositional Visual Planning and Generation with Large Language Models},
  author = {Feng, Weixi and Zhu, Wanrong and Fu, Tsu-Jui and Jampani, Varun and Akula, Arjun and He, Xuehai and Basu, Sugato and Wang, Xin Eric and Wang, William Yang},
  journal = {arXiv preprint arXiv:2305.15393},
  year = {2023}
}
```
