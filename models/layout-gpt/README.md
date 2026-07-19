---
language:
  - en
license: "other"
license_name: "review-needed"
license_link: "not recorded"
library_name: "pydantic-ai"
pipeline_tag: "text-to-image"
tags:
  - "layout-gpt"
  - "layout-generation"
---

# Model Card for LayoutGPT

![paper](https://img.shields.io/static/v1?label=paper&message=LayoutGPT&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=review-needed&color=yellow&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=NSR-1K&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=prompt--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package implements [LayoutGPT](https://arxiv.org/abs/2305.15393) as a [Pydantic AI](https://ai.pydantic.dev/) agent that reproduces the prompt, exemplar-selection, and parser behavior of the original text-to-layout method.

## Model Details

### Model Description

LayoutGPT is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `pydantic-ai`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** English prompts for prompt-only operation.
- **License:** review-needed.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** [LayoutGPT repository](https://github.com/weixi-feng/LayoutGPT)
- **Paper:** [arXiv 2305.15393](https://arxiv.org/abs/2305.15393)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| prompt configuration | n/a | no learned checkpoint; `save_pretrained` stores reloadable prompt configuration |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

```bash
uv sync --package layout-gpt
```

```python
from pydantic_ai.models.test import TestModel
from layout_gpt import LayoutGPTAgent

model = TestModel(custom_output_args={"elements": []})
agent = LayoutGPTAgent.from_pretrained(".cache/layout-gpt/prompt-config", model=model)
print(agent)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| NSR-1K | vendor/layout-gpt dataset metadata | not mirrored |

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

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

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

### 1. Prepare Optional Weight Cache

LayoutGPT itself has no checkpoint weights. This command prepares a cache
directory for optional provider or downstream assets without downloading
anything.

```bash
export LAYOUT_GPT_CACHE_DIR="${LAYOUT_GPT_CACHE_DIR:-.cache/layout-gpt}"
mkdir -p "${LAYOUT_GPT_CACHE_DIR}"
printf 'LayoutGPT has no package weights to download. Cache: %s\n' "${LAYOUT_GPT_CACHE_DIR}"
```

### 2. Regenerate Vendor Golden Data

The generated JSON path is controlled by `LAYOUT_GPT_CACHE_DIR`.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uv run --package layout-gpt python models/layout-gpt/scripts/generate_vendor_golden.py \
  --output "${LAYOUT_GPT_CACHE_DIR:-.cache/layout-gpt}/vendor-golden.json"
```

### 3. Run Parity Tests

```bash
uv run --package layout-gpt pytest models/layout-gpt/tests/vendor_parity -m vendor_parity
```

### 4. Convert Checkpoints

There is no checkpoint conversion step for LayoutGPT. The package converts
provider text responses into typed layout outputs at runtime.

```bash
uv run --package layout-gpt python - <<'PY'
print("LayoutGPT has no checkpoint conversion step.")
PY
```

### 5. Run Loading Smoke

There is no `from_pretrained` checkpoint for LayoutGPT. This smoke verifies the
installed package and output conversion path.

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

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream architecture needed for conversion and inference while exposing the common layout schema.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layout-gpt ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: review-needed.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
