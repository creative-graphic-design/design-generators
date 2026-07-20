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

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2305.15393&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2305.15393)
![venue](https://img.shields.io/static/v1?label=venue&message=NeurIPS+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&style=flat-square&logo=pydantic&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=NSR-1K&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package exposes [LayoutGPT](https://arxiv.org/abs/2305.15393) as a [`pydantic-ai`](https://ai.pydantic.dev/) agent for prompt-based text-to-layout generation, including the original prompt format, exemplar-selection modes, and output parser.

## Model Details

### Model Description

LayoutGPT is a prompt-based layout agent that turns natural-language scene descriptions and in-context examples into structured layout elements. It has no learned checkpoint; the package stores prompt, parser, and exemplar configuration while `pydantic-ai` supplies the model backend. Public outputs use normalized center `xywh` boxes in `[0, 1]`, request-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Weixi Feng et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** English prompts for prompt-only operation.
- **License:** MIT.

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

LayoutGPT builds the same in-context prompts as the original 2D script, calls any `pydantic-ai` compatible chat model, and parses CSS-style layout lines into typed Python objects and the shared layout output schema. The supported settings are `counting` and `spatial`; exemplar selection supports `fixed-random` and `k-similar`.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The packaged behavior follows the upstream prompt fixtures, exemplar metadata, parser rules, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing prompt configurations or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository and install the prompt-only workspace member. LayoutGPT has no learned checkpoints; the example uses saved prompt configuration and a parsed output object, so it runs without provider credentials.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-gpt
uv run --package layout-gpt python
```

```python
from layout_gpt import LayoutGPTAgent
from layout_gpt.schema import LayoutGPTConfig, LayoutGPTOutput, LayoutItem2D

agent = LayoutGPTAgent(config=LayoutGPTConfig(setting="counting", icl_type="fixed-random", k=0))
agent.save_pretrained(".cache/layout-gpt/prompt-config")
loaded = LayoutGPTAgent.from_pretrained(".cache/layout-gpt/prompt-config")

output = LayoutGPTOutput(
    prompt="there is one clock in the image",
    canvas_size=64,
    items=[LayoutItem2D(label="clock", left=0.25, top=0.25, width=0.5, height=0.5)],
    raw_text="clock {height: 32px; width: 32px; top: 16px; left: 16px; }",
    id2label={0: "clock"},
)
layout_output = output.to_layout_generation_output()
print(loaded.config.setting)
print(layout_output.bbox)
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

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated prompt fixtures and parser-reference JSON. Provider outputs, generated artifacts, and downloaded datasets are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or prompt fixture where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality, exact token or byte equality, or explicitly stated numeric tolerance against the vendor path.

### Parity Results

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

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-gpt/REPRODUCING.md) for the commands that prepare prompt assets, generate reference outputs, run parity checks, save prompt configuration, and smoke-test local loading.


## Environmental Impact

No new model training is performed by this prompt-only package. Parity costs depend on the selected prompt fixtures, provider configuration, and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutGPT builds a structured prompt from task examples, calls a configured `pydantic-ai` model, and parses CSS-like layout text into boxes and labels. For `k-similar`, pass a query embedding function and candidate embeddings to `run_sync()` or `build_prompt()`; CLIP remains optional.

### Compute Infrastructure

Vendor parity commands are deterministic CPU checks for prompt serialization, exemplar selection, and parser behavior.

#### Hardware

CPU is sufficient for import, smoke tests, and recorded vendor parity checks.

#### Software

Use `uv run --package layout-gpt ...` from the repository root so workspace dependency sources and extras resolve correctly.

The demo script accepts a configured `pydantic-ai` model and a vendor NSR-1K JSON file.

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
