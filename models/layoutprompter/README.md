---
language:
  - en
license: "mit"
library_name: "pydantic-ai"
pipeline_tag: "other"
tags:
  - "layoutprompter"
  - "layout-generation"
  - "neurips-2023"
datasets:
  - "creative-graphic-design/PubLayNet"
  - "creative-graphic-design/Rico"
  - "PosterLayout"
---

# Model Card for LayoutPrompter

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2311.06495&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2311.06495)
![venue](https://img.shields.io/static/v1?label=venue&message=NeurIPS+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&style=flat-square&logo=pydantic&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PosterLayout&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package exposes [LayoutPrompter](https://arxiv.org/abs/2311.06495) as a [`pydantic-ai`](https://ai.pydantic.dev/) agent for few-shot layout generation from prompt exemplars and structured parser output.

## Model Details

### Model Description

LayoutPrompter is a prompt-based layout agent that selects in-context exemplars and asks a language model to complete structured layout coordinates. It has no learned checkpoint; `save_pretrained` stores prompt configuration and parser settings, while `pydantic-ai` supplies the runtime model backend. Public outputs use normalized center `xywh` boxes in `[0, 1]`, request-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Jiawei Lin et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** English prompts for prompt-only operation.
- **License:** MIT.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration LayoutPrompter](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter)
- **Paper:** [arXiv 2311.06495](https://arxiv.org/abs/2311.06495)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| prompt configuration | n/a | no learned checkpoint; `save_pretrained` stores reloadable prompt configuration |

## Uses

### Direct Use

Use this package for research inference, prompt-serialization checks, and vendor-parity validation of generated layouts.

Pass any `pydantic-ai` model object through `LayoutPrompterConfig.model`. If `model` is omitted, the agent checks `LAYOUTPROMPTER_MODEL`, then `PYDANTIC_AI_MODEL`, and finally falls back to `openai:gpt-4o-mini`.

Real provider calls require the credentials used by the selected `pydantic-ai` provider. For the default OpenAI provider, set `OPENAI_API_KEY`; for other providers, set the provider-specific API key before running `run_sync()` or the demo script. The `TestModel` example below is an offline stub that returns the canned element supplied in `custom_output_args`; it does not call an LLM.

`condition_type` accepts the public names and common vendor aliases: `label`, `label_size`, `relation`, `completion`, `refinement`, `text`, `cat_cond`, `gen_t`, `size_cond`, `gen_ts`, `gen_r`, `partial`, and `refine`. The package includes prompt serialization and parsing for `seq` and `html` layouts. String inputs are accepted for dataset names, condition types, prompt formats, output type, and box format.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The packaged behavior follows the upstream prompt fixtures, parser rules, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing prompt configurations or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository and install the prompt-only workspace member. LayoutPrompter has no learned checkpoints; the example uses `pydantic-ai`'s `TestModel` so it runs without provider credentials.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layoutprompter
uv run --package layoutprompter python
```

```python
import numpy as np
from pydantic_ai.models.test import TestModel

from layoutprompter import LayoutPrompter, LayoutPrompterConfig

model = TestModel(custom_output_args={"elements": [{"label": "text", "bbox": {"left": 12, "top": 16, "width": 24, "height": 32}}]})
agent = LayoutPrompter(LayoutPrompterConfig(dataset="webui", model=model, shuffle=False, num_prompt=1))
agent.save_pretrained(".cache/layoutprompter/prompt-config")

train_data = [{"labels": np.asarray([0]), "bboxes": np.asarray([[8, 10, 20, 15]]), "discrete_gold_bboxes": np.asarray([[8, 10, 20, 15]])}]
test_data = {"labels": np.asarray([0]), "bboxes": np.asarray([[0, 0, 1, 1]]), "discrete_gold_bboxes": np.asarray([[0, 0, 1, 1]])}
output = agent.run_sync(train_data, test_data)
print(output.bbox.shape)
print(output.labels.tolist())
print(output.mask.tolist())
print(output.id2label[0])
```

```text
(1, 1, 4)
[[0]]
[[True]]
text
```

LayoutPrompter records are dict-like examples with `labels`, `bboxes`, and `discrete_gold_bboxes`. Labels are dataset-local integer ids; `bboxes` and `discrete_gold_bboxes` are left-top-width-height boxes on the dataset's discrete canvas grid: PubLayNet uses `120x160`, RICO25 uses `90x160`, PosterLayout uses `102x150`, and WebUI uses `120x120`.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PosterLayout | unknown | built-in label vocabulary |
| WebUI | unknown | deterministic local fixture |

Built-in label vocabularies are available for `publaynet`, `rico`, `posterlayout`, and `webui`.

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
| Prompt byte equality | 1 prompt fixture | exact byte match | pass |
| Exemplar selection | 2 selected exemplar ids | exact id match | pass |
| Parser golden output | 2 arrays (`labels`, `bbox`) | exact value match | pass |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutprompter/REPRODUCING.md) for the commands that prepare prompt assets, generate reference outputs, run parity checks, save prompt configuration, and smoke-test local loading.


## Environmental Impact

No new model training is performed by this prompt-only package. Parity costs depend on the selected prompt fixtures, provider configuration, and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutPrompter serializes few-shot exemplars, calls a configured `pydantic-ai` model, and parses either `seq` or `html` layout text into normalized boxes and request-local labels. `save_pretrained` stores prompt configuration and parser settings rather than learned weights.

The reference implementation is Microsoft LayoutGeneration, under `vendor/ms-layout-generation/LayoutPrompter` when the vendor source is available in this repository. The demo script uses a tiny synthetic WebUI-style example:

```bash
uv run --package layoutprompter python models/layoutprompter/scripts/demo.py
```

Without `OPENAI_API_KEY`, the demo exits with a skip message.

### Compute Infrastructure

Vendor parity commands are deterministic CPU checks for prompt serialization, exemplar selection, and parser behavior.

#### Hardware

CPU is sufficient for import, smoke tests, and recorded vendor parity checks.

#### Software

Use `uv run --package layoutprompter ...` from the repository root so workspace dependency sources and extras resolve correctly.

The demo script uses a tiny synthetic WebUI-style example.

## License

Repository wrapper code is Apache-2.0. The original Microsoft LayoutGeneration repository is MIT licensed; use any dataset or LLM provider outputs under their own terms.

## Citation

```bibtex
@inproceedings{lin2023layoutprompter,
  title = {LayoutPrompter: Awaken the Design Ability of Large Language Models},
  author = {Lin, Jiawei and Guo, Jiaqi and Sun, Shizhao and Yang, Zijiang and Lou, Jian-Guang and Zhang, Dongmei},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```
