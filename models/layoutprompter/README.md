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

![paper](https://img.shields.io/static/v1?label=paper&message=LayoutPrompter&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=NeurIPS%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PosterLayout&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package provides [LayoutPrompter](https://arxiv.org/abs/2311.06495) as a [Pydantic AI](https://ai.pydantic.dev/) agent for prompt-based few-shot layout generation with normalized layout-array outputs.

## Model Details

### Model Description

LayoutPrompter is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `pydantic-ai`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** English prompts for prompt-only operation.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

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

Pass any Pydantic AI model object through `LayoutPrompterConfig.model`. If `model` is omitted, the agent checks `LAYOUTPROMPTER_MODEL`, then `PYDANTIC_AI_MODEL`, and finally falls back to `openai:gpt-4o-mini`.

`condition_type` accepts the public names and common vendor aliases: `label`, `label_size`, `relation`, `completion`, `refinement`, `text`, `cat_cond`, `gen_t`, `size_cond`, `gen_ts`, `gen_r`, `partial`, and `refine`. The package includes prompt serialization and parsing for `seq` and `html` layouts. Closed string modes such as dataset names, condition types, prompt formats, output type, and box format are normalized to enums at public boundaries while string inputs remain accepted.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The packaged behavior follows the upstream prompt fixtures, parser rules, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing prompt configurations or comparing new results against the original implementation.

## How to Get Started with the Model

```bash
uv sync --package layoutprompter
```

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
print(output.bbox)  # normalized center xywh with shape (batch, elements, 4)
```

Provider-backed configuration uses the same `LayoutPrompterConfig.model` field.

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

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PosterLayout | not recorded | built-in label vocabulary |
| WebUI | not recorded | deterministic local fixture |

Built-in label vocabularies are available for `publaynet`, `rico`, `posterlayout`, and `webui`.

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
| Prompt byte equality | 1 prompt fixture | exact byte match | pass |
| Exemplar selection | 2 selected exemplar ids | exact id match | pass |
| Parser golden output | 2 arrays (`labels`, `bbox`) | exact value match | pass |
| Torch-to-numpy output equivalence | 1 public parser fixture | exact JSON match for `bbox`, `labels`, `mask`, and `id2label` | pass |

## Reproducibility

This section reproduces the parity verification against the original implementation.
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

## Model Examination

Interpretability and failure-analysis artifacts are not recorded in the current README.

## Environmental Impact

No new model training is performed by this prompt-only package. Parity costs depend on the selected prompt fixtures, provider configuration, and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream architecture needed for conversion and inference while exposing the common layout schema.

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

Regular package checks are:

```bash
uv run --package layoutprompter pytest models/layoutprompter/tests -m "not vendor_parity and not integration"
uv run --package layoutprompter --with pytest-cov pytest models/layoutprompter/tests -m "not vendor_parity and not integration" --cov=layoutprompter --cov-report=term-missing
uv run --package layoutprompter ruff check models/layoutprompter
uv run --package layoutprompter ty check models/layoutprompter
```

The current package coverage under the coverage command above is 99%.

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
