---
# Model README draft template. Hub-facing cards are generated separately through
# `laygen.common.model_card` and `huggingface_hub.ModelCard.from_template`.
language:
  # Use `en` unless the model consumes or emits natural-language content in another language.
  - en
license: "<license-id>"
# Choose a Hub-recognized license id; use `other` only with license_name/license_link below.
license_name: "<custom-license-name>"
# Fill only when `license: other`; otherwise remove this key.
license_link: "<license-file-or-url>"
# Fill only when `license: other`; otherwise remove this key.
library_name: "<diffusers-or-transformers>"
# Match the runtime integration used by the package.
pipeline_tag: "<hub-task-tag>"
# Choose the closest Hub task tag; document layout-generation specificity in tags.
tags:
  # Include method, package, dataset, and `layout-generation` or `poster-generation`.
  - "<method-tag>"
  - "<layout-generation-or-poster-generation>"
datasets:
  # Use canonical dataset ids. RICO25 uses the Rico Hub dataset with config
  # ui-screenshots-and-hierarchies-with-semantic-annotations.
  - "creative-graphic-design/Rico"
metrics:
  # Use Hub metric ids when applicable; remove this key when parity has no Hub metric id.
  - "<metric-id>"
base_model: "<base-model-hub-id>"
# Use a Hub model id when the checkpoint derives from one; otherwise remove this key.
# Required only for weight-backed packages. Prompt-only packages without learned
# checkpoints must not include `model-index`.
model-index:
  - name: "<model-id>"
    results:
      - task:
          type: "<hub-task-tag>"
          name: "<human-readable-task-name>"
        dataset:
          type: "creative-graphic-design/Rico"
          name: "<dataset-name>"
          config: "ui-screenshots-and-hierarchies-with-semantic-annotations"
          split: "<split-used-for-parity-or-evaluation>"
        metrics:
          - type: "<metric-or-vendor-parity>"
            value: "<numeric-or-string-result>"
            name: "<metric-display-name>"
---

<!-- --8<-- [start:card] -->

<!-- Replace <model-id> with the planned `creative-graphic-design/<hub-repo-id>`. -->
<!-- Include the arXiv, venue, license, base-library, dataset, vendor-parity, and Hub-status badges required by the model README checker. -->

# Model Card for <model-id>

This package ports <method linked to its paper>, <venue and key idea>, into a <linked framework>-style package.

<!-- Include the literature method name, conference or journal if known, and a stable paper URL. -->
<!-- Link the framework name to its official documentation, for example [`🧨diffusers`](https://huggingface.co/docs/diffusers/index). -->

## Model Details

### Model Description

<longer description of what the converted checkpoint does and what package loads it>

<!-- Mention normalized center xywh outputs, labels, mask semantics, and whether the wrapper is `🤗transformers` or `🧨diffusers` style. -->

- **Developed by:** <upstream-authors-or-lab>
- **Funded by [optional]:** <funding-or-more-information-needed>
- **Shared by [optional]:** creative-graphic-design
- **Model type:** <content-agnostic|content-aware>; task: <task-agnostic|task-aware|single-task|evaluation|saliency>; conditioning: <canonical values in conventions order>.
- **Language(s) (NLP):** <not-applicable-or-language-list>
- **License:** <license-id-or-unknown>
- **Finetuned from model [optional]:** <base-model-or-not-applicable>

### Model Sources

<!-- Link the exact sources used for conversion and parity. -->

- **Repository:** <original-implementation-url>
- **Paper [optional]:** <paper-or-arxiv-url>
- **Demo [optional]:** <demo-url-or-not-available>
- **Released weights:** <released-weights-url-or-status>

## Supported Checkpoints

<!-- Use the model issue's checkpoint/dataset/task matrix. Mark an unpublished Hub id as not-published until publication is complete. Prompt-only packages describe reusable configuration and exemplars rather than learned checkpoints. -->

| Checkpoint           | Hub ID                                  | Status        |
| -------------------- | --------------------------------------- | ------------- |
| <dataset-or-variant> | `creative-graphic-design/<hub-repo-id>` | not-published |

## Uses

<!-- Address intended users and downstream effects for generated layouts. -->

### Direct Use

<direct research or inference use case>

### Downstream Use

<!-- Describe how generated layouts may feed rendering, design tooling, retrieval, or evaluation pipelines. -->

<how generated layouts may feed rendering, design tooling, retrieval, or evaluation pipelines>

### Out-of-Scope Use

<misuse and unsupported production, OCR, rendering, accessibility, or semantic-understanding claims>

## Bias, Risks, and Limitations

<known dataset, domain, layout-quality, license, and checkpoint limitations>

### Recommendations

<review, validation, and dataset-specific evaluation recommendations>

## How to Get Started with the Model

<!-- Lead with a local converted checkpoint path that works before Hub publication. Mark Hub ids as not-published until they are published. -->

Install the package and its unpublished workspace dependencies together.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "<member-name> @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/<slug>"
```

<!-- Include posgen in the same pip command when required, and preserve required shared-package extras. -->

For local conversion and reproduction, use the repository checkout. Follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/<slug>/REPRODUCING.md) to create `.cache/<slug>/converted/<local-checkpoint-dir>` before loading it.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package <member-name>
uv run --package <member-name> python
```

```python
from <package_name> import <PipelineOrModelClass>

pipe = <PipelineOrModelClass>.from_pretrained(
    ".cache/<slug>/converted/<local-checkpoint-dir>",
)
# After Hub publication: from_pretrained("creative-graphic-design/<hub-repo-id>")
out = pipe(batch_size=1, seed=0)

print(out.bbox)    # normalized center xywh boxes
print(out.labels)  # dataset-local integer labels
print(out.mask)    # valid element mask
```

```text
# Paste the captured output from running the snippet as written.
```

## Training Details

### Training Data

<!-- Use canonical dataset ids and note dataset-specific configs or quirks. -->

| Dataset   | Dataset ID                                                                                               | Notes                                                             |
| --------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| RICO25    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)           | config `ui-screenshots-and-hierarchies-with-semantic-annotations` |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default config                                                    |

### Training Procedure

<!-- State whether this package converts released weights or trains from scratch. -->

#### Preprocessing [optional]

<bbox-format, label-space, tokenizer, image, saliency, or hierarchy preprocessing>

#### Training Hyperparameters

- **Training regime:** <original-training-regime-or-not-retrained>

#### Speeds, Sizes, Times [optional]

<training-time-checkpoint-size-or-more-information-needed>

## Evaluation

<!-- Keep package-specific evaluation facts here. -->

### Testing Data, Factors & Metrics

#### Testing Data

<vendor-reference-datasets-and-fixtures>

#### Factors

<dataset-or-condition-disaggregation-used-in-parity>

#### Metrics

<exact-match-tolerance-and-smoke-test-metrics>

### Parity Results

<!-- Fill with numeric results from the real vendor-parity suite; do not replace this with prose. -->

| Check                                    |        Cases | Criterion            | Result                  |
| ---------------------------------------- | -----------: | -------------------- | ----------------------- |
| <tokenizer-or-forward-or-sampling-check> | <case-count> | <exact-or-tolerance> | <passed-count-or-error> |

## Reproducibility

Reproduce agreement with the original implementation using [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/<slug>/REPRODUCING.md), which covers asset download, original-code reference generation, required-asset parity checks, conversion, and local loading.

<!-- Put the command walkthrough in models/<slug>/REPRODUCING.md, not in this README. -->
<!-- For prompt-only methods, replace checkpoint conversion with configuration/exemplar serialization and state that there are no learned checkpoints. For retraining claims, link models/<slug>/TRAINING.md and report only its demonstrated dataset/seed scope. -->
<!-- Commands that need CUDA should use CUDA_VISIBLE_DEVICES=<gpu-index> and explain that <gpu-index> is the selected local CUDA device. Do not hardcode machine-specific GPU numbers. -->

## Model Examination [optional]

<interpretability-or-error-analysis-status>

## Environmental Impact

<training-or-conversion-carbon-information-or-more-information-needed>

## Technical Specifications [optional]

### Model Architecture and Objective

<architecture-objective-and-output-schema>

### Compute Infrastructure

<conversion-and-parity-compute-requirements>

#### Hardware

<cpu-gpu-requirements>

#### Software

<python-package-and-vendor-extra-requirements>

## Citation

<!-- Verify the upstream paper citation against the paper or proceedings page before filling this block. Do not infer author lists from memory. -->

```bibtex
<bibtex-citation>
```

## Glossary [optional]

<project-terms-such-as-xywh-mask-tokenizer-or-parity>

## More Information [optional]

<additional-links-or-follow-up-issues>

## Model Card Authors [optional]

creative-graphic-design maintainers.

## Model Card Contact

Open an issue or pull request in the creative-graphic-design design-generators repository.

<!-- --8<-- [end:card] -->
