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
  # Use approved dataset ids from issue #2 and issue #60.
  - "creative-graphic-design/<dataset-id>"
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
          type: "creative-graphic-design/<dataset-id>"
          name: "<dataset-name>"
          config: "<dataset-config-or-not-applicable>"
          split: "<split-used-for-parity-or-evaluation>"
        metrics:
          - type: "<metric-or-vendor-parity>"
            value: "<numeric-or-string-result>"
            name: "<metric-display-name>"
---

# Model Card for <model-id>

<!-- Replace <model-id> with the planned `creative-graphic-design/<hub-repo-id>`. -->

<one-sentence summary with method name, venue, and arXiv or paper link>

<!-- Include the literature method name, conference or journal if known, and a stable paper URL. -->

## Model Details

### Model Description

<longer description of what the converted checkpoint does and what package loads it>

<!-- Mention normalized center xywh outputs, labels, mask semantics, and whether the wrapper is 🤗 `transformers` or `diffusers` style. -->

- **Developed by:** <upstream-authors-or-lab>
- **Funded by [optional]:** <funding-or-more-information-needed>
- **Shared by [optional]:** creative-graphic-design
- **Model type:** <architecture-and-task>
- **Language(s) (NLP):** <not-applicable-or-language-list>
- **License:** <license-id-or-unknown>
- **Finetuned from model [optional]:** <base-model-or-not-applicable>

### Model Sources [optional]

<!-- Link the exact sources used for conversion and parity. -->

- **Repository:** <original-implementation-url>
- **Paper [optional]:** <paper-or-arxiv-url>
- **Demo [optional]:** <demo-url-or-not-available>
- **Released weights:** <released-weights-url-or-status>

## Supported Checkpoints

<!-- Project-specific section placed after Model Sources because checkpoint publication state is model provenance. Use planned Hub ids from the model issue and mark Status as public, not-pushed, or planned. -->

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| <dataset-or-variant> | `creative-graphic-design/<hub-repo-id>` | <public-or-not-pushed> |

## Uses

<!-- Address intended users and downstream effects for generated layouts. -->

### Direct Use

<direct research or inference use case>

### Downstream Use [optional]

<how generated layouts may feed rendering, design tooling, retrieval, or evaluation pipelines>

### Out-of-Scope Use

<misuse and unsupported production, OCR, rendering, accessibility, or semantic-understanding claims>

## Bias, Risks, and Limitations

<known dataset, domain, layout-quality, license, and checkpoint limitations>

### Recommendations

<review, validation, and dataset-specific evaluation recommendations>

## How to Get Started with the Model

<!-- Use a local converted checkpoint path that works before Hub publication. -->

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

## Training Details

### Training Data

<!-- Use approved creative-graphic-design dataset ids and note dataset-specific quirks from issue #2 and issue #60. -->

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| <dataset-name> | `creative-graphic-design/<dataset-id>` | <pinned-config-or-dataset-quirk> |

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

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| <tokenizer-or-forward-or-sampling-check> | <case-count> | <exact-or-tolerance> | <passed-count-or-error> |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/<slug>/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

<!-- Put the command walkthrough in models/<slug>/REPRODUCING.md, not in this README. -->

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

## Citation [optional]

<!-- State the upstream paper citation required by the model issue. -->

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
