---
language:
  - en
license: "MIT"
library_name: "diffusers"
pipeline_tag: "text-to-image"
tags:
  - "layout-flow"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutFlow"
    results:
      - task:
          type: "text-to-image"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/Rico"
          name: "RICO25"
          config: "ui-screenshots-and-hierarchies-with-semantic-annotations"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayoutFlow

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2403.18187&color=b31b1b&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=ECCV%202024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

LayoutFlow ports the ECCV 2024 flow-matching model for layout generation into a Diffusers-style package.

## Model Details

### Model Description

LayoutFlow is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `diffusers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** https://github.com/julianguerreiro/LayoutFlow
- **Paper:** https://arxiv.org/abs/2403.18187

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/layout-flow-rico25` | not-published |
| PubLayNet | `creative-graphic-design/layout-flow-publaynet` | not-published |

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
uv sync --package layout-flow
```

```python
from layout_flow import LayoutFlowPipeline

pipe = LayoutFlowPipeline.from_pretrained(
    "creative-graphic-design/layout-flow-rico25",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | `creative-graphic-design/PubLayNet` | default |

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

| Dataset | Vector field max abs | Vector field max rel | Euler trajectory max abs | Euler trajectory max rel |
| --- | ---: | ---: | ---: | ---: |
| PubLayNet | 0.0 | 0.0 | 0.0 | 0.0 |
| RICO25 | 0.0 | 0.0 | 0.0 | 0.0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

### Prerequisites

```bash
git submodule update --init vendor/layout-flow
```

Run the commands below from the repository root. The original source under `vendor/layout-flow` is read-only; downloads and generated artifacts are written under `.cache/layout-flow`.

1. Download the original checkpoints. This creates `.cache/layout-flow/original/checkpoints/checkpoint_PubLayNet_LayoutFlow.ckpt` and `.cache/layout-flow/original/checkpoints/checkpoint_RICO_LayoutFlow.ckpt`.

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py
```

2. Generate vendor golden/reference tensors. `CUDA_VISIBLE_DEVICES=3` selects the requested GPU; the script writes `.cache/layout-flow/golden/*_vendor_vector_field.pt` and `.cache/layout-flow/golden/summary.json`.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layout-flow --extra vendor python models/layout-flow/scripts/generate_reference_outputs.py --dataset all
```

3. Run the parity pytest suite against both released checkpoints.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layout-flow --extra vendor pytest models/layout-flow/tests/vendor_parity -m vendor_parity -rs
```

4. Convert both checkpoints to local Diffusers pipeline directories. These commands write `.cache/layout-flow/converted/publaynet` and `.cache/layout-flow/converted/rico25`, each with a `README.md`.

```bash
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset publaynet
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset rico25
```

5. Smoke-test local `from_pretrained` loading.

```bash
uv run --package layout-flow python - <<'PY'
from layout_flow import LayoutFlowPipeline

for dataset in ["publaynet", "rico25"]:
    pipe = LayoutFlowPipeline.from_pretrained(f".cache/layout-flow/converted/{dataset}")
    out = pipe(batch_size=1, num_elements=4, seed=0, num_inference_steps=2)
    print(dataset, out.bbox.shape, out.labels.shape, type(out).__name__)
PY
```

Each script supports `--help` with defaults documented:

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py --help
uv run --package layout-flow python models/layout-flow/scripts/generate_reference_outputs.py --help
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py --help
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

Use `uv run --package layout-flow ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: MIT.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
