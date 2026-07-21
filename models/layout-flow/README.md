---
language:
  - en
license: "mit"
library_name: "diffusers"
pipeline_tag: "other"
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
          type: "other"
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

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2403.18187&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2403.18187)
![venue](https://img.shields.io/static/v1?label=venue&message=ECCV+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutFlow](https://arxiv.org/abs/2403.18187), the ECCV 2024 flow-matching model for layout generation, into a 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

LayoutFlow is a continuous-flow `diffusers` pipeline that predicts layout vector fields for RICO25 and PubLayNet. It supports unconditional and conditional generation through normalized condition aliases while converting vendor coordinate conventions into the shared public schema. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Julian Jorge Andrade Guerreiro et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.

### Model Sources

- **Repository:** [LayoutFlow repository](https://github.com/julianguerreiro/LayoutFlow)
- **Paper:** [arXiv 2403.18187](https://arxiv.org/abs/2403.18187)
- **Project page:** [LayoutFlow project page](https://julianguerreiro.github.io/layoutflow/)
- **Original checkpoint host:** [JulianGuerreiro/LayoutFlow](https://huggingface.co/JulianGuerreiro/LayoutFlow)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layout-flow-rico25`](https://huggingface.co/creative-graphic-design/layout-flow-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/layout-flow-publaynet`](https://huggingface.co/creative-graphic-design/layout-flow-publaynet) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

The original method models layouts as continuous flow-matching trajectories over normalized boxes and analog-bit category labels, then solves the generation path with an increasing-time ODE from `t=0` to `t=1`.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-flow/REPRODUCING.md). Those steps create `.cache/layout-flow/converted/rico25`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-flow
uv run --package layout-flow python
```

```python
from layout_flow import LayoutFlowPipeline

path = ".cache/layout-flow/converted/rico25"
# After Hub publication: from_pretrained("creative-graphic-design/layout-flow-rico25")
pipe = LayoutFlowPipeline.from_pretrained(path)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

The released LayoutFlow checkpoints were trained on the RICO and PubLayNet splits distributed by the original authors; this repository aligns dataset metadata to the org datasets above.

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

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or prompt fixture where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality, exact token or byte equality, or explicitly stated numeric tolerance against the vendor path.

### Parity Results

| Dataset | Compared path | Cases | Assertion |
| --- | --- | ---: | --- |
| PubLayNet | vendor `LayoutDMBackbone` vector field vs. converted vector field | 1 synthetic batch | `max_abs <= 1e-6` (`atol=1e-6`), `max_rel <= 1e-5` (`rtol=1e-5`) |
| RICO25 | vendor `LayoutDMBackbone` vector field vs. converted vector field | 1 synthetic batch | `max_abs <= 1e-6` (`atol=1e-6`), `max_rel <= 1e-5` (`rtol=1e-5`) |

Parity is tested against the original vendor `LayoutDMBackbone` vector-field path using the released checkpoints. The local pipeline uses `LayoutFlowEulerScheduler` for inference, but no committed parity test compares an Euler trajectory against the vendor.

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-flow/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutFlow models layouts as continuous vectors and predicts the flow field used to sample element boxes and labels. The processor accepts unconditional and conditional aliases such as `gen_t`, `cat_cond`, and `size_cond`, then converts vendor coordinate conventions to normalized center `xywh` outputs.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layout-flow ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is MIT licensed.

## Citation

```bibtex
@inproceedings{guerreiro2024layoutflow,
  title={LayoutFlow: Flow Matching For Layout Generation},
  author={Guerreiro, Julian Jorge Andrade and Inoue, Naoto and Masui, Kento and Otani, Mayu and Nakayama, Hideki},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2024},
  organization={Springer}
}
```
