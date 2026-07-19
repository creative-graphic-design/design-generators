---
language:
  - en
license: "other"
license_name: "review-needed"
license_link: "unknown"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "layoutdiffusion"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutDiffusion"
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

# Model Card for LayoutDiffusion

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.11589&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutDiffusion](https://arxiv.org/abs/2303.11589), the ICCV 2023 discrete layout diffusion model, into a [Diffusers](https://huggingface.co/docs/diffusers/index)-style package for RICO25 and PubLayNet.

## Model Details

### Model Description

LayoutDiffusion is a discrete Diffusers pipeline for RICO25 and PubLayNet layouts. It samples tokenized label and box sequences with unconditional, label-conditioned, and refinement modes, then converts vendor token coordinates into the shared layout schema. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Junyi Zhang et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** unknown.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration LayoutDiffusion](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion) is the vendored source under `vendor/ms-layout-generation/LayoutDiffusion`; original checkpoint assets are downloaded from [Junyi42/layoutdiffusion](https://huggingface.co/Junyi42/layoutdiffusion).
- **Paper:** [arXiv 2303.11589](https://arxiv.org/abs/2303.11589)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutdiffusion-rico25`](https://huggingface.co/creative-graphic-design/layoutdiffusion-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/layoutdiffusion-publaynet`](https://huggingface.co/creative-graphic-design/layoutdiffusion-publaynet) | not-published |

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
uv sync --package layoutdiffusion
```

```python
from layoutdiffusion import LayoutDiffusionPipeline

pipe = LayoutDiffusionPipeline.from_pretrained(
    "creative-graphic-design/layoutdiffusion-rico25",
)
out = pipe(batch_size=1, generator=None, condition_type="unconditional", seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

The converted checkpoints are not yet on the Hugging Face Hub; until then, convert locally and load the `.cache/layoutdiffusion/converted/...` path produced by REPRODUCING.md.

Use `condition_type` to select unconditional generation or supported
conditioning paths. `unconditional`, `label`, and `refinement` are the supported
canonical modes for this package; aliases such as `ugen`, `cat_cond`, `c`,
`gen_t`, and `refine` are normalized before sampling. Unsupported canonical
modes raise explicit errors instead of falling back silently.

```python
unconditional = pipe(batch_size=1, condition_type="ugen", seed=101)
label_conditioned = pipe(
    condition_type="cat_cond",  # canonical alias: "label"
    labels=[[1, 2, 3]],
    num_elements=[3],
    seed=101,
)
refined = pipe(
    condition_type="refine",  # canonical alias: "refinement"
    labels=[[1, 2, 3]],
    bbox=[[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.1, 0.2]]],
    mask=[[True, True, True]],
    seed=101,
)
print(unconditional.bbox, label_conditioned.mask, refined.bbox)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

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

### Results

The `## Parity Results` table reports the available numeric agreement evidence.

## Parity Results

| Dataset | Comparison target | Cases | Agreement criterion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| RICO25 | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| RICO25 | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `1.0252e-05`, max rel `5.9171e-03` |
| RICO25 | Full unconditional sample | 121 tokens | exact id match | 121/121 |
| PubLayNet | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| PubLayNet | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| PubLayNet | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `5.7220e-06`, max rel `5.6200e-04` |
| PubLayNet | Full unconditional sample | 121 tokens | exact id match | 121/121 |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutdiffusion/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


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

Use `uv run --package layoutdiffusion ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original LayoutDiffusion checkout does not include a LayoutDiffusion-specific top-level license; only the vendored Transformers license is present. Converted weights should not claim an OSS license until upstream license status is confirmed.

## Citation

```bibtex
@inproceedings{zhang2023layoutdiffusion,
  title = {LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models},
  author = {Junyi Zhang and Jiaqi Guo and Shizhao Sun and Jian-Guang Lou and Dongmei Zhang},
  booktitle = {ICCV},
  year = {2023}
}
```
