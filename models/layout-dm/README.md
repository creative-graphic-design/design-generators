---
language:
  - en
license: "apache-2.0"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "layout-dm"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutDM"
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

# Model Card for LayoutDM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.08137&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.08137)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
[![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/creative-graphic-design/layoutdm-rico25)

This package ports [LayoutDM](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html), the CVPR 2023 discrete diffusion model for controllable layout generation, into a [Diffusers](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

LayoutDM is a discrete diffusion layout generator for controllable UI and document layouts. It denoises tokenized element categories, positions, and sizes, with condition aliases for unconditional, label, label-size, completion, and refinement workflows. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Naoto Inoue et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [LayoutDM repository](https://github.com/CyberAgentAILab/layout-dm)
- **Paper:** [CVPR 2023 paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html)
- **Project page:** [LayoutDM project page](https://cyberagentailab.github.io/layout-dm/)
- **Starter checkpoint bundle:** [layoutdm_starter.zip](https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutdm-rico25`](https://huggingface.co/creative-graphic-design/layoutdm-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/layoutdm-publaynet`](https://huggingface.co/creative-graphic-design/layoutdm-publaynet) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

LayoutDM supports controllable discrete diffusion over layout tokens. The converted checkpoints follow the original release for RICO25 mobile UI layouts and PubLayNet document page layouts, both with at most 25 elements.

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
uv sync --package layout-dm
```

```python
from layout_dm import LayoutDMPipeline

pipe = LayoutDMPipeline.from_pretrained(
    "creative-graphic-design/layoutdm-rico25",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

The converted checkpoints are not yet on the Hugging Face Hub; until then, convert locally and load the `.cache/layout-dm/converted/...` path produced by REPRODUCING.md.

Conditional generation uses the same pipeline call with `condition_type`, `bbox`,
`labels`, and `mask`. Canonical names such as `label`, `label_size`,
`completion`, and `refinement` are accepted, and vendor aliases such as
`gen_t`, `gen_ts`, `cat_cond`, `c`, `cwh`, `partial`, and `refine` are normalized
at the pipeline boundary.

```python
out = pipe(
    condition_type="gen_t",  # canonical alias: "label"
    labels=[[1, 2, 3]],
    bbox=[[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.1, 0.2]]],
    mask=[[True, True, True]],
    seed=0,
)
print(out.bbox)
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

| Dataset | Tokenizer exact | Deterministic exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| RICO25 | 125/125 | 125/125 | 0 | 0 |
| PubLayNet | 125/125 | 125/125 | 0 | 0 |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-dm/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutDM represents each layout as discrete category, position, and size tokens and denoises those tokens through a discrete diffusion process. The processor normalizes vendor aliases such as `gen_t`, `gen_ts`, `partial`, and `refine` into public condition modes before decoding generated tokens into normalized boxes and labels.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layout-dm ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under the Apache-2.0 license.

## Citation

```bibtex
@inproceedings{inoue2023layout,
  title = {{LayoutDM: Discrete Diffusion Model for Controllable Layout Generation}},
  author = {Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023},
  pages = {10167-10176}
}
```
