---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "ralf"
  - "layout-generation"
  - "retrieval-augmented"
  - "content-aware-layout"
datasets:
  - "creative-graphic-design/CGL-Dataset"
  - "creative-graphic-design/PKU-PosterLayout"
model-index:
  - name: "RALF"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/CGL-Dataset"
          name: "CGL"
          config: "ralf-style"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for RALF

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2311.13602&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2311.13602)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PKU&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [RALF](https://arxiv.org/abs/2311.13602), the CVPR 2024 retrieval-augmented content-aware layout generator, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

RALF combines retrieved layout examples with content images and saliency maps for poster layout generation. This port exposes a `PreTrainedModel`, processor, tokenizer, retrieval helper, and `laygen.pipelines.LayoutGenerationPipeline` subclass for the CGL and PKU unconditional checkpoints. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Kotaro Kikuchi et al.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [RALF repository](https://github.com/CyberAgentAILab/RALF)
- **Paper:** [arXiv 2311.13602](https://arxiv.org/abs/2311.13602)
- **Project page:** [RALF project page](https://cyberagentailab.github.io/RALF/)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| CGL unconditional | [`creative-graphic-design/ralf-cgl-unconditional`](https://huggingface.co/creative-graphic-design/ralf-cgl-unconditional) | not-published |
| PKU unconditional | [`creative-graphic-design/ralf-pku-unconditional`](https://huggingface.co/creative-graphic-design/ralf-pku-unconditional) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of content-aware generated layouts.

Unconditional generation accepts optional content images, saliency maps, and explicit retrieval examples. Selected retrieval indexes are returned in `intermediates["retrieval"]` when `return_intermediates=True`.

Calling `pipe()` without images or retrieval data is a smoke/debug path that uses a zero image and zero retrieval memory. Use real content images, saliency maps, and retrieved examples for paper-equivalent inference.

### Downstream Use

Generated layouts may feed poster rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, saliency evaluation, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints and datasets. Dataset coverage, label vocabularies, saliency assumptions, retrieval quality, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Install the package directly from this repository. The command includes shared packages when they are not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "ralf @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/ralf"
```

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ralf/REPRODUCING.md). Those steps create `.cache/ralf/converted/ralf-cgl-unconditional-strict`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package ralf
uv run --package ralf python
```

```python
from ralf import RalfPipeline

path = ".cache/ralf/converted/ralf-cgl-unconditional-strict"
# After Hub publication: from_pretrained("creative-graphic-design/ralf-cgl-unconditional")
pipe = RalfPipeline.from_pretrained(path)
out = pipe(condition_type="unconditional", seed=0, top_k=5)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| CGL | [`creative-graphic-design/CGL-Dataset`](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset) | `ralf-style` poster layout conversion |
| PKU | [`creative-graphic-design/PKU-PosterLayout`](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout) | `ralf-style` poster layout conversion |

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Inputs and outputs are normalized to the public layout schema at package boundaries. Vendor-specific saliency maps, retrieval tensors, boxes, labels, and token ids stay inside package adapters and parity fixtures.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or fixture where the package has recorded evidence.

#### Metrics

Metrics are exact state-dict equality, exact tensor equality, or explicitly stated numeric tolerance against the vendor path.

### Parity Results

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| CGL | `ralf_uncond_cgl` strict conversion | 1 | 664 source keys, 664 target keys, 664 matched keys; missing and unexpected keys empty; converted state dict equals original checkpoint | passed |
| CGL | `ralf_uncond_cgl` local-vs-vendor logits | 1 synthetic GPU 0 batch | `max_abs_diff=0.0` | passed |
| PKU | `ralf_uncond_pku10` strict conversion | 1 | 664 source keys, 664 target keys, 664 matched keys; missing and unexpected keys empty; converted state dict equals original checkpoint | passed |
| PKU | `ralf_uncond_pku10` local-vs-vendor logits | 1 synthetic GPU 0 batch | `max_abs_diff=0.0` | passed |
| Synthetic CPU smoke | local `save_pretrained` reload | 1 | `from_pretrained` succeeds and returns `bbox`, `labels`, `mask`, and `id2label` | passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ralf/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

RALF loads the released retrieval-augmented layout transformer checkpoints and predicts poster elements from content-aware inputs plus retrieved examples. The current runtime surface covers unconditional CGL and PKU checkpoint inference; label, label-size, completion, refinement, relation, and other content-image variants raise explicit unsupported-condition errors.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and local smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package ralf ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under the Apache-2.0 license.

## Citation

```bibtex
@inproceedings{kikuchi2024ralf,
  title = {{Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation}},
  author = {Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```
