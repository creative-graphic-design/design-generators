---
language:
  - en
license: "other"
license_name: "upstream-license-review-needed"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "ds-gan"
  - "posterlayout"
  - "layout-generation"
  - "poster-generation"
datasets:
  - "PosterLayout"
model-index:
  - name: "DS-GAN"
    results:
      - task:
          type: "other"
          name: "Content-aware poster layout generation"
        dataset:
          type: "PosterLayout"
          name: "PosterLayout"
          config: "PKU"
          split: "test"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for DS-GAN

[![paper](https://img.shields.io/static/v1?label=paper&message=CVPR+2023&color=blue&style=flat-square)](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PosterLayout&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [DS-GAN](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html), the content-aware poster layout generator from PosterLayout, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

DS-GAN is a content-aware poster layout generator that predicts text, logo, and underlay element geometry from an RGB poster canvas plus saliency maps. The package wraps the released generator as `DSGANModel`, `DSGANProcessor`, and `DSGANPipeline`; public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** HsiaoYuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware poster layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** upstream license review needed before publishing converted weights.

### Model Sources

- **Repository:** [PosterLayout-CVPR2023 repository](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023)
- **Paper:** [PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PKU PosterLayout DS-GAN Epoch 300 | [`creative-graphic-design/ds-gan-pku-posterlayout`](https://huggingface.co/creative-graphic-design/ds-gan-pku-posterlayout) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of content-aware poster layout generation.

Conditional generation accepts an RGB content image plus one saliency map, or PFPNet and BASNet saliency maps that are merged before resizing:

```python
from ds_gan import DSGANPipeline

pipe = DSGANPipeline.from_pretrained(
    "creative-graphic-design/ds-gan-pku-posterlayout"
)
out = pipe(images=image, saliency=saliency, seed=0)
print(out.bbox, out.labels, out.mask)
```

Model/processor usage is also available for lower-level checks:

```python
from ds_gan import DSGANModel, DSGANProcessor

model = DSGANModel.from_pretrained(
    "creative-graphic-design/ds-gan-pku-posterlayout"
).eval()
processor = DSGANProcessor.from_pretrained(
    "creative-graphic-design/ds-gan-pku-posterlayout"
)
encoded = processor(images=image, saliency=saliency)
out = model(**encoded)
decoded = processor.decode(class_probs=out.class_probs, bbox=out.bbox)
print(decoded.bbox, decoded.labels, decoded.mask)
```

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream poster composition systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoint, saliency preprocessing path, and PosterLayout data release. Dataset coverage, visual style, label vocabulary, and layout quality inherit the limits of those sources. The upstream repository does not include a top-level redistribution license.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation. Review generated layouts before downstream use and evaluate separately for each target poster domain.

## How to Get Started with the Model

Install the package directly from this repository. The command includes shared packages when they are not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "ds-gan @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/ds-gan"
```

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/REPRODUCING.md). Those steps create `.cache/ds-gan/converted/ds-gan-pku-posterlayout`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package ds-gan
uv run --package ds-gan python
```

```python
import torch

from ds_gan import DSGANPipeline

path = ".cache/ds-gan/converted/ds-gan-pku-posterlayout"
# After Hub publication: from_pretrained("creative-graphic-design/ds-gan-pku-posterlayout")
pipe = DSGANPipeline.from_pretrained(path, local_files_only=True)
out = pipe(pixel_values=torch.zeros(1, 4, 350, 240), seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| PosterLayout | PosterLayout | PKU release agreement; `box_elem` stores pixel `ltrb` boxes and `cls_elem` includes `text`, `logo`, `underlay`, and `INVALID` |

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Inputs and outputs are normalized to the public layout schema at package boundaries. RGB canvases and saliency maps are resized to `(350, 240)`. PFPNet and BASNet saliency are merged by native-resolution pixelwise max before resize. Vendor class id `0` represents no object and becomes `mask=False`.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures from the released DS-GAN checkpoint and PosterLayout test canvases. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is scoped to the PKU PosterLayout DS-GAN generator, fixed seed `0`, content-image conditioning, and 905 test canvases.

#### Metrics

Metrics are exact tensor equality for model outputs, exact processor pixel-value equality against the regenerated vendor fixture, and exact public postprocessing equality for `bbox`, `labels`, and `mask`.

### Parity Results

| Check | Cases | Match criterion | Result |
| --- | ---: | --- | --- |
| Vendor DS-GAN class probabilities | 905 canvases x 32 slots x 4 classes | `rtol=0`, `atol=0`, `max_abs=0.0`, `mismatched=0` | passed |
| Vendor DS-GAN boxes | 905 canvases x 32 slots x 4 values | `rtol=0`, `atol=0`, `max_abs=0.0`, `mismatched=0` | passed |
| Processor `pixel_values` | 905 canvases x 4 x 350 x 240 pixels | `max_abs=0.0`, `mismatched=0` against regenerated fixture | passed |
| Public postprocessing | `bbox`, `labels`, and `mask` for 905 decoded layouts | exact equality against decoded vendor fixture | passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No new model training is performed by this conversion package. Conversion and parity costs depend on local hardware and the selected checkpoint.

## Technical Specifications

### Model Architecture and Objective

DS-GAN loads the released PosterLayout generator checkpoint. A four-channel ResNet-FPN image encoder initializes a bidirectional LSTM over an initial layout tensor shaped `(B, 32, 2, 4)`. The generator predicts vendor class probabilities over `no_object`, `text`, `logo`, and `underlay`, plus normalized center `xywh` boxes.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and smoke tests. CUDA is recommended for heavyweight vendor parity against the released checkpoint.

#### Software

Use `uv run --package ds-gan ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The upstream PosterLayout repository does not include a top-level redistribution license; converted weights and Hub model repositories are held until license and release terms are reviewed.

## Citation

```bibtex
@inproceedings{hsu2023posterlayout,
  title = {PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout},
  author = {Hsu, HsiaoYuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2023},
  pages = {6018--6026}
}
```
