---
language:
  - en
license: "agpl-3.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layoutganpp"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
  - "creative-graphic-design/magazine"
model-index:
  - name: "LayoutGAN++"
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

# Model Card for LayoutGAN++

[![DOI](https://img.shields.io/static/v1?label=DOI&message=10.1145%2F3474085.3475497&color=blue&style=flat-square&logo=doi&logoColor=white)](https://doi.org/10.1145/3474085.3475497)
![venue](https://img.shields.io/static/v1?label=venue&message=ACM+MM+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=AGPL--3.0&color=orange&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Magazine&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/magazine)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutGAN++](https://doi.org/10.1145/3474085.3475497), the [Const-layout](https://github.com/ktrk115/const_layout) generator method, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package under the literature method name.

## Model Details

### Model Description

LayoutGAN++ is a `transformers`-style wrapper around the Const-layout generator for RICO25, PubLayNet, and Magazine layouts. It generates element boxes from label prompts with the converted generator and processor while keeping vendor-specific label and coordinate handling behind the package boundary. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Kotaro Kikuchi et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** agpl-3.0.

### Model Sources

- **Repository:** [Const-layout repository](https://github.com/ktrk115/const_layout)
- **Paper:** [DOI 10.1145/3474085.3475497](https://doi.org/10.1145/3474085.3475497)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutganpp-rico`](https://huggingface.co/creative-graphic-design/layoutganpp-rico) | not-published |
| PubLayNet | [`creative-graphic-design/layoutganpp-publaynet`](https://huggingface.co/creative-graphic-design/layoutganpp-publaynet) | not-published |
| Magazine | [`creative-graphic-design/layoutganpp-magazine`](https://huggingface.co/creative-graphic-design/layoutganpp-magazine) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

Conditional generation accepts label prompts for the converted generator:

```python
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
)
out = pipe(labels=[["text", "figure"]], seed=0)
print(out.bbox, out.labels, out.mask)
```

Model/processor usage is also available for lower-level checks:

```python
from layoutganpp import LayoutGANPPModel, LayoutGANPPProcessor

model = LayoutGANPPModel.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
).eval()
processor = LayoutGANPPProcessor.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
)
encoded = processor([["text", "figure"]])
out = model.generate(**encoded, seed=0)
print(processor.batch_decode(out.bbox, out.labels, out.mask))
```

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Install the package directly from this repository. The command includes shared packages when they are not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "layoutganpp @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layoutganpp"
```

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutganpp/REPRODUCING.md). Those steps create `.cache/layoutganpp/converted/layoutganpp-rico`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layoutganpp
uv run --package layoutganpp python
```

```python
from layoutganpp import LayoutGANPPPipeline

path = ".cache/layoutganpp/converted/layoutganpp-rico"
# After Hub publication: from_pretrained("creative-graphic-design/layoutganpp-rico")
pipe = LayoutGANPPPipeline.from_pretrained(path)
out = pipe(labels=[["Toolbar", "Image"]], seed=0)

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
| Magazine | [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine) | polygon-based train-only source |

The LayoutGAN++ `rico` checkpoint uses the original 13-class RICO label space (`Toolbar`, `Image`, `Text`, ...), not the 25-class RICO25 mapping used by most other layout packages in this repository. The dataset badge and metadata point to [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) because that is the canonical hosted source; the processor keeps the checkpoint-local RICO13 label mapping in `id2label`.

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

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| Rico | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox; `torch.testing.assert_close(atol=1e-6, rtol=1e-5)` | passed |
| PubLayNet | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox; `torch.testing.assert_close(atol=1e-6, rtol=1e-5)` | passed |
| Magazine | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 33, 4)` bbox; `torch.testing.assert_close(atol=1e-6, rtol=1e-5)` | passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutganpp/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutGAN++ loads the released Const-layout generator checkpoints and predicts element boxes from label prompts. The runtime surface covers generator inference and processor decoding only; the original latent optimization loop, training pipeline, rendered image generation, and dataset preprocessing code are outside this wrapper.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layoutganpp ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under GNU AGPLv3. Converted checkpoints and package metadata keep `agpl-3.0` model-card metadata.

## Citation

```bibtex
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {ACM International Conference on Multimedia},
    series = {MM '21},
    year = {2021},
    pages = {88--96},
    doi = {10.1145/3474085.3475497}
}
```
