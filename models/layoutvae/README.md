---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layoutvae"
  - "layout-generation"
datasets:
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutVAE"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/PubLayNet"
          name: "PubLayNet"
          split: "reference fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayoutVAE

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=1907.10719&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1907.10719)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2019&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package implements LayoutVAE as a stochastic scene layout generator that predicts PubLayNet document element boxes from a requested label set.

## Model Details

### Model Description

LayoutVAE is a `🤗transformers`-style implementation for label-conditioned PubLayNet layout generation. It returns normalized center `xywh` boxes, dataset-local labels, a valid-element mask, and `id2label`.

- **Developed by:** Akash Abdu Jyothi, Thibaut Durand, Jiawei He, Leonid Sigal, and Greg Mori.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** mit.

### Model Sources

- **Repository:** [Layout generation baselines](https://github.com/Layout-Generation/layout-generation)
- **Paper:** [arXiv 1907.10719](https://arxiv.org/abs/1907.10719)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PubLayNet | [`creative-graphic-design/layoutvae-publaynet`](https://huggingface.co/creative-graphic-design/layoutvae-publaynet) | not-published |

## Uses

### Direct Use

Use the pipeline for PubLayNet label-conditioned layout generation:

```python
from layoutvae import LayoutVAEPipeline

pipe = LayoutVAEPipeline.from_pretrained("creative-graphic-design/layoutvae-publaynet")
out = pipe(labels=[["text", "figure"]], seed=0)
print(out.bbox, out.labels, out.mask)
```

### Downstream Use

Generated layouts can feed document mockup rendering, layout evaluation, and design-system experiments after task-specific validation.

### Out-of-Scope Use

Do not use generated layouts as production document annotations, OCR, or accessibility metadata without separate validation.

## Bias, Risks, and Limitations

Generated layouts inherit PubLayNet label coverage and the stochastic behavior of the released method. Label-set conditioning does not enforce semantic reading order or document content consistency.

### Recommendations

Run the parity workflow before comparing converted checkpoints or publishing model artifacts.

## How to Get Started with the Model

Install the package directly from this repository. The command includes the shared layout library.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "layoutvae @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layoutvae"
```

Clone this repository, install the workspace member, and run the conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutvae/REPRODUCING.md). Those steps create `.cache/layoutvae/converted/layoutvae-publaynet`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layoutvae
uv run --package layoutvae python
```

```python
from layoutvae import LayoutVAEPipeline

path = ".cache/layoutvae/converted/layoutvae-publaynet"
# After Hub publication: from_pretrained("creative-graphic-design/layoutvae-publaynet")
pipe = LayoutVAEPipeline.from_pretrained(path)
out = pipe(labels=[["text", "figure"]], seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | PubMed Central document layout labels |

### Training Procedure

This member converts released behavior and does not train a new model in this repository.

#### Preprocessing

Public labels are PubLayNet IDs `0..4`; internal empty slots are exposed through `mask=False`.

#### Training Hyperparameters

- **Training regime:** original released training; not rerun in this repository.

#### Speeds, Sizes, Times

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Parity uses local-only fixed label-set fixtures and reference outputs generated by running the original modules. Large tensors and weights are not committed.

#### Factors

Parity is disaggregated by fixed-latent tensor math and seeded stochastic generation.

#### Metrics

Metrics are exact tensor equality for state-dict conversion, fixed-latent forward agreement for original-module outputs, and documented seeded agreement for the local PyTorch runtime.

### Parity Results

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| PubLayNet | original count and box state dicts vs. converted `LayoutVAEModel` | 74 tensors | `torch.testing.assert_close` defaults | passed |
| PubLayNet | fixed-latent count and box forward outputs vs. original modules | 2 label-set fixtures | max abs diff <= `1e-6`; observed `5.364418e-7` on boxes and `0.0` on counts | passed |
| PubLayNet | seeded stochastic end-to-end generation | 2 label-set fixtures | labels match; raw boxes are not claimed bit-exact because random draws are consumed through different PyTorch distribution paths | documented |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutvae/REPRODUCING.md) for the commands that prepare reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No new model training is performed here. Conversion and parity checks run on CPU for the first slice.

## Technical Specifications

### Model Architecture and Objective

LayoutVAE combines an autoregressive count module and an autoregressive box module. Public inference is label-conditioned PubLayNet generation.

### Compute Infrastructure

CPU is sufficient for conversion, parity, and smoke tests in this slice.

#### Hardware

No GPU is required for the listed commands.

#### Software

Use `uv run --package layoutvae ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is MIT licensed.

## Citation

```bibtex
@inproceedings{jyothi2019layoutvae,
  title = {LayoutVAE: Stochastic Scene Layout Generation from a Label Set},
  author = {Jyothi, Akash Abdu and Durand, Thibaut and He, Jiawei and Sigal, Leonid and Mori, Greg},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2019}
}
```
