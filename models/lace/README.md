---
language:
  - en
license: "mit"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "lace"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "RICO13"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LACE"
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

# Model Card for LACE

![OpenReview](https://img.shields.io/static/v1?label=OpenReview&message=kJ0qp9Xdsh&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=ICLR+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO13&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LACE](https://openreview.net/forum?id=kJ0qp9Xdsh), an ICLR 2024 layout diffusion editor, into a [Diffusers](https://huggingface.co/docs/diffusers/index)-style package for PubLayNet and RICO checkpoints.

## Model Details

### Model Description

LACE is a Diffusers-style layout generator that samples layouts under learned aesthetic constraints for document and UI domains. Converted checkpoints run unconditional generation for PubLayNet, RICO25, and the RICO13 label space when the corresponding original weights are available. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Jian Chen et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.

### Model Sources

- **Repository:** [LACE repository](https://github.com/puar-playground/LACE)
- **Paper:** [OpenReview paper page](https://openreview.net/forum?id=kJ0qp9Xdsh)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PubLayNet | [`creative-graphic-design/lace-publaynet`](https://huggingface.co/creative-graphic-design/lace-publaynet) | not-published |
| RICO13 | [`creative-graphic-design/lace-rico13`](https://huggingface.co/creative-graphic-design/lace-rico13) | planned; public vendor checkpoint not present in model.tar.gz |
| RICO25 | [`creative-graphic-design/lace-rico25`](https://huggingface.co/creative-graphic-design/lace-rico25) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

LACE runs unconditional generation from converted PubLayNet and RICO checkpoints. Local original checkpoints can be converted with:

```bash
uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output .cache/lace/converted/lace-publaynet
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

```bash
uv sync --package lace
```

```python
from lace import LacePipeline

pipe = LacePipeline.from_pretrained(
    "creative-graphic-design/lace-publaynet",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

The converted checkpoints are not yet on the Hugging Face Hub; until then, convert locally and load the `.cache/lace/converted/...` path produced by REPRODUCING.md.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| RICO13 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | vendor-derived RICO13 mapping |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

The original LACE project trains on PubLayNet and Rico annotations prepared as max-25 layout sequences.

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

| Dataset | Test | Shape | Max abs | rtol | atol |
| --- | --- | ---: | ---: | ---: | ---: |
| PubLayNet | Denoiser logits | `(2, 25, 10)` | 0.0 | 0 | 0 |
| Rico25 | Denoiser logits | `(2, 25, 30)` | 0.0 | 0 | 0 |

Conversion smoke tests pass for PubLayNet and Rico25. Rico13 conversion parity is skipped unless `.cache/lace/original/model/rico13_best.pt` is supplied, because the public `model.tar.gz` archive contains only `publaynet_best.pt` and `rico25_best.pt`.

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/lace/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream continuous diffusion architecture needed for conversion and inference while exposing the common layout schema. It converts original checkpoints into a `LacePipeline` that generates normalized center `xywh` boxes, category labels, and masks.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package lace ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original LACE implementation is released under the MIT License. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms.

Vendor links:

- Original implementation: https://github.com/puar-playground/LACE
- Checkpoints and datasets: https://huggingface.co/datasets/puar-playground/LACE
- Paper: https://openreview.net/forum?id=kJ0qp9Xdsh

## Citation

```bibtex
@inproceedings{
    chen2024towards,
    title={Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints},
    author={Jian Chen and Ruiyi Zhang and Yufan Zhou and Changyou Chen},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=kJ0qp9Xdsh}
}
```
