---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "coarse-to-fine"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "Coarse-to-Fine"
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

# Model Card for Coarse-to-Fine

[![paper](https://img.shields.io/static/v1?label=paper&message=AAAI&color=blue&style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/19994)
![venue](https://img.shields.io/static/v1?label=venue&message=AAAI+2022&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [Coarse-to-Fine](https://ojs.aaai.org/index.php/AAAI/article/view/19994), an AAAI 2022 hierarchy-first layout generator, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package that returns the shared layout schema.

## Model Details

### Model Description

Coarse-to-Fine generates page or UI layouts through a hierarchy-aware `transformers` model: it predicts coarse group structure before refining element boxes and labels for RICO25 or PubLayNet. It supports unconditional local inference from converted checkpoints and exposes hierarchy traces through `intermediates` when requested. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Zhaoyun Jiang et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration Coarse-to-Fine](https://github.com/microsoft/LayoutGeneration/tree/main/Coarse-to-Fine)
- **Paper:** [AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/19994)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/coarse-to-fine-rico25`](https://huggingface.co/creative-graphic-design/coarse-to-fine-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/coarse-to-fine-publaynet`](https://huggingface.co/creative-graphic-design/coarse-to-fine-publaynet) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

The released checkpoints support unconditional hierarchical generation only. Canonical `condition_type` names are normalized at the API boundary, and unsupported modes fail explicitly.

| `condition_type` | Required inputs | Support |
| --- | --- | --- |
| `unconditional` | none | supported |
| `label`, `label_size`, `completion`, `refinement`, `text`, `content_image`, `relation`, `hierarchical`, `retrieval` | not applicable | raises `NotImplementedError` |

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/coarse-to-fine/REPRODUCING.md). Those steps create `.cache/coarse-to-fine/converted/rico25`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package coarse-to-fine
uv run --package coarse-to-fine python
```

```python
from coarse_to_fine import (
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)

path = ".cache/coarse-to-fine/converted/rico25"
# After Hub publication: from_pretrained("creative-graphic-design/coarse-to-fine-rico25")
model = CoarseToFineForLayoutGeneration.from_pretrained(path)
processor = CoarseToFineProcessor.from_pretrained(path)
pipe = CoarseToFinePipeline(model=model, processor=processor)

out = pipe(batch_size=2, seed=0, return_intermediates=True)
print(out.bbox.shape)
print(out.labels.shape)
print(out.id2label[int(out.labels[out.mask][0])])
print(out.intermediates["hierarchy"]["group_bbox"].shape)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

RICO25 public labels are zero-based; vendor tensors use one-based labels with SOS/EOS ids. PubLayNet COCO-style document boxes are converted to normalized `ltwh`, discretized on the vendor 128-bin grid, and returned publicly as normalized center `xywh`.

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

| Dataset | Comparison target | Cases | Agreement criterion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| RICO25 | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `3.004e-05` |
| PubLayNet | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| PubLayNet | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `7.75e-06` |

Detailed max-abs values for the logits checks were:

| Dataset | Group bbox | Group label histogram | Grouped bbox | Grouped label | Argmax |
| --- | ---: | ---: | ---: | ---: | --- |
| RICO25 | `3.004e-05` | `2.026e-06` | `2.098e-05` | `2.193e-05` | exact |
| PubLayNet | `3.814e-06` | `5.960e-07` | `5.600e-06` | `7.750e-06` | exact |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/coarse-to-fine/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

Coarse-to-Fine uses an autoregressive Transformer decoder with separate heads for element labels, group label histograms, and four coordinate bins. Generated hierarchy data is returned through `intermediates` when requested. There is no single unified token vocabulary for layout serialization, so Coarse-to-Fine uses a `ProcessorMixin` processor instead of a `PreTrainedTokenizer`.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package coarse-to-fine ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original Microsoft LayoutGeneration repository is MIT licensed. Converted code in this package is for research conversion and evaluation of the released Coarse-to-Fine checkpoints.

## Citation

```bibtex
@inproceedings{jiang2022coarse,
  title = {Coarse-to-Fine Generative Modeling for Graphic Layouts},
  author = {Jiang, Zhaoyun and Sun, Shizhao and Zhu, Jihua and Lou, Jian-Guang and Zhang, Dongmei},
  booktitle = {AAAI},
  year = {2022}
}
```
