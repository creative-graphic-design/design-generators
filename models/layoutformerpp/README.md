---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layoutformerpp"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutFormer++"
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

# Model Card for LayoutFormer++

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2208.08037&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutFormer++](https://arxiv.org/abs/2208.08037), the autoregressive layout transformer method, into a [Transformers](https://huggingface.co/docs/transformers/index)-style pipeline.

## Model Details

### Model Description

LayoutFormer++ is a Transformers layout generator that models layout sequences across label, label-size, relation, refinement, completion, and unconditional tasks. Converted checkpoints use task-specific token vocabularies for RICO25 and PubLayNet while preserving the vendor discrete layout representation internally. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Zhaoyun Jiang et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration LayoutFormer++](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B)
- **Paper:** [arXiv 2208.08037](https://arxiv.org/abs/2208.08037)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 label | [`creative-graphic-design/layoutformerpp-rico-label`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-label) | not-published |
| RICO25 label-size | [`creative-graphic-design/layoutformerpp-rico-label-size`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-label-size) | not-published |
| RICO25 relation | [`creative-graphic-design/layoutformerpp-rico-relation`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-relation) | not-published |
| RICO25 refinement | [`creative-graphic-design/layoutformerpp-rico-refinement`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-refinement) | not-published |
| RICO25 completion | [`creative-graphic-design/layoutformerpp-rico-completion`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-completion) | not-published |
| RICO25 unconditional | [`creative-graphic-design/layoutformerpp-rico-unconditional`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-unconditional) | not-published |
| PubLayNet label | [`creative-graphic-design/layoutformerpp-publaynet-label`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-label) | not-published |
| PubLayNet label-size | [`creative-graphic-design/layoutformerpp-publaynet-label-size`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-label-size) | not-published |
| PubLayNet relation | [`creative-graphic-design/layoutformerpp-publaynet-relation`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-relation) | not-published |
| PubLayNet refinement | [`creative-graphic-design/layoutformerpp-publaynet-refinement`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-refinement) | not-published |
| PubLayNet completion | [`creative-graphic-design/layoutformerpp-publaynet-completion`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-completion) | not-published |
| PubLayNet unconditional | [`creative-graphic-design/layoutformerpp-publaynet-unconditional`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-unconditional) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

Each released task has an incompatible task-specific checkpoint, so Hub ids include the task suffix. Supported public conditions map to vendor tasks as follows: `gen_t` -> `label`, `gen_ts` -> `label_size`, `gen_r` -> `relation`, `refinement` -> `refinement`, `completion` -> `completion`, and `ugen` -> `unconditional`.

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
uv sync --package layoutformerpp
```

```python
from layoutformerpp import LayoutFormerPPPipeline

pipe = LayoutFormerPPPipeline.from_pretrained(
    "creative-graphic-design/layoutformerpp-rico-label",
)
out = pipe(condition_type="label", labels=[["Text", "Image"]], max_length=16)

print(out.bbox)  # normalized center xywh, shape (B, S, 4)
print(out.labels)  # dataset-local ids, padding controlled by out.mask
print(out.id2label)
```

The converted checkpoints are not yet on the Hugging Face Hub; until then, convert locally and load the `.cache/layoutformerpp/converted/...` path produced by REPRODUCING.md.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

RICO25 public outputs use zero-based dataset-local ids, while the internal LayoutFormer++ tokenizer uses one-based `label_<id>` tokens. PubLayNet COCO-style document boxes are converted to the internal discrete LayoutFormer++ `ltwh` token grid and returned publicly as normalized center `xywh`.

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

| Checkpoint | Public checkpoint | Vocab source | Tokenizer | Logits max abs | Logits max rel | Generation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `rico_gen_t` | present | `ckpts/rico_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `rico_gen_ts` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `rico_gen_r` | present | `ckpts/rico_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `rico_refinement` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `rico_completion` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `rico_ugen` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_gen_t` | present | `ckpts/publaynet_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `publaynet_gen_ts` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `publaynet_gen_r` | present | `ckpts/publaynet_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `publaynet_refinement` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `publaynet_completion` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_ugen` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutformerpp/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


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

Use `uv run --package layoutformerpp ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is MIT licensed in the parent Microsoft LayoutGeneration repository. Converted code in this package is for research conversion and evaluation of the released LayoutFormer++ checkpoints.

## Citation

```bibtex
@inproceedings{jiang2023layoutformerpp,
  title = {LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Yihan and Deng, Jingwen and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {CVPR},
  year = {2023}
}
```
