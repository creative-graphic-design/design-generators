---
language:
  - en
license: "other"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layout-action"
  - "layout-generation"
datasets:
  - "RICO13"
  - "creative-graphic-design/PubLayNet"
  - "InfoPPT"
model-index:
  - name: "LayoutAction"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "RICO13"
          name: "RICO13"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "requires original assets"
            name: "Vendor parity"
---

# Model Card for LayoutAction

[![paper](https://img.shields.io/static/v1?label=paper&message=AAAI+2023&color=blue&style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/26277)
![venue](https://img.shields.io/static/v1?label=venue&message=AAAI+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO13&color=informational&style=flat-square)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![dataset](https://img.shields.io/static/v1?label=dataset&message=InfoPPT&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutAction](https://ojs.aaai.org/index.php/AAAI/article/view/26277), the AAAI 2023 action-sequence layout generator, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

LayoutAction is an autoregressive GPT-style model over synthetic action tokens. Each layout element is represented by one label token and four `(option, object_reference, value)` triples for `x`, `y`, `w`, and `h`. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** LayoutAction authors.
- **Shared by:** creative-graphic-design.
- **Model type:** content-agnostic layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** original LayoutAction license not found; converted weight publication requires maintainer approval.

### Model Sources

- **Repository:** [LayoutActionProject](https://github.com/BERYLSHEEP/LayoutActionProject)
- **Paper:** [AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/26277)
- **Original assets:** Google Drive resources referenced by the original repository.

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO13 | `creative-graphic-design/layout-action-rico13` | local conversion only |
| PubLayNet | `creative-graphic-design/layout-action-publaynet` | local conversion only |
| InfoPPT | `creative-graphic-design/layout-action-infoppt` | blocked pending license and data approval |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of action-token layout generation.

```python
from layout_action import LayoutActionPipeline

pipe = LayoutActionPipeline.from_pretrained(
    "creative-graphic-design/layout-action-publaynet"
)
out = pipe(batch_size=2, condition_type="unconditional", seed=42)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
```

### Downstream Use

The converted package can be used as a token-level causal LM for LayoutAction action sequences or through `LayoutActionPipeline` for the common layout schema.

### Out-of-Scope Use

The package does not publish converted weights while the original LayoutAction license is unresolved. It does not implement image-conditioned, relation-conditioned, retrieval, refinement, or label-size generation.

## Bias, Risks, and Limitations

Layout quality follows the original checkpoints and datasets. RICO uses the vendor RICO13 subset and label order, PubLayNet uses the original document labels, and InfoPPT remains vendor-distribution-only until an org-hosted dataset exists.

### Recommendations

Use local converted checkpoints only after validating the original asset provenance and license. Treat stochastic top-k parity as device- and RNG-order-sensitive until exact reference artifacts are regenerated.

## How to Get Started with the Model

Install the workspace member and load a converted checkpoint directory.

```bash
uv run --package layout-action python models/layout-action/scripts/smoke_from_pretrained.py \
  .cache/layout-action/converted/layout-action-publaynet
```

## Training Details

### Training Data

The original checkpoints were trained on vendor-processed RICO13, PubLayNet, and InfoPPT resources. RICO13 uses a LayoutAction-specific label order, PubLayNet matches the original document labels, and InfoPPT is read from the vendor distribution.

### Training Procedure

Training follows the original GPT-style causal language-model objective over padded action-token sequences. This package focuses on checkpoint conversion and inference; it does not add a new training entry point.

## Evaluation

### Parity Results

| Check | Cases | Match Criterion | Result |
| --- | ---: | --- | --- |
| Strict checkpoint conversion | 2 | `missing_keys == []` and `unexpected_keys == []` for RICO13 and PubLayNet | Pass |
| Fixed-input logits | 6 | `torch.equal` for RICO13/PubLayNet across `random_generate`, `category_generate`, and `completion_generate` references | Pass |
| Stochastic top-k sequences | 6 | Exact token ids with seed 42 on `CUDA_VISIBLE_DEVICES=0` | Pass |
| Category forced labels | 2 | Exact forced label-token positions for RICO13 and PubLayNet | Pass |
| Decoded layouts | 6 | Exact `bbox`, `labels`, and `mask` after token parity | Pass |

The reference artifacts were generated from the released Google Drive checkpoints with synthetic fixed action-token prompts because the public Resources folder includes LayoutGAN++ processed data but not LayoutAction `preprocess_data` files. The converted checkpoints matched the vendor GPT and sampler bit-exactly for `random_generate`, `category_generate`, and `completion_generate`. Checkpoint SHA256 values were `c3a65ff7c1bf996d76f56ae15070c6c41f817744e5fb91752e1cb96600231614` for RICO13 and `08cd7d6bd2c745e90c8c7ce3c8b25627d004d7d7efc69b986e0f2b6b9e89ec17` for PubLayNet. InfoPPT was not present in the downloaded folder and remains pending original-author assets.

## Reproducibility

The original-implementation agreement checks can be reproduced by downloading the vendor assets, generating references with one selected GPU, running the vendor-parity tests, converting checkpoints, and running the local smoke test.

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-action/REPRODUCING.md) for the copy-pasteable commands.

## Citation

```bibtex
@inproceedings{layoutaction2023,
  title = {Layout Generation as Intermediate Action Sequence Prediction},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2023}
}
```
