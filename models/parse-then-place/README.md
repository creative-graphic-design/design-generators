---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "parse-then-place"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "Web"
model-index:
  - name: "Parse-Then-Place"
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

# Model Card for Parse-Then-Place

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2308.12700&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Web&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [Parse-Then-Place](https://arxiv.org/abs/2308.12700), the ICCV 2023 two-stage parsed-scene and placement method, into a [Transformers](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

Parse-Then-Place generates layouts through a two-stage Transformers pipeline: a semantic parser first extracts layout intent from text, then a placement model predicts element positions. Converted checkpoints include parser and placement subfolders for RICO and Web variants. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Jiawei Lin et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration repository](https://github.com/microsoft/LayoutGeneration)
- **Paper:** [arXiv 2308.12700](https://arxiv.org/abs/2308.12700)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO finetune | [`creative-graphic-design/parse-then-place-rico-finetune`](https://huggingface.co/creative-graphic-design/parse-then-place-rico-finetune) | not-published |
| RICO pretrain | [`creative-graphic-design/parse-then-place-rico-pretrain`](https://huggingface.co/creative-graphic-design/parse-then-place-rico-pretrain) | not-published |
| Web finetune | [`creative-graphic-design/parse-then-place-web-finetune`](https://huggingface.co/creative-graphic-design/parse-then-place-web-finetune) | not-published |
| Web pretrain | [`creative-graphic-design/parse-then-place-web-pretrain`](https://huggingface.co/creative-graphic-design/parse-then-place-web-pretrain) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

The pipeline owns the two-stage text-to-layout workflow: semantic parser -> placement constraint -> placement model -> common `LayoutGenerationOutput`. Full checkpoint inference requires converted `semantic_parser/` and `placement/` subfolders containing standard `AutoModelForSeq2SeqLM` weights.

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
uv sync --package parse-then-place
```

```python
from parse_then_place import (
    ParseThenPlaceConfig,
    ParseThenPlacePipeline,
    ParseThenPlaceProcessor,
)

config = ParseThenPlaceConfig(dataset_name="rico")
processor = ParseThenPlaceProcessor.from_config("rico")
pipe = ParseThenPlacePipeline(config=config, processor=processor)
out = pipe(prompt="create a screen with text", layout_text="text 0 0 10 20")

print(out.bbox.shape, out.labels.tolist())
```

The converted checkpoints are not yet on the Hugging Face Hub; until then, convert locally and load the `.cache/parse-then-place/converted/...` path produced by REPRODUCING.md.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| Web | unknown | vendor web pretraining split |

Datasets are RICO text-to-layout and WebUI as distributed by the original repository. WebUI is not yet mirrored under the `creative-graphic-design` org, so conversion scripts accept the original asset tree as the interim source.

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

| Check | Cases | Criterion | Result |
| --- | ---: | --- | ---: |
| RICO finetune stage-2 token ids | 1 prompt x 5 samples | Exact generated token-id sequence match | 5/5 |
| RICO finetune stage-2 decoded layouts | 1 prompt x 5 samples | Exact decoded layout string match | 5/5 |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/parse-then-place/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream two-stage text-to-layout architecture needed for conversion and inference while exposing the common layout schema. It loads standard seq2seq stage models from `semantic_parser/` and `placement/` subfolders.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package parse-then-place ...` from the repository root so workspace dependency sources and extras resolve correctly.

The original adapter/deepspeed training stack uses legacy pins (`adapter_transformers==3.0.1`, `deepspeed==0.5.10`, `wandb==0.12.10`) that conflict with this workspace's modern Hugging Face stack. Keep those pins in a separate vendor-parity environment when flattening stage-1 adapter weights; the published runtime artifact should be standard Transformers weights.

## License

Repository wrapper code is Apache-2.0. The original implementation is MIT licensed and published at `microsoft/LayoutGeneration` under `Parse-Then-Place`.

## Citation

```bibtex
@InProceedings{lin2023parse,
  title={A Parse-Then-Place Approach for Generating Graphic Layouts from Textual Descriptions},
  author={Lin, Jiawei and Guo, Jiaqi and Sun, Shizhao and Xu, Weijiang and Liu, Ting and Lou, Jian-Guang and Zhang, Dongmei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
