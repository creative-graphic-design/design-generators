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

![paper](https://img.shields.io/static/v1?label=paper&message=ICCV%202023&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Web&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

This package ports [Parse-Then-Place](https://arxiv.org/abs/2308.12700), the ICCV 2023 two-stage parsed-scene and placement method, into a [Transformers](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

Parse-Then-Place is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `transformers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

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
from parse_then_place import ParseThenPlacePipeline

pipe = ParseThenPlacePipeline.from_pretrained(
    "creative-graphic-design/parse-then-place-rico-finetune",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| Web | not recorded | vendor web pretraining split |

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Inputs and outputs are normalized to the public layout schema at package boundaries. Vendor-specific boxes, tokens, prompts, or analog bits stay inside package adapters and parity fixtures.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training time and carbon measurements are not recorded in the current README.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or prompt fixture where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality, exact token or byte equality, or explicitly stated numeric tolerance against the vendor path.

### Results

The numeric agreement record is the `## Parity Results` table below. Rows marked as not recorded are documentation gaps rather than inferred measurements.

## Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | ---: |
| RICO finetune stage-2 token ids | 1 prompt x 5 samples | Exact generated token-id sequence match | 5/5 |
| RICO finetune stage-2 decoded layouts | 1 prompt x 5 samples | Exact decoded layout string match | 5/5 |

## Reproducibility

Reproduce original-implementation agreement by downloading vendor assets,
generating fixed-seed references on one GPU, running marked parity tests, then
converting and smoke-loading the pipeline checkpoint.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/download_original_assets.py \
  --output-dir .cache/parse-then-place/assets

CUDA_VISIBLE_DEVICES=4 uv run --package parse-then-place python models/parse-then-place/scripts/generate_reference_outputs.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --seed 42 \
  --num-examples 1 \
  --output-dir .cache/parse-then-place/reference/rico-finetune

CUDA_VISIBLE_DEVICES=4 PARSE_THEN_PLACE_ORIGINAL_ROOT=.cache/parse-then-place/assets \
  PARSE_THEN_PLACE_REFERENCE_DIR=.cache/parse-then-place/reference/rico-finetune \
  uv run --package parse-then-place pytest models/parse-then-place/tests/vendor_parity -m vendor_parity

uv run --package parse-then-place python models/parse-then-place/scripts/convert_original_checkpoint.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --parser-state-path .cache/parse-then-place/assets/ckpt/rico/stage1/pytorch_model.bin \
  --output-dir .cache/parse-then-place/converted/rico-finetune

uv run --package parse-then-place python - <<'PY'
from pathlib import Path

from parse_then_place import ParseThenPlacePipeline

root = Path(".cache/parse-then-place/converted/rico-finetune")
pipeline = ParseThenPlacePipeline.from_pretrained(
    root,
    local_files_only=True,
)
print(
    pipeline.config.dataset_name,
    pipeline.parser is not None,
    pipeline.placement is not None,
)
PY
```

## Model Examination

Interpretability and failure-analysis artifacts are not recorded in the current README.

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

Use `uv run --package parse-then-place ...` from the repository root so workspace dependency sources and extras resolve correctly.

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
