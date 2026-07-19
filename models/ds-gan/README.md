---
language:
  - en
license: other
license_name: upstream-license-review-needed
library_name: transformers
pipeline_tag: image-to-image
tags:
  - ds-gan
  - posterlayout
  - layout-generation
  - poster-generation
datasets:
  - creative-graphic-design/PKU-PosterLayout
model-index:
  - name: creative-graphic-design/ds-gan-pku-posterlayout
    results:
      - task:
          type: image-to-image
          name: Content-aware poster layout generation
        dataset:
          type: creative-graphic-design/PKU-PosterLayout
          name: PKU PosterLayout
          config: not-applicable
          split: test
        metrics:
          - type: vendor-parity
            value: exact
            name: DS-GAN vendor forward exact match
---

# Model Card for creative-graphic-design/ds-gan-pku-posterlayout

DS-GAN is the content-aware poster layout generator from PosterLayout, CVPR 2023.

## Model Details

### Model Description

This package converts the PosterLayout DS-GAN generator into a Transformers-style workspace member. The public pipeline accepts an RGB poster/background image plus saliency, runs the generator, and returns normalized center `xywh` boxes, dataset-local zero-based labels, and a boolean mask where `True` means a valid element.

- **Developed by:** HsiaoYuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang
- **Shared by:** creative-graphic-design
- **Model type:** ResNet-FPN conditioned GAN generator with a CNN-LSTM layout decoder
- **Language(s) (NLP):** Not applicable
- **License:** Upstream license review needed before publishing converted weights
- **Finetuned from model:** Not applicable

### Model Sources

- **Repository:** https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023
- **Paper:** https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html
- **Released weights:** Google Drive / PKU Netdisk links in the upstream README

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PKU PosterLayout DS-GAN Epoch 300 | `creative-graphic-design/ds-gan-pku-posterlayout` | not-pushed |

## Uses

### Direct Use

Use this package for research inference and conversion checks of content-aware poster layout generation.

### Downstream Use

Generated boxes and labels can be rendered by a separate design tool or used as layout priors for poster composition experiments.

### Out-of-Scope Use

This is not an OCR model, renderer, accessibility verifier, or production design QA system. It predicts layout structure only.

## Bias, Risks, and Limitations

The upstream repository does not include a top-level redistribution license. PKU PosterLayout access requires the upstream release agreement. Generated layouts may overlap, omit important elements, or reflect dataset-specific poster styles.

### Recommendations

Review generated layouts before downstream use, validate boxes against application constraints, and evaluate separately for each target poster domain.

## How to Get Started with the Model

```bash
uv sync --package ds-gan
```

```python
from ds_gan import DSGANPipeline

pipe = DSGANPipeline.from_pretrained(
    "creative-graphic-design/ds-gan-pku-posterlayout",
)
out = pipe(images=image, saliency=saliency, seed=0)

print(out.bbox)    # normalized center xywh boxes
print(out.labels)  # 0=text, 1=logo, 2=underlay
print(out.mask)    # valid element mask
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| PKU PosterLayout | `creative-graphic-design/PKU-PosterLayout` | `annotations.box_elem` is pixel `ltrb`; `annotations.cls_elem` includes `text`, `logo`, `underlay`, and `INVALID`. |

### Training Procedure

This package converts released DS-GAN weights and does not retrain the model.

#### Preprocessing

The processor resizes RGB images and saliency maps to `(350, 240)`, merges PFPNet and BASNet saliency by pixelwise max, and returns `pixel_values` shaped `(B, 4, 350, 240)` in `[0, 1]`. Public PKU labels exclude `INVALID`; vendor class id `0` is treated as no object and represented by `mask=False`.

#### Training Hyperparameters

- **Training regime:** Original DS-GAN training from the upstream PosterLayout implementation.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity fixtures are regenerated from the original implementation, released DS-GAN checkpoint, and PKU PosterLayout test images.

#### Factors

Parity is scoped to the PKU PosterLayout DS-GAN generator and fixed content-image conditioning.

#### Metrics

Forward parity compares class probabilities and normalized center boxes by exact tensor equality unless a documented hardware or operator-order root cause requires tolerance.

### Results

The converted model matches the vendor DS-GAN generator exactly when the released checkpoint and public PKU PosterLayout test images are regenerated with `CUDA_VISIBLE_DEVICES=1`.

## Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| `save_pretrained` -> `from_pretrained` smoke | 1 | output schema has `bbox`, `labels`, `mask`, `id2label` | passing in unit tests |
| Vendor DS-GAN class probabilities | 905 PKU test canvases x 32 slots | `rtol=0`, `atol=0` against regenerated vendor fixture | max abs diff `0.0`, exact |
| Vendor DS-GAN boxes | 905 PKU test canvases x 32 slots | `rtol=0`, `atol=0` against regenerated vendor fixture | max abs diff `0.0`, exact |

## Reproducibility

This section reproduces the original-implementation agreement checks for `creative-graphic-design/ds-gan-pku-posterlayout`.

Download vendor assets:

```bash
uv run --package ds-gan python models/ds-gan/scripts/download_original.py \
  --output-dir .cache/ds-gan/original \
  --include-test-data
```

Generate vendor reference outputs:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package ds-gan --extra vendor python models/ds-gan/scripts/generate_reference_outputs.py \
  --output .cache/ds-gan/fixtures/pku/reference_seed0.pt \
  --seed 0
```

Run vendor parity tests:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package ds-gan pytest models/ds-gan/tests/vendor_parity -m vendor_parity -rs
```

Convert checkpoints:

```bash
uv run --package ds-gan python models/ds-gan/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/ds-gan/original/DS-GAN-Epoch300.pth \
  --output-dir .cache/ds-gan/converted/ds-gan-pku-posterlayout
```

Run `from_pretrained` smoke tests:

```bash
uv run --package ds-gan python - <<'PY'
from pathlib import Path
import torch
from ds_gan import DSGANPipeline

path = Path(".cache/ds-gan/converted/ds-gan-pku-posterlayout")
pipe = DSGANPipeline.from_pretrained(path, local_files_only=True)
out = pipe(pixel_values=torch.zeros(1, 4, 350, 240), seed=0)
assert out.bbox.shape[-1] == 4
assert out.labels.shape == out.mask.shape
print(out.bbox.shape, out.labels.shape)
PY
```

## Environmental Impact

Original training compute and emissions are not reported in the released checkpoint bundle.

## Technical Specifications

### Model Architecture and Objective

DS-GAN uses a four-channel ResNet-FPN image encoder to initialize a bidirectional LSTM over an initial layout tensor shaped `(B, 32, 2, 4)`. The generator predicts vendor class probabilities over `no object`, `Text`, `Logo`, and `Underlay`, plus normalized center `xywh` boxes.

### Compute Infrastructure

Vendor parity regeneration should run on one selected CUDA device with fixed NumPy and Torch seeds.

#### Hardware

CUDA GPU recommended for parity against the released checkpoint.

#### Software

Use the root uv workspace and the `ds-gan` package commands shown above.
