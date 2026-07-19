---
language:
  - en
license: "other"
license_name: "review-needed"
license_link: "not recorded"
library_name: "diffusers"
pipeline_tag: "text-to-image"
tags:
  - "layoutdiffusion"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutDiffusion"
    results:
      - task:
          type: "text-to-image"
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

# Model Card for LayoutDiffusion

![paper](https://img.shields.io/static/v1?label=paper&message=ICCV%202023&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

This package ports [LayoutDiffusion](https://arxiv.org/abs/2303.11589), the ICCV 2023 discrete layout diffusion model, into a [Diffusers](https://huggingface.co/docs/diffusers/index)-style package for RICO25 and PubLayNet.

## Model Details

### Model Description

LayoutDiffusion is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `diffusers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** review-needed.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** not recorded in the current README
- **Paper:** [arXiv 2303.11589](https://arxiv.org/abs/2303.11589)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutdiffusion-rico25`](https://huggingface.co/creative-graphic-design/layoutdiffusion-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/layoutdiffusion-publaynet`](https://huggingface.co/creative-graphic-design/layoutdiffusion-publaynet) | not-published |

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
uv sync --package layoutdiffusion
```

```python
from layoutdiffusion import LayoutDiffusionPipeline

pipe = LayoutDiffusionPipeline.from_pretrained(
    "creative-graphic-design/layoutdiffusion-rico25",
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
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

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

| Dataset | Comparison target | Cases | Agreement criterion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| RICO25 | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| RICO25 | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `1.0252e-05`, max rel `5.9171e-03` |
| RICO25 | Full unconditional sample | 121 tokens | exact id match | 121/121 |
| PubLayNet | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| PubLayNet | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| PubLayNet | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `5.7220e-06`, max rel `5.6200e-04` |
| PubLayNet | Full unconditional sample | 121 tokens | exact id match | 121/121 |

## Reproducibility

These commands reproduce the original-implementation agreement checks against the vendored LayoutDiffusion code, fixed seed 101, and converted local checkpoints.

```bash
uv run --package layoutdiffusion --extra download \
  python models/layoutdiffusion/scripts/download_original.py \
  --output-dir .cache/layoutdiffusion/original
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion --extra vendor \
  --with spacy --with pyyaml --with sacremoses \
  python models/layoutdiffusion/scripts/generate_reference_outputs.py \
  --dataset all \
  --seed 101
```

```bash
uv run --package layoutdiffusion \
  python models/layoutdiffusion/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --checkpoint-dir .cache/layoutdiffusion/original/results/checkpoint/discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5 \
  --checkpoint-name ema_0.9999_175000.pt \
  --output-dir .cache/layoutdiffusion/converted/layoutdiffusion-rico25
```

```bash
uv run --package layoutdiffusion \
  python models/layoutdiffusion/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --checkpoint-dir .cache/layoutdiffusion/original/results/checkpoint/gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub \
  --checkpoint-name ema_0.9999_400000.pt \
  --output-dir .cache/layoutdiffusion/converted/layoutdiffusion-publaynet
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion pytest \
  models/layoutdiffusion/tests -m vendor_parity
```

```bash
uv run --package layoutdiffusion python - <<'PY'
from layoutdiffusion import LayoutDiffusionPipeline

for name in ["layoutdiffusion-rico25", "layoutdiffusion-publaynet"]:
    pipe = LayoutDiffusionPipeline.from_pretrained(f".cache/layoutdiffusion/converted/{name}")
    out = pipe(batch_size=1, seed=101, num_inference_steps=1)
    print(name, out.bbox.shape, out.labels.shape, out.mask.shape)
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

Use `uv run --package layoutdiffusion ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: review-needed.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
