---
language:
  - en
license: "mit"
library_name: "diffusers"
pipeline_tag: "text-to-image"
tags:
  - "lace"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LACE"
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

# Model Card for LACE

![OpenReview](https://img.shields.io/static/v1?label=OpenReview&message=kJ0qp9Xdsh&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=ICLR%202024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO13&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

LACE ports the ICLR 2024 layout diffusion editor into a Diffusers-style package for PubLayNet and RICO checkpoints.

## Model Details

### Model Description

LACE is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `diffusers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** https://github.com/puar-playground/LACE
- **Paper:** https://openreview.net/forum?id=kJ0qp9Xdsh

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PubLayNet | `creative-graphic-design/lace-publaynet` | not-published |
| RICO13 | `creative-graphic-design/lace-rico13` | planned; public vendor checkpoint not present in model.tar.gz |
| RICO25 | `creative-graphic-design/lace-rico25` | not-published |

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

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | ui-screenshots-and-hierarchies-with-semantic-annotations |
| RICO13 | `creative-graphic-design/Rico` | vendor-derived RICO13 mapping |
| PubLayNet | `creative-graphic-design/PubLayNet` | default |

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

| Dataset | Test | Shape | Max abs | rtol | atol |
| --- | --- | ---: | ---: | ---: | ---: |
| PubLayNet | Denoiser logits | `(2, 25, 10)` | 0.0 | 0 | 0 |
| Rico25 | Denoiser logits | `(2, 25, 30)` | 0.0 | 0 | 0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Run the following commands from the repository root. They keep downloaded
weights under `.cache/lace/original`, local reference metadata under
`.cache/lace/reference`, and converted pipelines under `.cache/lace/converted`.
The commands use `CUDA_VISIBLE_DEVICES=2`; change that value if your CUDA device
assignment differs.

Prerequisite for a fresh clone:

```bash
git submodule update --init vendor/lace
```

1. Download and extract the original checkpoint archive:

```bash
uv run --package lace python models/lace/scripts/download_vendor_assets.py \
  --output-dir .cache/lace/original \
  --filename model.tar.gz \
  --extract
```

Expected files after extraction:

```text
.cache/lace/original/model/publaynet_best.pt
.cache/lace/original/model/rico25_best.pt
```

2. Record local reference metadata for the available vendor checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output-dir .cache/lace/reference/publaynet \
  --seed 123

CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output-dir .cache/lace/reference/rico25 \
  --seed 123
```

Generated metadata:

```text
.cache/lace/reference/publaynet/metadata.json
.cache/lace/reference/rico25/metadata.json
```

3. Run the parity tests:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace pytest \
  models/lace/tests/vendor_parity/test_lace_parity.py \
  -m vendor_parity -vv -rs
```

Expected result with the public archive is `4 passed, 1 skipped`. The skipped
case is `test_checkpoint_conversion_smoke[rico13-rico13_best.pt]`, because the
public archive does not contain `.cache/lace/original/model/rico13_best.pt`.

4. Convert the available checkpoints:

```bash
rm -rf .cache/lace/converted/lace-publaynet .cache/lace/converted/lace-rico25

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output .cache/lace/converted/lace-publaynet

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output .cache/lace/converted/lace-rico25
```

Each output directory contains Diffusers pipeline files and `README.md`:

```text
.cache/lace/converted/lace-publaynet/README.md
.cache/lace/converted/lace-rico25/README.md
```

5. Load the converted checkpoints with `from_pretrained` and run a short smoke:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python - <<'PY'
from lace import LacePipeline

for name in ["lace-publaynet", "lace-rico25"]:
    pipe = LacePipeline.from_pretrained(f".cache/lace/converted/{name}")
    pipe = pipe.to("cuda")
    out = pipe(batch_size=1, seed=0, num_inference_steps=2)
    in_range = bool((out.bbox >= 0).all() and (out.bbox <= 1).all())
    print(name, tuple(out.bbox.shape), tuple(out.labels.shape), in_range)
PY
```

Expected output shapes are `(1, 25, 4)` for `bbox` and `(1, 25)` for `labels`;
the final boolean should be `True`.

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

Use `uv run --package lace ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: MIT.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
