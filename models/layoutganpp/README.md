---
language:
  - en
license: "agpl-3.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layoutganpp"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
  - "creative-graphic-design/magazine"
model-index:
  - name: "LayoutGAN++"
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

# Model Card for LayoutGAN++

![DOI](https://img.shields.io/static/v1?label=DOI&message=10.1145%2F3474085.3475497&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=SIGGRAPH%20Asia%202021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=AGPL--3.0&color=orange&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Magazine&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

This package ports [LayoutGAN++](https://doi.org/10.1145/3474085.3475497), the [Const-layout](https://github.com/ktrk115/const_layout) generator method, into a [Transformers](https://huggingface.co/docs/transformers/index)-style package under the literature method name.

## Model Details

### Model Description

LayoutGAN++ is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `transformers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** agpl-3.0.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** [Const-layout repository](https://github.com/ktrk115/const_layout)
- **Paper:** [DOI 10.1145/3474085.3475497](https://doi.org/10.1145/3474085.3475497)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutganpp-rico`](https://huggingface.co/creative-graphic-design/layoutganpp-rico) | not-published |
| PubLayNet | [`creative-graphic-design/layoutganpp-publaynet`](https://huggingface.co/creative-graphic-design/layoutganpp-publaynet) | not-published |
| Magazine | [`creative-graphic-design/layoutganpp-magazine`](https://huggingface.co/creative-graphic-design/layoutganpp-magazine) | not-published |

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
uv sync --package layoutganpp
```

```python
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained(
    "creative-graphic-design/layoutganpp-rico",
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
| Magazine | not mirrored | polygon-based train-only source |

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

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| Rico | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |
| PubLayNet | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |
| Magazine | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 33, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/const-layout` before generating vendor references. The command examples keep all local artifacts under `.cache/layoutganpp/` and never modify `vendor/const-layout`.

Step 1 downloads the original released checkpoints. Generated files are `.cache/layoutganpp/original/layoutganpp_{rico,publaynet,magazine}.pth.tar`.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/download_original_weights.py \
  --output-dir .cache/layoutganpp/original \
  --dataset all
```

Step 2 generates golden vendor fixtures with the const-layout code. The generated files are `.cache/layoutganpp/fixtures/<dataset>/reference_seed0.pt`. Set `CUDA_VISIBLE_DEVICES` to the GPU index you want the vendor model to use.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output .cache/layoutganpp/fixtures/rico/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output .cache/layoutganpp/fixtures/publaynet/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output .cache/layoutganpp/fixtures/magazine/reference_seed0.pt \
  --seed 0 \
  --batch-size 3
```

Step 3 runs the vendor parity tests against converted checkpoints. If step 4 has not been run yet, pytest will skip the cases because `.cache/layoutganpp/converted/layoutganpp-<dataset>` is missing.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp pytest models/layoutganpp/tests/vendor_parity -m vendor_parity -q
```

Step 4 converts each original checkpoint into a Transformers-style directory. Each output directory contains model weights, config, processor files, and a generated `README.md` model card.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-rico

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-publaynet

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-magazine
```

Step 5 runs a `from_pretrained` smoke check for each converted checkpoint. The expected bbox shapes are `(1, 2, 4)`.

```bash
uv run --package layoutganpp python - <<'PY'
from layoutganpp import LayoutGANPPPipeline

for dataset, labels in {
    "rico": [["Toolbar", "Image"]],
    "publaynet": [["text", "figure"]],
    "magazine": [["text", "image"]],
}.items():
    pipe = LayoutGANPPPipeline.from_pretrained(
        f".cache/layoutganpp/converted/layoutganpp-{dataset}"
    )
    out = pipe(labels=labels, seed=0)
    print(dataset, tuple(out.bbox.shape))
PY
```

After step 4, rerun step 3. The expected parity results are:

```text
rico: shape=(3, 9, 4), max_abs=0, max_rel=0
publaynet: shape=(3, 9, 4), max_abs=0, max_rel=0
magazine: shape=(3, 33, 4), max_abs=0, max_rel=0
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

Use `uv run --package layoutganpp ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under GNU AGPLv3. Converted checkpoints and package metadata keep `agpl-3.0` model-card metadata.

## Citation

```bibtex
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {ACM International Conference on Multimedia},
    series = {MM '21},
    year = {2021},
    pages = {88--96},
    doi = {10.1145/3474085.3475497}
}
```
