---
language:
  - en
license: "MIT"
library_name: "transformers"
pipeline_tag: "text-to-image"
tags:
  - "layout-corrector"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
  - "cyberagent/crello"
model-index:
  - name: "Layout-Corrector"
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

# Model Card for Layout-Corrector

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2409.16689&color=b31b1b&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=arXiv&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

Layout-Corrector ports the released transformer corrector that refines existing layouts with LayoutDM starter checkpoints.

## Model Details

### Model Description

Layout-Corrector is a training-free corrector module for discrete diffusion layout generators such as LayoutDM. Layout-Corrector is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `transformers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** https://github.com/line/Layout-Corrector
- **Paper:** https://arxiv.org/abs/2409.16689

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/layout-corrector-rico25` | not-published |
| PubLayNet | `creative-graphic-design/layout-corrector-publaynet` | not-published |
| Crello | `creative-graphic-design/layout-corrector-crello` | not-published |

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
uv sync --package layout-corrector
```

```python
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained(
    "creative-graphic-design/layout-corrector-rico25",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

When the LayoutDM starter pipeline and corrector model are loaded separately,
the explicit construction path is:

```python
pipe = LayoutCorrectorPipeline(layout_dm=layout_dm, corrector=corrector)
```

`scores` contains the Layout-Corrector confidence score when the selected call
path returns confidence estimates.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | `creative-graphic-design/PubLayNet` | default |
| Crello | `cyberagent/crello` | canonical source until an org mirror exists |

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

| Dataset | Seed | Token exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| RICO25 | 0 | 250/250 | 0 | 0 |
| RICO25 | 1 | 250/250 | 0 | 0 |
| RICO25 | 2 | 250/250 | 0 | 0 |
| PubLayNet | 0 | 250/250 | 0 | 0 |
| PubLayNet | 1 | 250/250 | 0 | 0 |
| PubLayNet | 2 | 250/250 | 0 | 0 |
| Crello | 0 | 250/250 | 0 | 0 |
| Crello | 1 | 250/250 | 0 | 0 |
| Crello | 2 | 250/250 | 0 | 0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites: run from the repository root and use an environment with CUDA when reproducing GPU parity. The commands below use GPU index 5; change it if needed. Initialize the required vendor submodules once with `git submodule update --init vendor/layout-corrector vendor/layout-dm`.

1. Download the original starter kit. This creates `.cache/layout-corrector/original/layout_corrector_starter.zip` and extracts `.cache/layout-corrector/original/layout_corrector_starter_kit/download`.

```bash
uv run --package layout-corrector --extra download python models/layout-corrector/scripts/download_original.py \
  --output-dir .cache/layout-corrector/original
```

2. Download the LayoutDM starter kit used by the nested LayoutDM parity fixtures. This creates `.cache/layout-dm/original/download`.

```bash
uv run --package layout-dm --extra download python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

3. Generate LayoutDM reference fixtures when refreshing LayoutDM golden files. Layout-Corrector logits parity does not need committed golden tensors, but LayoutDM parity fixtures remain useful for the nested pipeline.

```bash
for dataset in rico25 publaynet; do
  CUDA_VISIBLE_DEVICES=5 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/download \
    --output-dir "models/layout-dm/tests/vendor_parity/fixtures/${dataset}" \
    --sampling deterministic \
    --seed 0 \
    --batch-size 1
done
```

4. Convert the nested LayoutDM checkpoints for the supported LayoutDM datasets, then copy those converted pipelines into the Layout-Corrector parity directory structure. The current LayoutDM conversion script reads the released seed-0 checkpoint for each dataset and does not accept a `--seed` argument. Layout-Corrector logits parity only needs the nested tokenizer and scheduler interface, so the same converted nested LayoutDM pipeline is reused across corrector seeds. Crello-bbox uses the PubLayNet-compatible nested LayoutDM pipeline used by the parity test.

```bash
for dataset in rico25 publaynet; do
  uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/download \
    --output-dir ".cache/layout-corrector/converted/layoutdm/${dataset}/base"
done

for dataset in rico25 publaynet; do
  for seed in 0 1 2; do
    rm -rf ".cache/layout-corrector/converted/layoutdm/${dataset}/${seed}"
    cp -a \
      ".cache/layout-corrector/converted/layoutdm/${dataset}/base" \
      ".cache/layout-corrector/converted/layoutdm/${dataset}/${seed}"
  done
done

for seed in 0 1 2; do
  rm -rf ".cache/layout-corrector/converted/layoutdm/crello-bbox/${seed}"
  mkdir -p .cache/layout-corrector/converted/layoutdm/crello-bbox
  cp -a \
    .cache/layout-corrector/converted/layoutdm/publaynet/base \
    ".cache/layout-corrector/converted/layoutdm/crello-bbox/${seed}"
done
```

5. Run parity. This executes nine checks and prints exact-match and logits tolerance values.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --package layout-corrector --extra vendor pytest \
  models/layout-corrector/tests/vendor_parity \
  -q -m vendor_parity -s
```

6. Convert a Layout-Corrector checkpoint and run a `from_pretrained` smoke test.

```bash
uv run --package layout-corrector --extra convert python models/layout-corrector/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download \
  --corrector-job-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download/pretrained_weights/rico25/layout_corrector/0 \
  --layout-dm-dir .cache/layout-corrector/converted/layoutdm/rico25/0 \
  --output-dir .cache/layout-corrector/converted/layout-corrector-rico25-smoke

uv run --package layout-corrector python - <<'PY'
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained(
    ".cache/layout-corrector/converted/layout-corrector-rico25-smoke"
)
out = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
print(out.sequences.shape)
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

Use `uv run --package layout-corrector ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: MIT.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
