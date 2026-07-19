---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "text-to-image"
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

# Model Card for Coarse-to-Fine

![paper](https://img.shields.io/static/v1?label=paper&message=AAAI&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=AAAI%202022&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

Coarse-to-Fine ports the AAAI 2022 hierarchy-first layout generator into a [Transformers](https://huggingface.co/docs/transformers/index)-style package that returns the shared layout schema.

## Model Details

### Model Description

Coarse-to-Fine is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `transformers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

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
uv sync --package coarse-to-fine
```

```python
from coarse_to_fine import CoarseToFinePipeline

pipe = CoarseToFinePipeline.from_pretrained(
    "creative-graphic-design/coarse-to-fine-rico25",
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
| RICO25 | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| RICO25 | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `3.004e-05` |
| PubLayNet | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| PubLayNet | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `7.75e-06` |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites:

- Run commands from the repository root.
- Use `uv sync --package coarse-to-fine --extra vendor` once before the first vendor run.
- Initialize the vendor implementation with `git submodule update --init vendor/ms-layout-generation`.
- Keep downloaded weights and generated references under `.cache/coarse-to-fine/`; these files are local artifacts and are not committed.
- Set `CUDA_VISIBLE_DEVICES=3` for the vendor reference/parity run.

1. Download the public checkpoints into `.cache/coarse-to-fine/original`.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/download_original.py \
  --output-dir .cache/coarse-to-fine/original
```

2. Convert both public checkpoints into Transformers format.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --dataset rico25 \
  --output-dir .cache/coarse-to-fine/converted/rico25

uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --dataset publaynet \
  --output-dir .cache/coarse-to-fine/converted/publaynet
```

3. Generate vendor reference tensors.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset rico25 \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/rico25

CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset publaynet \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/publaynet
```

4. Run vendor parity tests against cached references.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine pytest \
  models/coarse-to-fine/tests/vendor_parity \
  -m vendor_parity \
  -q
```

Expected result with the fixtures above:

```text
4 passed
```

5. Smoke-test local `from_pretrained`.

```bash
uv run --package coarse-to-fine python - <<'PY'
from coarse_to_fine import (
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)

for dataset in ("rico25", "publaynet"):
    path = f".cache/coarse-to-fine/converted/{dataset}"
    model = CoarseToFineForLayoutGeneration.from_pretrained(path)
    processor = CoarseToFineProcessor.from_pretrained(path)
    pipe = CoarseToFinePipeline(model=model, processor=processor)
    out = pipe(batch_size=1, seed=0)
    assert out.bbox.shape == (1, 20, 4)
    assert out.labels.shape == (1, 20)
    assert bool(out.mask.any())
    print(model.config.dataset, out.bbox.shape, out.labels.shape)
PY
```

Expected output:

```text
rico25 torch.Size([1, 20, 4]) torch.Size([1, 20])
publaynet torch.Size([1, 20, 4]) torch.Size([1, 20])
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

Use `uv run --package coarse-to-fine ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: MIT.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
