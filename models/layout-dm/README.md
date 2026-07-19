---
language:
  - en
license: "apache-2.0"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "layout-dm"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutDM"
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

# Model Card for LayoutDM

![paper](https://img.shields.io/static/v1?label=paper&message=CVPR%202023&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

This package ports [LayoutDM](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html), the CVPR 2023 discrete diffusion model for controllable layout generation, into a [Diffusers](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

LayoutDM is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `diffusers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** [LayoutDM repository](https://github.com/CyberAgentAILab/layout-dm)
- **Paper:** [CVPR 2023 paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/layoutdm-rico25`](https://huggingface.co/creative-graphic-design/layoutdm-rico25) | not-published |
| PubLayNet | [`creative-graphic-design/layoutdm-publaynet`](https://huggingface.co/creative-graphic-design/layoutdm-publaynet) | not-published |

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
uv sync --package layout-dm
```

```python
from layout_dm import LayoutDMPipeline

pipe = LayoutDMPipeline.from_pretrained(
    "creative-graphic-design/layoutdm-rico25",
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

| Dataset | Tokenizer exact | Deterministic exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| RICO25 | 125/125 | 125/125 | 0 | 0 |
| PubLayNet | 125/125 | 125/125 | 0 | 0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Run these commands from the repository root unless a block explicitly changes directory. The commands assume CUDA is available as device `0`, the vendored original implementation is present at `vendor/layout-dm`, and generated files may be written under `.cache/` and `models/layout-dm/tests/vendor_parity/fixtures/`. If this worktree does not contain the vendor submodule contents, pass `--vendor-dir /path/to/design-generators/vendor/layout-dm` to `generate_reference_outputs.py`.

Prerequisite:

```bash
git submodule update --init vendor/layout-dm
```

### 1. Download Original Weights

This downloads `layoutdm_starter.zip` and extracts the original release bundle. The repo-local cache location is `.cache/layout-dm/original`; the extracted starter directory used by later steps is `.cache/layout-dm/original/download`.

```bash
uv run --package layout-dm python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

### 2. Generate Golden Reference Tensors

This writes local-only parity fixtures for each dataset:

- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/tokenizer_io.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/denoiser_forward.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/sample_unconditional.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/meta.json`

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir models/layout-dm/tests/vendor_parity/fixtures/rico25 \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1

CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
  --dataset publaynet \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir models/layout-dm/tests/vendor_parity/fixtures/publaynet \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1
```

### 3. Run Vendor Parity Tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm pytest models/layout-dm/tests/vendor_parity -m vendor_parity -rs
```

Expected result with the fixtures above:

```text
6 passed
```

### 4. Convert Checkpoints And Smoke Test from_pretrained

The conversion step writes Diffusers pipeline directories under `.cache/layout-dm/converted/`. Each output includes model weights, tokenizer files, scheduler/processor config, and a Hub-style `README.md` model card.

Expected local output roots:

```text
.cache/layout-dm/converted/layoutdm-rico25/
.cache/layout-dm/converted/layoutdm-publaynet/
```

```bash
uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir .cache/layout-dm/converted/layoutdm-rico25

uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir .cache/layout-dm/converted/layoutdm-publaynet
```

Smoke test both converted checkpoints:

```bash
uv run --package layout-dm python - <<'PY'
from pathlib import Path
from layout_dm import LayoutDMPipeline

for dataset in ("rico25", "publaynet"):
    path = Path(".cache/layout-dm/converted") / f"layoutdm-{dataset}"
    pipe = LayoutDMPipeline.from_pretrained(path)
    out = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
    assert out.bbox.shape[-1] == 4
    assert out.labels.shape == out.mask.shape
    assert (path / "README.md").exists()
    print(dataset, out.bbox.shape, out.labels.shape)
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

Use `uv run --package layout-dm ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under the Apache-2.0 license.

## Citation

```bibtex
@inproceedings{inoue2023layout,
  title = {{LayoutDM: Discrete Diffusion Model for Controllable Layout Generation}},
  author = {Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023},
  pages = {10167-10176}
}
```
