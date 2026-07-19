---
language:
  - en
license: "cc-by-nc-4.0"
library_name: "diffusers"
pipeline_tag: "text-to-image"
tags:
  - "layousyn"
  - "layout-generation"
model-index:
  - name: "LayouSyn"
    results:
      - task:
          type: "text-to-image"
          name: "Layout generation"
        dataset:
          type: "not recorded"
          name: "GRIT"
          config: "grounded captions"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayouSyn

![paper](https://img.shields.io/static/v1?label=paper&message=LayouSyn&color=blue&style=flat-square)
![venue](https://img.shields.io/static/v1?label=venue&message=review-needed&color=yellow&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=CC--BY--NC--4.0&color=orange&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=GRIT&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=COCO--grounded&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=tolerance--verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square)

LayouSyn wraps the released grounded layout synthesis checkpoints in a [Diffusers](https://huggingface.co/docs/diffusers/index)-style pipeline with normalized public layout outputs.

## Model Details

### Model Description

LayouSyn is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `diffusers`.

- **Developed by:** original upstream authors; see Model Sources.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** cc-by-nc-4.0.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** not recorded in the current README

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| GRIT | [`creative-graphic-design/layousyn-grit`](https://huggingface.co/creative-graphic-design/layousyn-grit) | not-published |
| COCO grounded | [`creative-graphic-design/layousyn-coco-grounded`](https://huggingface.co/creative-graphic-design/layousyn-coco-grounded) | not-published |
| GRIT fine-tuned on COCO grounded | [`creative-graphic-design/layousyn-grit-ft-coco-grounded`](https://huggingface.co/creative-graphic-design/layousyn-grit-ft-coco-grounded) | not-published |

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
uv sync --package layousyn
```

```python
from layousyn import LayouSynPipeline

pipe = LayouSynPipeline.from_pretrained(
    "creative-graphic-design/layousyn-grit",
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
| GRIT | not recorded | grounded captions |
| COCO grounded | not recorded | grounded objects |

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
| --- | ---: | --- | --- |
| Alpha-scale beta respacing | 1 | `timestep_map`, helper betas, and scheduler betas exact | pass; max abs 0 |
| Denoiser logits | 1 | exact tensor match | pass; max abs 0 |
| First DDIM scheduler step | 1 | exact `pred_xstart` and `prev_sample` tensor match | pass; max abs 0 / 0 |
| Full 40-step sample public bbox | 1 | exact normalized public `xywh` tensor match | pass; max abs 0 |

## Reproducibility

Reproduce original-implementation agreement by generating CUDA fixed-seed vendor
references with real T5 and SentenceTransformer embeddings, converting the same
checkpoint, and running the marked parity tests against those external assets.

```bash
SNAPSHOT="$(uv run --package layousyn python models/layousyn/scripts/download_original.py)"
CKPT="${SNAPSHOT}/grit/model.pt"
CONFIG="${SNAPSHOT}/grit/config.json"

CUDA_VISIBLE_DEVICES=5 uv run --package layousyn --extra vendor python models/layousyn/scripts/save_reference_outputs.py \
  --vendor-root vendor/lay-your-scene \
  --ckpt "${CKPT}" \
  --ckpt-config "${CONFIG}" \
  --output-dir .cache/layousyn/reference \
  --seed 0 \
  --caption "a person sitting on a bench" \
  --concept person \
  --concept bench \
  --cfg-scale 2.0 \
  --aspect-ratio 1.0 \
  --num-sampling-steps 40

uv run --package layousyn python models/layousyn/scripts/convert_checkpoint.py \
  --checkpoint-path "${CKPT}" \
  --config-path "${CONFIG}" \
  --output-dir .cache/layousyn/converted \
  --variant-name grit

CUDA_VISIBLE_DEVICES=5 \
LAYOUSYN_REFERENCE_DIR=.cache/layousyn/reference \
LAYOUSYN_CONVERTED_DIR=.cache/layousyn/converted \
uv run --package layousyn pytest models/layousyn/tests -m vendor_parity -q

CUDA_VISIBLE_DEVICES=5 uv run --package layousyn python - <<'PY'
import torch
from layousyn import LayouSynPipeline

inputs = torch.load(".cache/layousyn/reference/inputs.pt", map_location="cuda")
pipe = LayouSynPipeline.from_pretrained(".cache/layousyn/converted").to("cuda")
pipe.set_progress_bar_config(disable=True)
out = pipe(
    prompt="a person sitting on a bench",
    labels=[["person", "bench"]],
    caption_embeds=inputs["pipeline_caption_embeds"],
    caption_padding_mask=inputs["pipeline_caption_padding_mask"],
    concept_embeds=inputs["pipeline_concept_embeds"],
    aspect_ratio=inputs["pipeline_aspect_ratio"],
    num_inference_steps=1,
    guidance_scale=2.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
print(out.bbox.shape, out.id2label)
PY
```

The CUDA 5 Grit checkpoint parity run for this PR passes 4/4 marked checks:
alpha-scale beta respacing, denoiser logits, first scheduler step, and full
sample public layout output. The `from_pretrained` smoke prints
`torch.Size([1, 60, 4]) {0: 'person', 1: 'bench'}`.

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

Use `uv run --package layousyn ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. Upstream code, checkpoints, datasets, and prompt provider outputs keep their original licenses and terms. Recorded upstream license status: cc-by-nc-4.0.

## Citation

Cite the original method when publishing results. Citation metadata is preserved from the original README when available; otherwise it is not recorded in this README.
