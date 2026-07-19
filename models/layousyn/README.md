---
language:
  - en
license: "cc-by-nc-4.0"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "layousyn"
  - "layout-generation"
datasets:
  - "GRIT"
  - "COCO-grounded"
model-index:
  - name: "LayouSyn"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "unknown"
          name: "GRIT"
          config: "grounded captions"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayouSyn

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2505.04718&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.04718)
![license](https://img.shields.io/static/v1?label=license&message=CC--BY--NC--4.0&color=orange&style=flat-square&logo=creativecommons&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=GRIT&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=COCO--grounded&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square)
[![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/creative-graphic-design/layousyn-grit)

This package wraps [LayouSyn](https://arxiv.org/abs/2505.04718), the [Lay-Your-Scene](https://github.com/mlpc-ucsd/Lay-Your-Scene) natural-scene layout method, in a 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index)-style pipeline with normalized public layout outputs.

## Model Details

### Model Description

LayouSyn generates natural-scene object layouts from caption and concept embeddings with a diffusion-transformer denoiser. The package targets GRIT and COCO-grounded variants, taking text/concept conditions and returning object boxes aligned to the requested concept labels. Public outputs use normalized center `xywh` boxes in `[0, 1]`, request-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Dhruv Srivastava et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** cc-by-nc-4.0.

### Model Sources

- **Repository:** [Lay-Your-Scene repository](https://github.com/mlpc-ucsd/Lay-Your-Scene)
- **Paper:** [arXiv 2505.04718](https://arxiv.org/abs/2505.04718)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| GRIT | [`creative-graphic-design/layousyn-grit`](https://huggingface.co/creative-graphic-design/layousyn-grit) | not-published |
| COCO grounded | [`creative-graphic-design/layousyn-coco-grounded`](https://huggingface.co/creative-graphic-design/layousyn-coco-grounded) | not-published |
| GRIT fine-tuned on COCO grounded | [`creative-graphic-design/layousyn-grit-ft-coco-grounded`](https://huggingface.co/creative-graphic-design/layousyn-grit-ft-coco-grounded) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

The public pipeline accepts text/concept conditions with caption and concept embeddings. A tiny constructed pipeline can be smoke-tested without downloaded weights:

```bash
uv run --package layousyn python
```

```python
import torch
from layousyn import LayouSynDiTModel, LayouSynPipeline, LayouSynProcessor, LayouSynScheduler

model = LayouSynDiTModel(
    model_name="DiT-D1-H32-N1",
    max_in_len=2,
    max_y_len=3,
    concept_in_channels=4,
    y_in_channels=4,
)
pipe = LayouSynPipeline(
    model=model,
    scheduler=LayouSynScheduler(num_train_timesteps=100),
    processor=LayouSynProcessor(max_in_len=2, max_y_len=3, concept_in_channels=4, y_in_channels=4),
)
out = pipe(
    labels=[["person", "bench"]],
    caption_embeds=torch.zeros(1, 3, 4),
    caption_padding_mask=torch.tensor([[False, True, True]]),
    concept_embeds=torch.zeros(1, 2, 4),
    num_inference_steps=1,
    guidance_scale=1.0,
    seed=0,
)
print(out.bbox.shape)
```

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layousyn/REPRODUCING.md). Those steps create `.cache/layousyn/converted`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layousyn
uv run --package layousyn python
```

```python
from layousyn import LayouSynPipeline

path = ".cache/layousyn/converted"
# After Hub publication: from_pretrained("creative-graphic-design/layousyn-grit")
pipe = LayouSynPipeline.from_pretrained(path)
out = pipe(batch_size=1, seed=0, num_inference_steps=1)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| GRIT | unknown | grounded captions |
| COCO grounded | unknown | grounded objects |

COCO-17, COCO-Caption-Grounded, GriT, and NSR-1K are not yet available in the `creative-graphic-design` Hugging Face org. Until those imports exist, parity and conversion scripts follow the original repository's dataset/download path.

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

### Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| Alpha-scale beta respacing | 1 | `timestep_map`, helper betas, and scheduler betas exact | pass; max abs 0 |
| Denoiser logits | 1 | exact tensor match | pass; max abs 0 |
| First DDIM scheduler step | 1 | exact `pred_xstart` and `prev_sample` tensor match | pass; max abs 0 / 0 |
| Full 40-step sample public bbox | 1 | exact normalized public `xywh` tensor match | pass; max abs 0 |

The parity path matches the original implementation with TF32 enabled, the vendor `nn.MultiheadAttention` `need_weights=True` code path, and the vendor CPU-to-device timestep-frequency embedding order. Those settings remove the previous denoiser-logit and full-sample drift.

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layousyn/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.


## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayouSyn uses a diffusion transformer over object-layout tokens conditioned on caption and concept embeddings. The scheduler iteratively denoises concept-aligned boxes, and the processor maps generated object positions back to normalized center `xywh` boxes with request-local labels.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layousyn ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original Lay-Your-Scene implementation is CC BY-NC 4.0. Converted checkpoints and model cards must retain the non-commercial notice.

## Citation

```bibtex
@article{srivastava2025layousyn,
  title = {LayouSyn: Compositional Scene Layout Generation via Semantic-Syntax Alignment},
  author = {Srivastava, Dhruv and Tripathi, Shashank and Choudhary, Sarthak and Balasubramanian, Vineeth N.},
  journal = {arXiv preprint arXiv:2505.04718},
  year = {2025}
}
```
