---
language:
  - en
license: "other"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "radm"
  - "layout-generation"
  - "poster-layout"
datasets:
  - "creative-graphic-design/CGL-Dataset"
model-index:
  - name: "RADM"
    results:
      - task:
          type: "other"
          name: "Content-aware poster layout generation"
        dataset:
          type: "creative-graphic-design/CGL-Dataset"
          name: "CGL"
          split: "local reference fixture"
        metrics:
          - type: "vendor-parity"
            value: "not run"
            name: "Vendor parity"
---

# Model Card for RADM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2306.09086&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.09086)
![venue](https://img.shields.io/static/v1?label=venue&message=CIKM+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [RADM](https://arxiv.org/abs/2306.09086), the CIKM 2023 relation-aware diffusion method for controllable poster layout generation, into a [`🧨diffusers`](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

RADM is a proposal-box diffusion pipeline for content-aware poster layout generation. The public pipeline accepts a content image plus optional RADM text-feature tensors, runs reverse diffusion over normalized proposal boxes, and returns normalized center `xywh` boxes, dataset-local labels, a valid-element `mask`, `id2label`, scores, and optional denoising trajectory metadata.

- **Developed by:** RADM authors.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware poster layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** unconfirmed for the checked original source and any user-supplied or converted weights.

### Model Sources

- **Repository:** [RADM repository](https://github.com/JD-GenX/RADM)
- **Paper:** [arXiv 2306.09086](https://arxiv.org/abs/2306.09086)
- **Original checkpoint host:** the checked RADM README publishes dataset and testing-feature assets only; no released checkpoint is published.

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| CGL | `creative-graphic-design/radm-cgl` | not-published |
| CGL-v2 | `creative-graphic-design/radm-cgl-v2` | not-published |

## Uses

### Direct Use

Use this package for local RADM conversion experiments, CPU smoke tests, and gated original-code parity preparation after a user-supplied local checkpoint and license terms have been confirmed.

The first public condition is `content_image`. `images` or `content["image"]` is required; `text_features` and `text_mask` are auxiliary payload fields under the same content-image condition.

### Downstream Use

Generated poster layouts may feed research evaluation, poster composition studies, or downstream rendering systems after dataset, license, and quality review.

### Out-of-Scope Use

Do not use synthetic RADM smoke artifacts as release checkpoints, claim original-code agreement without generated references, or redistribute converted weights before local checkpoint provenance and license status are resolved.

## Bias, Risks, and Limitations

The checked RADM README publishes dataset and testing-feature assets only; no public RADM checkpoint is published. The checked source has no license file, so weight redistribution is blocked until maintainers verify the source license, checkpoint provenance, and dataset terms.

### Recommendations

Use explicit local paths for user-supplied checkpoints, datasets, text features, and vendor source. Run the gated parity workflow before comparing generated layouts against the original method.

## How to Get Started with the Model

Install the package directly from this repository. The command includes shared packages when they are not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "radm @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/radm"
```

Clone this repository, install the workspace member, and run the conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/impl-radm/models/radm/REPRODUCING.md). Those steps create `.cache/radm/converted/cgl`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package radm
uv run --package radm python
```

```python
from PIL import Image
from radm import RADMPipeline

path = ".cache/radm/converted/cgl"
# After Hub publication: from_pretrained("creative-graphic-design/radm-cgl")
pipe = RADMPipeline.from_pretrained(path)
out = pipe(Image.new("RGB", (512, 768)), seed=0)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

The package-local adapter accepts explicit CGL and CGL-v2 COCO-style annotation,
image, and precomputed 768-dimensional text-feature paths. It does not download
data. Data provenance and the CGL-v2 license remain unresolved; see
[TRAINING.md](TRAINING.md) for the current gate status.

### Training Procedure

Phase 1 ships the member-scoped LightningCLI entry surface and captured
effective recipe under `configs/training`. The checked original recipe uses one
GPU, batch size 16, AdamW at `2.5e-5`, weight decay `1e-4`, warmup plus
milestones at 150k and 220k, and `MAX_ITER=250000`. The model predicts four
classes while the five-entry CGL vocabulary is preserved explicitly in the
captured class mapping. Source-generated S0-S2 evidence is accepted; the
two-layer S3 record separates synchronized graph/operation checks from the
natural trajectory drift envelope. It is not authoritative loader,
checkpoint-round-trip, or production-wiring evidence, and no S4-S5 or 300-step
lockstep claim is made. See [TRAINING.md](TRAINING.md) for the staged protocol
and remaining gates.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity requires local original-code outputs generated from a user-supplied checkpoint, CGL dataset root, text-feature root, and one selected GPU. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by checkpoint hash, config hash, seed, timestep sequence, proposal noise, denoiser logits, denoised boxes, post-threshold boxes, labels, scores, and NMS indices.

#### Metrics

Metrics are exact scheduler/timestep agreement, exact tensor equality where the converted and original paths share operations, and explicit tolerance only after dtype, TF32, operation-order, and Detectron2 versus torchvision differences are diagnosed.

### Parity Results

| Dataset | Compared path | Cases | Assertion |
| --- | --- | ---: | --- |
| CGL | original Detectron2 RADM path vs. converted `🧨diffusers` path | 0 | not run; no released checkpoint or local parity assets were available |
| Synthetic smoke | randomly initialized RADM pipeline save/load | 1 | schema smoke only, no original-code parity claim |

No accepted original-code inference parity number is available yet. The package includes a gated parity harness that fails under `PARITY_REQUIRE=1` when required local assets are absent.

## Reproducibility

Reproduce original-implementation agreement by supplying an approved local
checkpoint and data assets, capturing the original Detectron2 reference state,
running the gated parity tests, converting the checkpoint, and smoke-testing
`from_pretrained` as described in
[REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/impl-radm/models/radm/REPRODUCING.md).
Training-stage status and the future member-scoped training commands are in
[TRAINING.md](TRAINING.md).

## Environmental Impact

No model weights are included. Conversion, reference generation, and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

RADM samples proposal boxes and denoises them with relation-aware image and text conditioning. The package-level runtime keeps the image/text feature processor, proposal scheduler, denoiser, and postprocessing separate so converted artifacts can load through `🧨diffusers` component folders.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the original Detectron2 path is available.

#### Hardware

CPU is sufficient for import, tiny random-weight smoke tests, and config round trips. CUDA is required for heavyweight original-code parity.

#### Software

Use `uv run --package radm ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The checked original RADM source has no license file, and weight redistribution is blocked until license and checkpoint provenance are confirmed.

## Citation

```bibtex
@inproceedings{zhang2023radm,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Zhang, Yiheng and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  year={2023}
}
```
