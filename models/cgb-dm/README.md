---
language:
  - en
license: "apache-2.0"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "cgb-dm"
  - "layoutdit"
  - "poster-layout"
  - "layout-generation"
  - "content-aware-layout"
datasets:
  - "creative-graphic-design/PKU-PosterLayout"
  - "creative-graphic-design/CGL-Dataset"
model-index:
  - name: "CGB-DM"
    results:
      - task:
          type: "other"
          name: "Content-aware poster layout generation"
        dataset:
          type: "creative-graphic-design/PKU-PosterLayout"
          name: "PKU PosterLayout"
          split: "test"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for CGB-DM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2407.15233&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.15233)
![venue](https://img.shields.io/static/v1?label=venue&message=arXiv+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PKU&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports CGB-DM, the content-aware poster layout diffusion model from LayoutDiT, into a [`🧨 diffusers`](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

CGB-DM generates poster layouts from content images, saliency information, and optional layout constraints. The package exposes `CGBDMTransformerModel`, `CGBDMProcessor`, `CGBDMScheduler`, and `CGBDMPipeline`; public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Yu Li, Yifan Chen, Gongye Liu, Fei Yin, Qingyan Bai, Jie Wu, Hongfa Wang, Ruihang Chu, and Yujiu Yang.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware poster layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0 for the original implementation and this package.

### Model Sources

- **Repository:** [LayoutDiT repository](https://github.com/yuli0103/LayoutDiT)
- **Paper:** [CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Model](https://arxiv.org/abs/2407.15233)
- **Training assets:** [CGB-DM asset release page](https://github.com/yuli0103/CGB-DM)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PKU PosterLayout CGB-DM | `creative-graphic-design/cgb-dm-pku-posterlayout` | planned after training |
| CGL CGB-DM | `creative-graphic-design/cgb-dm-cgl` | planned after training |

## Uses

### Direct Use

Use this package for research conversion checks, local training experiments, and content-aware poster layout generation after checkpoints have been trained or converted.

```python
import torch
from cgb_dm import CGBDMPipeline

pipe = CGBDMPipeline.from_pretrained("creative-graphic-design/cgb-dm-pku-posterlayout")
out = pipe(pixel_values=torch.zeros(1, 4, 384, 256), condition_type="content_image")
print(out.bbox, out.labels, out.mask)
```

Install the package with its shared libraries:

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "cgb-dm @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/cgb-dm"
```

### Downstream Use

Generated layouts may support poster composition tools, design evaluation, layout reranking, and controlled layout editing workflows after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared visual designs without separate review.

## Bias, Risks, and Limitations

CGB-DM behavior depends on the upstream data preparation path, including inpainted poster images, saliency maps, saliency boxes, CSV row ordering, and dataset label conventions. The first package version is train-first because released generator checkpoints were not available during implementation.

### Recommendations

Run the gated parity workflow before publishing trained checkpoints or comparing generated metrics. Review generated layouts separately for each poster domain and dataset split.

## How to Get Started with the Model

Install the package with its shared libraries:

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "cgb-dm @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/cgb-dm"
```

Clone this repository, install the workspace member, and run the training or conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/cgb-dm/REPRODUCING.md). Those steps create converted pipeline directories under `.cache/cgb-dm/converted/`.

## Training Details

### Training Data

PKU PosterLayout and CGL are supported through the original CGB-DM asset structure. The original asset zip remains authoritative for parity because it includes the image, saliency, saliency-box, and CSV ordering needed to mirror the upstream training loop.

### Training Procedure

Training uses the shared class-path [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) entry point with package-local YAML configs. Optimizer and scheduler defaults live in `models/cgb-dm/configs/training/*.yaml`.

```bash
uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml
```

## Evaluation

### Parity Results

The full vendor parity suite is not run unless local CGB-DM assets are present and `PARITY_REQUIRE=1` is set.

| Dataset | Stage | Cases | Criterion | Result |
| --- | --- | ---: | --- | --- |
| PKU PosterLayout | real vendor S0-S2 training-step parity | 1 fixed batch | S0 exact loader replay; S1 trace exact for integer/mask tensors and `atol=1e-7, rtol=1e-5` for CUDA floating tensors; S2 gradients `atol=1e-9, rtol=1e-5`, post-Adam parameters/state `atol=5e-7, rtol=2e-3` | passes locally with `PARITY_REQUIRE=1` |
| CGL | synthetic training-step parity | 0 | gated adapter present | not run |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/cgb-dm/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## License

The original implementation is Apache-2.0. This package is Apache-2.0.

## Citation

```bibtex
@article{li2024cgbdm,
  title = {CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Model},
  author = {Li, Yu and Chen, Yifan and Liu, Gongye and Yin, Fei and Bai, Qingyan and Wu, Jie and Wang, Hongfa and Chu, Ruihang and Yang, Yujiu},
  year = {2024},
  url = {https://arxiv.org/abs/2407.15233}
}
```
