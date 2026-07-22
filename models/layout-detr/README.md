---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layout-detr"
  - "layout-generation"
  - "content-image"
  - "ad-banner"
datasets:
  - "Ad Banner vendor distribution"
model-index:
  - name: "LayoutDETR"
    results:
      - task:
          type: "other"
          name: "Content-image layout generation"
        dataset:
          type: "Ad Banner vendor distribution"
          name: "Ad Banner"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "not-run"
            name: "Vendor parity"
---

# Model Card for LayoutDETR

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2212.09877&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2212.09877)
![venue](https://img.shields.io/static/v1?label=venue&message=ECCV+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Ad+Banner+vendor+distribution&color=informational&style=flat-square)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=not--run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports LayoutDETR, the ECCV 2024 content-image ad-banner layout generator, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

LayoutDETR generates normalized center `xywh` foreground text boxes for a background image plus per-element text strings and semantic labels. The public schema returns `bbox`, `labels`, `mask`, and `id2label`; optional `intermediates` carry latents, text rows, and postprocessing metadata.

- **Developed by:** Ning Yu, Chia-Chih Chen, Zeyuan Chen, Rui Meng, Gang Wu, Paul Josel, Juan Carlos Niebles, Caiming Xiong, and Ran Xu.
- **Shared by:** creative-graphic-design.
- **Model type:** content-image layout generation.
- **Language(s) (NLP):** English ad-banner text strings.
- **License:** Apache-2.0 for the original LayoutDETR repository; notices cover the vendor-acknowledged [StyleGAN3](https://github.com/NVlabs/stylegan3), [DETR](https://github.com/facebookresearch/detr), [Up-DETR](https://github.com/dddzg/up-detr), [BLIP/BERT](https://github.com/salesforce/BLIP), [LayoutGAN++](https://github.com/ktrk115/const_layout), [Pitt Image Ads](https://people.cs.pitt.edu/~kovashka/ads/), and [LaMa](https://github.com/advimman/lama) components.

### Model Sources

- **Repository:** [LayoutDETR repository](https://github.com/salesforce/LayoutDETR)
- **Paper:** [LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](https://arxiv.org/abs/2212.09877)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| LayoutDETR Ad Banner `G_ema` pickle | `creative-graphic-design/layout-detr-ad-banner` | not-published |

## Uses

### Direct Use

Use this package for research inference and local conversion checks after converting the released Ad Banner pickle.

```python
from PIL import Image
from layout_detr import LayoutDetrPipeline

pipe = LayoutDetrPipeline.from_pretrained(
    "creative-graphic-design/layout-detr-ad-banner",
)
output = pipe(
    Image.open("background.jpg").convert("RGB"),
    texts=["EVERYTHING 10% OFF", "SHOP NOW"],
    labels=["header", "button"],
    condition_type="content_image",
)
```

### Downstream Use

Downstream design tools can render returned boxes on the source background, compare generated layouts, or use `latents` in `intermediates` for deterministic replay.

### Out-of-Scope Use

The package is not intended for text-only layout generation, document layout synthesis, or production ad publishing without human review. LayoutDETR supports only `condition_type="content_image"` because the released generator is background-image conditioned and requires per-element text plus labels.

## Bias, Risks, and Limitations

The Ad Banner data remains on the vendor Google Drive distribution until an org-hosted dataset exists. Ordinary tests use synthetic images and local fixtures, so they do not measure model quality.

The converted runtime avoids StyleGAN CUDA custom ops. Extracting `G_ema` from the original pickle is a conversion-time vendor operation and records whether `torch_utils.ops` was imported while unpickling.

### Recommendations

Run the parity commands on one selected GPU before comparing research numbers, and visually inspect generated boxes when using new brands, image styles, or text lengths.

## How to Get Started with the Model

The Hub checkpoint is not published yet. Follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-detr/REPRODUCING.md) to convert the released weights locally and load the converted directory:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-detr
# The conversion creates `.cache/layout-detr/converted/layout-detr-ad-banner`.
```

```python
from layout_detr import LayoutDetrPipeline

pipe = LayoutDetrPipeline.from_pretrained(
    ".cache/layout-detr/converted/layout-detr-ad-banner",
    local_files_only=True,
)
# After Hub publication: from_pretrained("creative-graphic-design/layout-detr-ad-banner")
```

## Training Details

### Training Data

The released checkpoint was trained on the original Ad Banner vendor distribution, which contains 7,672 samples according to the vendor README. The data path is isolated behind the package processor and dataset adapter, with a TODO to switch to an org dataset when available.

### Training Procedure

Training follows the original LayoutDETR GAN/DETR objective and vendor environment. This package focuses on conversion and inference; training code is not added.

## Evaluation

### Parity Results

| Compared target | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| Released Ad Banner pickle unpickle | 1 checkpoint | extract `G_ema` and record conversion report | passes with conversion-time Transformers compatibility shims; `torch_utils.ops` was imported |
| Strict converted state load | 1 checkpoint | all remapped tensors strict-load into `LayoutDetrForConditionalGeneration` | fails: 852 source keys, 41 target keys, 6 loaded keys, 34 missing keys, 845 unexpected keys, and `fc_z.weight` shape `[768, 36]` vs `[768, 4]` |
| Vendor `G_ema` forward reference | 0 | bitwise equality on `bbox_fake` | not run because strict checkpoint conversion failed before a comparable converted forward path existed |
| Converted `from_pretrained` smoke | 1 synthetic case | schema and local load | passes in ordinary tests |
| Custom-op boundary | 0 | converted runtime imports no `torch_utils.ops` | documented by parity test hook |

## Reproducibility

Reproduce the original-implementation agreement checks by following [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-detr/REPRODUCING.md).

## License

The original LayoutDETR code is Apache-2.0. The vendor README acknowledges StyleGAN3, DETR, Up-DETR, BLIP, LayoutGAN++, Pitt Image Ads, and LaMa components; keep those notices with any redistributed converted checkpoint.

## Citation

```bibtex
@inproceedings{yu2024layoutdetr,
  title={LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer},
  author={Yu, Ning and Chen, Chia-Chih and Chen, Zeyuan and Meng, Rui and Wu, Gang and Josel, Paul and Niebles, Juan Carlos and Xiong, Caiming and Xu, Ran},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
