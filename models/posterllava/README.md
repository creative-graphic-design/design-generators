---
license: other
license_name: upstream-license-review-needed
library_name: transformers
pipeline_tag: other
tags:
  - layout-generation
  - poster-layout
  - llava
datasets:
  - Ad Banner
  - CGL
  - PosterLayout
  - QB-Poster
language:
  - en
model-index:
  - name: PosterLLaVA
    results:
      - task:
          type: other
          name: Content-image layout generation
        dataset:
          type: QB-Poster original distribution
          name: QB-Poster
          split: vendor parity fixture
        metrics:
          - type: vendor-parity
            value: not-run
            name: Vendor parity
---

# Model Card for PosterLLaVA

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2406.02884&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2406.02884)
![venue](https://img.shields.io/static/v1?label=venue&message=IEEE+TMM+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=Ad+Banner&color=informational&style=flat-square)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PosterLayout&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=QB-Poster&color=informational&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=cpu-contract&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n/a&color=lightgrey&style=flat-square)

This package contains a PosterLLaVA processor and inference pipeline for parsing generated poster-layout JSON into normalized center `xywh` boxes, integer labels, valid-element masks, and `id2label`.

## Model Details

### Model Description

PosterLLaVA is a LLaVA-style multimodal recipe for generating poster layout JSON from a background image and layout instructions. The public output uses the shared layout schema with normalized center `xywh` boxes.

### Model Sources

- Paper: [PosterLLaVA](https://arxiv.org/abs/2406.02884)
- Original implementation: [PosterLLaVA source repository](https://github.com/PosterLLaVA/PosterLLaVA)
- Upstream checkpoint: `posterllava/posterllava_v0`

## Supported Checkpoints

| Checkpoint | Status | Notes |
| --- | --- | --- |
| `posterllava/posterllava_v0` | local loading supported | Upstream LLaVA-style checkpoint. |
| `creative-graphic-design/posterllava-v0` | blocked | Org redistribution is blocked pending license review. |

## Uses

### Direct Use

Use this package locally with an upstream PosterLLaVA checkpoint to generate or parse poster layout JSON from a background image and requested element count.

### Downstream Use

The normalized boxes, labels, and masks can feed poster-design tooling, layout evaluation, or controlled rendering pipelines that perform their own visual and licensing review.

### Out-of-Scope Use

Do not use this package as an image renderer, OCR model, accessibility verifier, or unreviewed production design system. The model emits layout structure only and can produce overlapping or implausible boxes.

## Bias, Risks, and Limitations

The Hugging Face metadata for `posterllava/posterllava_v0` advertises Apache-2.0, while the original implementation license and usage note are CC-BY-NC-4.0 and non-commercial. Until maintainers resolve that discrepancy, this package does not upload or redistribute weights under the `creative-graphic-design` org.

### Recommendations

Review generated layouts before downstream use, validate boxes against application constraints, and run domain-specific evaluation on the target poster data.

## How to Get Started with the Model

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "posterllava @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/posterllava"
```

```python
from PIL import Image

from posterllava import PosterLlavaConfig, PosterLlavaPipeline, PosterLlavaProcessor

config = PosterLlavaConfig(dataset_name="ad_banner")
processor = PosterLlavaProcessor.from_config(
    dataset_name=config.dataset_name,
    id2label=config.id2label,
    prompt_template=config.prompt_template,
)
pipe = PosterLlavaPipeline.from_pretrained(
    "posterllava/posterllava_v0",
    config=config,
    components={"processor": processor},
)

image = Image.open("poster-background.png").convert("RGB")
output = pipe(images=image, num_elements=5, do_sample=False)
```

## Training Details

### Training Data

The released checkpoint documentation names Ad Banner, CGL, PosterLayout, and QB-Poster. Ad Banner and QB-Poster are not yet mirrored in the `creative-graphic-design` Hugging Face org, so local reproduction uses the original distribution paths.

### Training Procedure

The package does not train or fine-tune PosterLLaVA. It provides local processor, parsing, and inference components for the released checkpoint.

## Evaluation

### Parity Results

| Check | Cases | Match Criterion | Result |
| --- | ---: | --- | --- |
| Prompt bytes | 2 | Exact match against original `conv_templates` output, including `<image>` placement | Pass in vendor-parity CPU comparison |
| Prompt token ids | 1 | Exact `IMAGE_TOKEN_INDEX=-200` insertion against original `tokenizer_image_token` | Pass in vendor-parity CPU comparison |
| JSON parser | 3 | Exact parsed objects against original `cli_multi.py` JSON-slice behavior on supported outputs | Pass in vendor-parity CPU comparison |
| Image preprocessing | 1 | Exact tensor match against original square-pad CLIP preprocessing | Pass in vendor-parity CPU comparison |
| Full 13B generation | 513 | Deterministic original-code run on QB-Poster validation with `seed=0`, TF32 disabled, and greedy decoding; 492 layouts parsed and 21 samples matched the original parser's empty-output behavior | Pass in gated vendor-parity validation |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/posterllava/REPRODUCING.md) for the commands that reproduce prompt, parser, image-preprocessing, gated full-generation parity, and local smoke checks.

## Citation

```bibtex
@misc{yang2024posterllava,
  title = {PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM},
  author = {Tao Yang and Yingmin Luo and Zhongang Qi and Yang Wu and Ying Shan and Chang Wen Chen},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2406.02884}
}
```
