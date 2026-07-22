---
language:
  - en
license: "other"
license_name: "upstream-license-review-needed"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "smarttext"
  - "text-placement"
  - "poster-generation"
  - "content-image"
  - "layout-generation"
datasets:
  - "SmartText demo"
model-index:
  - name: "SmartText"
    results:
      - task:
          type: "other"
          name: "Content-aware text placement"
        dataset:
          type: "SmartText demo"
          name: "SmartText vendor demo assets"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "bit-exact"
            name: "Vendor parity"
---

# Model Card for SmartText

[![paper](https://img.shields.io/static/v1?label=paper&message=TMM+2021&color=blue&style=flat-square)](https://ieeexplore.ieee.org/document/9520053)
![venue](https://img.shields.io/static/v1?label=venue&message=TMM+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=SmartText+demo&color=informational&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports SmartText, the TMM 2021 content-aware text placement method, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package with vendor-compatible `PreTrainedModel` components and a `laygen.pipelines.LayoutGenerationPipeline` subclass.

## Model Details

### Model Description

SmartText places text on natural images with a BASNet/GDI saliency model, deterministic candidate-region generation, and the multi-scale ShuffleNetV2 SMT scorer. Public outputs use normalized center `xywh` boxes in `[0, 1]`, integer text labels, a valid-element `mask`, `id2label={0: "text"}`, optional `scores`, and SmartText-specific saliency and candidate artifacts in `intermediates`.

- **Developed by:** Chenhui Li et al.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware text placement.
- **Language(s) (NLP):** not applicable.
- **License:** upstream license review needed before publishing converted weights.

### Model Sources

- **Repository:** [SmartText repository](https://github.com/chenqi008/SmartText)
- **Paper:** [Harmonious Textual Layout Generation over Natural Images via Deep Aesthetics Learning](https://ieeexplore.ieee.org/document/9520053)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| SMT released scorer plus GDI BASNet | `creative-graphic-design/smarttext-smt` | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of SmartText text placement over natural images.

Conditional generation accepts an RGB content image and prompt text:

```python
from PIL import Image
from smarttext import SmartTextPipeline

pipe = SmartTextPipeline.from_pretrained(
    "creative-graphic-design/smarttext-smt",
)
output = pipe(
    Image.open("image.jpg").convert("RGB"),
    prompt="ICME 2020\n6-10 July, London",
    condition_type="content_image",
)
```

### Downstream Use

Downstream systems can use the returned `bbox`, `labels`, `mask`, and `scores` to place a rendered text block or to compare candidate text placements for design automation research.

### Out-of-Scope Use

The package is not intended for text-only layout generation, document page layout synthesis, or safety-critical automated publishing workflows. SmartText only supports `condition_type="content_image"` because the method needs image saliency or candidate boxes to place text.

## Bias, Risks, and Limitations

The original SmartText training data is not distributed in this repository, and converted-weight redistribution needs upstream license review. The ordinary tests use synthetic images and optional vendor demo assets.

The RoI/RoD alignment modules are PyTorch ports of the vendor forward kernels. The original compiled extensions do not build against the current PyTorch extension API, so the reference harness imports vendor `smtModel.py` with the same PyTorch RoI/RoD shim instead of editing `vendor/`.

The shim follows the vendor CUDA kernel formulas directly: RoIAlign samples the scaled box lattice with `+1` width/height and bilinear interpolation, RoDAlign samples the whole feature lattice with `(height - 1.001)/(aligned_height - 1)` and zeroes samples inside the scaled box, and both `Avg` modules apply the vendor 2x2 stride-1 average pooling. Equivalence to the unavailable compiled extension is source-level rather than binary-verified.

### Recommendations

Use the parity commands before comparing research results, and inspect generated boxes visually when moving beyond the vendor demo images.

## How to Get Started with the Model

The Hub checkpoint is not published yet. Follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/smarttext/REPRODUCING.md) to convert the released weights locally and load the converted directory:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package smarttext
# The conversion creates `.cache/smarttext/converted/smarttext-smt`.
```

```python
from smarttext import SmartTextPipeline

pipe = SmartTextPipeline.from_pretrained(
    ".cache/smarttext/converted/smarttext-smt",
)

# After Hub publication: from_pretrained("creative-graphic-design/smarttext-smt")
```

## Training Details

### Training Data

The original SmartText training data is not included here. This package documents and verifies conversion of the released SMT scorer and GDI BASNet checkpoints.

### Training Procedure

Training is not implemented in this workspace member. The package focuses on architecture porting, checkpoint conversion, and vendor-parity inference.

## Evaluation

### Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| BASNet saliency maps | 3 vendor demo images | Exact PNG-space tensor match against vendor BASNet preprocessing and `save_output` resize path | Passed |
| Candidate JSON | 3 vendor demo images / 43 candidates | Exact JSON match against vendor `gen_boxes_multi` | Passed |
| Scorer inputs | 43 candidates | Exact tensor match against vendor `setup_test_dataset` RoE preprocessing | Passed |
| Raw SMT scores | 43 candidates | Bit-exact with `torch.use_deterministic_algorithms(True)`, TF32 disabled, and cuDNN disabled | Passed |
| Selected boxes | 3 images, top 3 each | Exact selected candidate indices | Passed |
| Text color | 3 selected top boxes | Exact hex color match against vendor `cal_best_color` with fixed KMeans seed and threadpool limit | Passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/smarttext/REPRODUCING.md) to reproduce the original-implementation agreement checks by downloading the released weights, generating vendor references on one GPU, converting checkpoints, and running the gated parity tests. cuDNN is disabled for parity because the first divergent op with cuDNN enabled is the scorer's first ShuffleNetV2 convolution (`Feat_ext.feature3.0.0`), which differs by a few float32 ULPs across processes even with deterministic algorithms enabled.

## License

The wrapper code follows this repository's license. Upstream SmartText code and converted-weight redistribution need review because the visible MIT file in `vendor/smarttext/LICENSE` names SSD authors rather than the SmartText authors.

## Citation

```bibtex
@article{li2021smarttext,
  title={Harmonious Textual Layout Generation over Natural Images via Deep Aesthetics Learning},
  author={Li, Chenhui and others},
  journal={IEEE Transactions on Multimedia},
  year={2021}
}
```
