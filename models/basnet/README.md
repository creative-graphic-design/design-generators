---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "basnet"
  - "saliency-detection"
  - "content-image"
datasets:
  - "SmartText demo"
model-index:
  - name: "BASNet"
    results:
      - task:
          type: "other"
          name: "Saliency detection"
        dataset:
          type: "SmartText demo"
          name: "SmartText vendor demo assets"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "bit-exact"
            name: "Vendor parity"
---

# Model Card for BASNet

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=1907.10719&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1907.10719)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2019&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=SmartText+demo&color=informational&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package implements BASNet as a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style salient object detection model that predicts foreground saliency maps from RGB images.

## Model Details

### Model Description

BASNet predicts foreground saliency maps from RGB images. The package exposes `BASNetModel`, `BASNetConfig`, and `BASNetImageProcessor` so downstream packages can load a converted checkpoint with `from_pretrained`, preprocess content images, and postprocess saliency tensors at the source image size.

- **Developed by:** Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane, and Martin Jagersand.
- **Shared by:** creative-graphic-design.
- **Model type:** salient object detection.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0 for repository code; checkpoint redistribution depends on the checkpoint source license.

### Model Sources

- **Repository:** [BASNet repository](https://github.com/xuebinqin/BASNet)
- **Paper:** [BASNet: Boundary-Aware Salient Object Detection](https://arxiv.org/abs/1907.10719)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| GDI BASNet | `creative-graphic-design/basnet-gdi` | not-published |

## Uses

### Direct Use

Use this package for saliency inference over RGB content images after converting a local BASNet checkpoint.

```python
import torch
from PIL import Image

from basnet import BASNetImageProcessor, BASNetModel

model = BASNetModel.from_pretrained(".cache/basnet/converted/basnet-gdi")
processor = BASNetImageProcessor.from_pretrained(".cache/basnet/converted/basnet-gdi")

image = Image.open("example.png").convert("RGB")
encoded = processor(image)
with torch.no_grad():
    output = model(encoded["pixel_values"])

saliency = processor.postprocess_saliency(
    output.saliency[0],
    output_size=tuple(encoded["image_sizes"][0].tolist()),
)
```

### Downstream Use

Downstream packages can use the returned saliency tensor as content-image evidence, for example when filtering candidate regions or placing text over images. SmartText uses this package as the shared BASNet implementation behind its saliency preprocessing path.

### Out-of-Scope Use

The package is not intended for general image segmentation benchmarks, training new BASNet checkpoints, or safety-critical publishing automation. It does not publish weights from the GDI checkpoint source.

## Bias, Risks, and Limitations

The converted checkpoint path depends on local assets that are not redistributed here. Saliency maps reflect the visual domain of the checkpoint source and can underrepresent small text, transparent objects, or graphic elements outside the source distribution.

### Recommendations

Run the parity commands before comparing downstream layout results, and inspect saliency maps on the target design domain before using them as placement constraints.

## How to Get Started with the Model

Install the package directly from this repository.

```bash
pip install "basnet @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/basnet"
```

The Hub checkpoint is not published yet. Follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/basnet/REPRODUCING.md) to convert a released checkpoint locally and load the converted directory:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package basnet
# The conversion creates `.cache/basnet/converted/basnet-gdi`.
```

```python
from basnet import BASNetModel

model = BASNetModel.from_pretrained(".cache/basnet/converted/basnet-gdi")

# After Hub publication: from_pretrained("creative-graphic-design/basnet-gdi")
```

## Training Details

### Training Data

Training data is not included in this workspace member. The conversion and parity checks use the SmartText demo assets as a small content-image fixture.

### Training Procedure

Training is not implemented in this workspace member. The package focuses on architecture implementation, checkpoint conversion, saliency preprocessing, and parity inference.

## Evaluation

### Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| BASNet saliency maps | 3 SmartText demo images / 196608 values | Bit-exact, `max_abs_diff=0.0`, `pearson_corr=1.0`, `rtol=0`, `atol=0`, same-device comparison | Passed |
| SmartText consumer path | 3 SmartText demo images / 43 candidates | Bit-exact saliency, scorer inputs, scores, selected boxes, and text color after GPU1 reference generation | Passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/basnet/REPRODUCING.md) to reproduce the original-implementation agreement checks by downloading or staging the released weights, generating references, converting checkpoints, running parity checks, and smoke-testing local loading.

## License

This package code is Apache-2.0. Checkpoint redistribution depends on the checkpoint source license and is not granted by this package.

## Citation

```bibtex
@inproceedings{qin2019basnet,
  title={BASNet: Boundary-Aware Salient Object Detection},
  author={Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar R. and Jagersand, Martin},
  booktitle={CVPR},
  year={2019}
}
```
