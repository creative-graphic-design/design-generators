---
license: apache-2.0
library_name: transformers
pipeline_tag: image-segmentation
tags:
  - saliency-detection
  - basnet
---

# BASNet

BASNet predicts foreground saliency maps for content-aware layout and poster
generation packages. This package provides a Transformers-style
`BASNetModel`, `BASNetConfig`, and `BASNetImageProcessor` that can be loaded with
`from_pretrained` and reused by downstream model packages.

## Install

```bash
pip install \
  "basnet @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/basnet"
```

## Usage

```python
import torch
from PIL import Image

from basnet import BASNetImageProcessor, BASNetModel

model = BASNetModel.from_pretrained("creative-graphic-design/basnet-gdi")
processor = BASNetImageProcessor.from_pretrained("creative-graphic-design/basnet-gdi")

image = Image.open("example.png").convert("RGB")
encoded = processor(image)
with torch.no_grad():
    output = model(encoded["pixel_values"])

saliency = processor.postprocess_saliency(
    output.saliency[0],
    output_size=tuple(encoded["image_sizes"][0].tolist()),
)
```

## Supported Checkpoints

| Checkpoint | Hub id | Status |
| --- | --- | --- |
| GDI BASNet / xuebinqin BASNet architecture | `creative-graphic-design/basnet-gdi` | not-published |

Weights are not published by this package. Convert a local checkpoint before
using `from_pretrained`:

```bash
uv run --package basnet python models/basnet/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/basnet/original/gdi-basnet.pth \
  --output-dir .cache/basnet/converted/basnet-gdi
```

## Datasets

The package itself is dataset-agnostic. Downstream packages use it for
content-image saliency maps before layout generation.

## Reproducibility

The BASNet agreement check reproduces the SmartText saliency parity path against
the original implementation's PNG-space output resize.

```bash
uv run --package basnet python models/basnet/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/smarttext/original/gdi-basnet.pth \
  --output-dir .cache/basnet/converted/basnet-gdi

uv run --package basnet pytest models/basnet/tests

PARITY_REQUIRE=1 uv run --package smarttext pytest \
  models/smarttext/tests/vendor_parity/test_smarttext_parity.py \
  -m vendor_parity
```

SmartText parity currently reports exact PNG-space tensor agreement for three
demo images when the local SmartText reference artifacts and converted
checkpoint are present.

## License

This package code is Apache-2.0. Checkpoint redistribution depends on the
checkpoint source license and is not granted by this package.

## Citation

```text
@inproceedings{qin2019basnet,
  title = {BASNet: Boundary-Aware Salient Object Detection},
  author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar R. and Jagersand, Martin},
  booktitle = {CVPR},
  year = {2019}
}
```

Original implementation: https://github.com/xuebinqin/BASNet
