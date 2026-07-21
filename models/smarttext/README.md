---
library_name: transformers
pipeline_tag: other
license: other
license_name: upstream-license-review-needed
tags:
  - smarttext
  - text-placement
  - poster-generation
  - content-image
  - layout-generation
---

# SmartText

SmartText places text on natural images with a saliency model, candidate-region generation, and a candidate scorer. This package exposes vendor-compatible `PreTrainedModel` ports of `BASNet(3, 1)` and the multi-scale ShuffleNetV2 SMT scorer, orchestrated by `SmartTextPipeline`.

## Usage

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

The output follows the shared layout schema: normalized center `xywh` `bbox`, integer `labels`, boolean `mask`, `id2label={0: "text"}`, optional `scores`, and SmartText-specific artifacts in `intermediates`.

## Supported Checkpoints

| Hub id | Contents | Status |
| --- | --- | --- |
| `creative-graphic-design/smarttext-smt` | BASNet/GDI saliency model, SMT scorer, processor, and pipeline config | Planned; Hub weight publishing is deferred to #78 |

## Data

The ordinary tests use synthetic images and optional vendor demo assets. The original SmartText training data is not hosted by `creative-graphic-design`, so this package does not add a shared dataset dependency.

## Preprocessing

`SmartTextImageProcessor` keeps the two original image paths separate: BASNet uses the original 256-pixel saliency input, while the scorer uses the original SmartText short-side resize rule, RGB values divided by `256.0`, and ImageNet mean/std normalization.

## Limitations

SmartText only supports `condition_type="content_image"`. Text-only generation is unsupported because the method needs image saliency or candidate boxes to place text. The RoI/RoD alignment modules are PyTorch ports of the vendor forward kernels. The original compiled extensions do not build against the current PyTorch extension API, so the reference harness imports vendor `smtModel.py` with the same PyTorch RoI/RoD shim instead of editing `vendor/`.

The shim follows the vendor CUDA kernel formulas directly: RoIAlign samples the scaled box lattice with `+1` width/height and bilinear interpolation, RoDAlign samples the whole feature lattice with `(height - 1.001)/(aligned_height - 1)` and zeroes samples inside the scaled box, and both `Avg` modules apply the vendor 2x2 stride-1 average pooling. Equivalence to the unavailable compiled extension is therefore source-level rather than binary-verified.

## Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| BASNet saliency maps | 3 vendor demo images | Exact PNG-space tensor match against vendor BASNet preprocessing and `save_output` resize path | Passed |
| Candidate JSON | 3 vendor demo images / 43 candidates | Exact JSON match against vendor `gen_boxes_multi` | Passed |
| Scorer inputs | 43 candidates | Exact tensor match against vendor `setup_test_dataset` RoE preprocessing | Passed |
| Raw SMT scores | 43 candidates | Bit-exact with `torch.use_deterministic_algorithms(True)`, TF32 disabled, and cuDNN disabled | Passed |
| Selected boxes | 3 images, top 3 each | Exact selected candidate indices | Passed |

## Reproducibility

The commands in `REPRODUCING.md` reproduce the original-implementation agreement checks by downloading the released weights, generating vendor references on one GPU, converting checkpoints, and running the gated parity tests. cuDNN is disabled for parity because the first divergent op with cuDNN enabled is the scorer's first ShuffleNetV2 convolution (`Feat_ext.feature3.0.0`), which differs by a few float32 ULPs across processes even with deterministic algorithms enabled.

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

Original implementation: `vendor/smarttext`.
