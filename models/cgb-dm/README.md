---
library_name: diffusers
pipeline_tag: image-to-layout
license: apache-2.0
tags:
  - layout-generation
  - poster-layout
datasets:
  - creative-graphic-design/PKU-PosterLayout
  - creative-graphic-design/CGL-Dataset
---

# CGB-DM

CGB-DM is a content-aware diffusion transformer for poster layout generation.
This package provides a Diffusers-style pipeline for training-produced
checkpoints and local conversion experiments.

## Usage

Install the workspace package with its shared libraries:

```bash
pip install -e lib/laygen -e lib/posgen -e models/cgb-dm
```

Load a converted checkpoint:

```python
import torch
from cgb_dm import CGBDMPipeline

pipe = CGBDMPipeline.from_pretrained("creative-graphic-design/cgb-dm-pku-posterlayout")
output = pipe(pixel_values=torch.zeros(1, 4, 384, 256), condition_type="content_image")
```

## Checkpoints

| Checkpoint | Dataset | Status |
| --- | --- | --- |
| `creative-graphic-design/cgb-dm-pku-posterlayout` | PKU PosterLayout | planned after training |
| `creative-graphic-design/cgb-dm-cgl` | CGL | planned after training |

## Datasets

PKU PosterLayout and CGL are supported. The original CGB-DM zip remains the
authoritative source for parity staging because it includes inpainted images,
saliency maps, saliency boxes, and CSV row ordering.

## Evaluation

### Parity Results

| Dataset | Stage | Cases | Criterion | Result |
| --- | --- | ---: | --- | --- |
| PKU PosterLayout | S0-S2 synthetic | 0 | gated adapter present | pending assets |
| CGL | S0-S2 synthetic | 0 | gated adapter present | pending assets |

## Reproducibility

The original-implementation agreement checks are reproduced by preparing the
original assets, generating reference metadata, running gated parity tests,
converting a training checkpoint, and smoke-testing `from_pretrained`.

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/download_original_assets.py
CUDA_VISIBLE_DEVICES=0 uv run --package cgb-dm python models/cgb-dm/scripts/generate_reference_outputs.py --dataset pku_posterlayout
PARITY_REQUIRE=1 uv run --package cgb-dm pytest models/cgb-dm/tests/vendor_parity -m vendor_parity
uv run --package cgb-dm python models/cgb-dm/scripts/convert_training_checkpoint.py --checkpoint .cache/cgb-dm/checkpoints/example.ckpt --output-dir .cache/cgb-dm/converted/pku
uv run --package cgb-dm python models/cgb-dm/scripts/smoke_from_pretrained.py --path .cache/cgb-dm/converted/pku
```

## Training

Use the shared class-path LightningCLI entry point:

```bash
uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml
```

## License

The original implementation is Apache-2.0. This package is Apache-2.0.

## Citation

```bibtex
@article{layoutdit2024,
  title = {LayoutDiT: Content-Graphic Balance Layout Generation with Diffusion Transformer},
  year = {2024}
}
```
