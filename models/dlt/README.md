---
language:
  - en
license: "apache-2.0"
library_name: "diffusers"
pipeline_tag: "other"
tags:
  - "dlt"
  - "layout-generation"
datasets:
  - "creative-graphic-design/PubLayNet"
  - "RICO13"
  - "creative-graphic-design/magazine"
model-index:
  - name: "DLT"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/PubLayNet"
          name: "PubLayNet"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for DLT

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.03755&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.03755)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO13&color=informational&style=flat-square)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Magazine&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/magazine)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [DLT](https://arxiv.org/abs/2303.03755), the ICCV 2023 joint continuous and discrete diffusion model for content-agnostic layout generation, into a [`🧨diffusers`](https://huggingface.co/docs/diffusers/index)-style package.

## Model Details

### Model Description

DLT denoises bounding boxes and category tokens with coupled continuous and discrete diffusion. The public pipeline returns normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`. Supported condition types are `unconditional`, `label`, and `label_size`; original aliases `all`, `whole_box`, and `loc` normalize to those canonical names.

- **Developed by:** Elad Levi, Eli Brosh, Mykola Mykhailych, and Meir Perez.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [DLT repository](https://github.com/wix-incubator/DLT)
- **Paper:** [arXiv 2303.03755](https://arxiv.org/abs/2303.03755)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| PubLayNet | `creative-graphic-design/dlt-publaynet` | not-published |
| RICO13 | `creative-graphic-design/dlt-rico13` | not-published |
| Magazine | `creative-graphic-design/dlt-magazine` | gated follow-up |

## Uses

### Direct Use

Use DLT for research inference, checkpoint conversion checks, and training experiments for content-agnostic layout generation.

### Downstream Use

Converted DLT checkpoints can serve as layout priors for document and UI generation systems that consume normalized boxes, labels, masks, and dataset-local label metadata.

### Out-of-Scope Use

DLT is not a content-aware poster model and does not inspect rendered text or image saliency. The package is not intended for production publishing decisions without dataset-specific evaluation.

## Bias, Risks, and Limitations

DLT inherits the layout distributions and annotation limits of its training datasets. RICO13 uses a vendor-derived label mapping, and Magazine support remains limited until polygon and train-only handling are completed.

### Recommendations

Evaluate generated layouts on the target dataset and condition mode before using them in downstream tools. Keep generated checkpoint directories and vendor reference assets outside git.

## How to Get Started with the Model

Install the shared layout library and DLT package directly from this repository:

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "dlt @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/dlt"
```

The Hub checkpoints are not published yet. Clone the repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/dlt/REPRODUCING.md). Those steps create `.cache/dlt/converted/publaynet`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package dlt
uv run --package dlt \
  python models/dlt/scripts/smoke_from_pretrained.py \
  --path .cache/dlt/converted/publaynet
```

```python
from dlt import DLTPipeline

# After Hub publication: from_pretrained("creative-graphic-design/dlt-publaynet")
pipe = DLTPipeline.from_pretrained(".cache/dlt/converted/publaynet")
out = pipe(batch_size=1, condition_type="unconditional", seed=0)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
```

## Training Details

### Training Data

PubLayNet uses [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet). RICO13 uses [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) with `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` and the DLT label mapping. Magazine uses [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine) after polygon handling is enabled.

### Training Procedure

Training uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) through the shared [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) entry point and full `class_path` YAML configs. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/dlt/TRAINING.md) for smoke, PubLayNet, and RICO13 commands.

## Evaluation

### Parity Results

| Check | Cases | Match criterion | Result |
| --- | ---: | --- | --- |
| Scheduler mapping and transition matrices | 1 synthetic config | exact tensor equality | local unit test passed |
| Fixed training step | 1 synthetic batch | finite scalar loss and trace fields | local training test passed |
| Released checkpoint inference | 0 | trained checkpoint required | not run |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/dlt/REPRODUCING.md) to reproduce the original-implementation agreement checks by downloading vendor assets, generating reference metadata on one GPU, running parity checks, converting checkpoints, and smoke-testing local loading.

## License

The original DLT implementation is Apache-2.0. This converted package is Apache-2.0.

## Citation

```bibtex
@inproceedings{levi2023dlt,
  title = {Discrete-Continuous Layout Transformer for Layout Generation},
  author = {Levi, Elad and Brosh, Eli and Mykhailych, Mykola and Perez, Meir},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2023}
}
```
