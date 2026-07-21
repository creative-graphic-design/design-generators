---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: other
tags:
  - flex-dm
  - layout-generation
datasets:
  - cyberagent/crello
  - creative-graphic-design/Rico
model-index:
  - name: Flex-DM
    results:
      - task:
          type: other
          name: Layout generation
        dataset:
          type: cyberagent/crello
          name: Crello
          split: vendor parity fixture
        metrics:
          - type: vendor-parity
            value: not run
            name: Vendor parity
---

# Model Card for Flex-DM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.18248&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.18248)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/cyberagent/crello)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=not--run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports Flex-DM, the MFP masked multi-field document model from [arXiv 2303.18248](https://arxiv.org/abs/2303.18248), to a 🤗 `transformers`-style package for document layout completion, refinement, and Crello image/text feature infilling.

## Model Details

### Model Description

Flex-DM models document elements as rows with element class, position, size, color, image, and text fields. The public pipeline returns normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`. Released checkpoints are masked document models for Crello and RICO.

- **Developed by:** Naoto Inoue et al.
- **Shared by:** creative-graphic-design.
- **Model type:** masked document layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [CyberAgentAILab/flex-dm](https://github.com/CyberAgentAILab/flex-dm)
- **Paper:** [Towards Flexible Multi-modal Document Models](https://arxiv.org/abs/2303.18248)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| Crello `ours-exp-ft` | `creative-graphic-design/flex-dm-crello-explicit-ft` | not-published |
| RICO `ours-exp-ft` | `creative-graphic-design/flex-dm-rico-explicit-ft` | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and original-implementation agreement checks for Flex-DM document infilling.

### Downstream Use

The package can be used as a component in document design tooling that needs masked layout completion or multimodal field infilling after converted checkpoints are available locally.

### Out-of-Scope Use

The package is not intended for production publishing workflows, legal document validation, or unrestricted generation of copyrighted design assets.

## Bias, Risks, and Limitations

The released checkpoints reflect the visual and structural distributions of their training datasets. Crello image/text fields depend on vendor preprocessing artifacts, and RICO label ordering must remain compatible with the released vocabulary.

### Recommendations

Validate outputs against the target dataset schema before using generated layouts in downstream evaluation. Run the original-implementation agreement checks before relying on converted checkpoints.

## How to Get Started with the Model

The Hub repos are not published yet. Until then, create a local converted checkpoint and load it from `.cache/flex-dm/converted`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package flex-dm
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello-explicit-ft
```

See [REPRODUCING.md](REPRODUCING.md) for the full reference-generation and parity workflow.

```python
from flex_dm import FlexDmPipeline

pipe = FlexDmPipeline.from_pretrained(".cache/flex-dm/converted/flex-dm-crello-explicit-ft")

# After Hub publication: from_pretrained("creative-graphic-design/flex-dm-crello-explicit-ft")
```

## Training Details

### Training Data

The original checkpoints were trained on vendor-preprocessed Crello and RICO data. Crello uses [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello) as the ordinary dataset reference, while RICO uses [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico) with the semantic-annotation config after vocabulary compatibility is verified.

### Training Procedure

Training follows the original TensorFlow implementation with `latent_dim=256`, `num_blocks=4`, `block_type=deepsvg`, `seq_type=default`, `arch_type=oneshot`, and fixed seed 0 in the released `args.json` files.

## Evaluation

### Parity Results

| checkpoint | dataset | task cases | exact items | tolerance items | result |
| --- | --- | ---: | ---: | ---: | --- |
| `ours-exp-ft` | Crello | 0 | 0 | 0 | not run |
| `ours-exp-ft` | RICO | 0 | 0 | 0 | not run |

## Reproducibility

Reproduce the original-implementation agreement checks with the walkthrough in [models/flex-dm/REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/flex-dm/REPRODUCING.md).

## License

Repository code is Apache-2.0. The original Flex-DM implementation is Apache-2.0. Dataset licensing follows the source dataset metadata.

## Citation

```bibtex
@inproceedings{inoue2023flexdm,
  title = {Towards Flexible Multi-modal Document Models},
  booktitle = {CVPR},
  year = {2023}
}
```
