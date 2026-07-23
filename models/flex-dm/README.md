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
            value: 16
            name: Vendor reference task cases
---

# Model Card for Flex-DM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.18248&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.18248)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/cyberagent/crello)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports Flex-DM, the MFP masked multi-field document model from [arXiv 2303.18248](https://arxiv.org/abs/2303.18248), to a `🤗transformers`-style package for document layout completion, refinement, and Crello image/text feature infilling.

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
| Crello `ours-exp-ft` | `creative-graphic-design/flex-dm-crello` | not-published |
| RICO `ours-exp-ft` | `creative-graphic-design/flex-dm-rico` | not-published |

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

Install the package directly from this repository. The command includes shared packages when they are not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "flex-dm @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/flex-dm"
```

The Hub repos are not published yet. Until then, create a local converted checkpoint and load it from `.cache/flex-dm/converted`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package flex-dm
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello
```

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/flex-dm/REPRODUCING.md) for the full reference-generation and parity workflow.

```python
from flex_dm import FlexDmPipeline

pipe = FlexDmPipeline.from_pretrained(".cache/flex-dm/converted/flex-dm-crello")

# After Hub publication: from_pretrained("creative-graphic-design/flex-dm-crello")
```

Flex-DM's public inference path is deterministic for the supported completion,
refinement, and image/text feature infilling modes. The pipeline accepts
`seed` and `generator` to match the common layout-generation API, and
`generator` takes precedence over `seed` when both are supplied, but the
current public path does not consume random numbers.

## Training Details

### Training Data

The original checkpoints were trained on vendor-preprocessed Crello and RICO data. Crello uses [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello) as the ordinary dataset reference, while RICO uses [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico) with the semantic-annotation config after vocabulary compatibility is verified.

### Training Procedure

Training follows the original TensorFlow implementation with `latent_dim=256`, `num_blocks=4`, `block_type=deepsvg`, `seq_type=default`, `arch_type=oneshot`, and fixed seed 0 in the released `args.json` files.

## Evaluation

### Parity Results

| checkpoint | dataset | vendor reference cases | exact state tensors | forward/layer probe | result |
| --- | --- | ---: | ---: | --- | --- |
| `ours-exp-ft` | Crello | 10 | 98 / 98 tensors, 2,812,257 params | 10 forward cases / 25 forward steps; max logit abs diff within `atol=1.1e-5`, `rtol=0` | reference generated with TF32 disabled; state mapping exact |
| `ours-exp-ft` | RICO | 6 | 88 / 88 tensors, 2,296,679 params | 6 forward cases / 15 forward steps; max logit abs diff within `atol=5.5e-6`, `rtol=0` | reference generated with TF32 disabled; state mapping exact |

Vendor parity tests load the saved TF32-disabled reference inputs from
`forward_cases/*.npz`, load the converted checkpoints with `from_pretrained`,
and execute every saved vendor forward step inside pytest. This covers
Crello's 10 task/iteration cases and RICO's 6 cases, including the four
forward steps used by `num_iter=4`. Layer probes also use one vendor test batch
after `preprocess_for_test` for every supported task (`elem`, `pos`, `attr`,
plus Crello `img` and `txt`). The first divergent block-0 operation in the
default vendor GPU path was the
attention score matmul `tf.matmul(q, k^T)`: TensorFlow 2.15 on A100 used TF32
there, while the public PyTorch model used fp32. Direct q/k isolation showed
the TF32 score differed from NumPy/PyTorch fp32 by `7.92e-3`, and PyTorch CUDA
TF32 matched the TensorFlow score. Final parity instead disables TensorFlow
TF32 during vendor reference and probe generation with
`NVIDIA_TF32_OVERRIDE=0` plus `--disable-tf32`, so the public PyTorch model
keeps its ordinary fp32 attention path. GELU was not involved because the
vendor MLP uses ReLU; Keras LayerNormalization epsilon is `1e-3` and is
matched by `FlexDmConfig.layer_norm_epsilon`; the additive mask remains
`-1e9` before softmax.

With TF32 disabled in the vendor run, all probe metadata records
`tf32_enabled=false`. The remaining first measurable difference is the fp32
attention-score reduction order at block 0, with max score diff up to
`7.63e-6`; block-0 attention output stays below `5.59e-8`. Across the
pytest-executed forward cases, Crello max logit diff is `1.00e-5` and RICO max
logit diff is `5.25e-6`.

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
