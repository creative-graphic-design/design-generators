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

<!-- --8<-- [start:card] -->

# Model Card for Flex-DM

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.18248&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.18248)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR%202023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/cyberagent/crello)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports Flex-DM, the MFP masked multi-field document model from [arXiv 2303.18248](https://arxiv.org/abs/2303.18248), to a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package for document layout completion, refinement, and Crello image/text feature infilling.

## Model Details

### Model Description

Flex-DM models document elements as rows with element class, position, size, color, image, and text fields. The public pipeline returns normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`. Released checkpoints are masked document models for Crello and RICO.

- **Developed by:** Naoto Inoue et al.
- **Shared by:** creative-graphic-design.
- **Model type:** content-agnostic; task: task-agnostic; conditioning: completion, refinement.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [CyberAgentAILab/flex-dm](https://github.com/CyberAgentAILab/flex-dm)
- **Paper:** [Towards Flexible Multi-modal Document Models](https://arxiv.org/abs/2303.18248)

## Supported Checkpoints

| Checkpoint           | Hub ID                                   | Status        |
| -------------------- | ---------------------------------------- | ------------- |
| Crello `ours-exp-ft` | `creative-graphic-design/flex-dm-crello` | not-published |
| RICO `ours-exp-ft`   | `creative-graphic-design/flex-dm-rico`   | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and original-implementation agreement checks for Flex-DM document infilling.

### Downstream Use

The package can be used as a component in document design tooling that needs masked layout completion or multimodal field infilling after converted checkpoints are available locally.

### Out-of-Scope Use

The package is not intended for production publishing workflows, legal document validation, or unrestricted generation of copyrighted design assets.

## Bias, Risks, and Limitations

The released checkpoints reflect the visual and structural distributions of their training datasets. Crello image/text fields depend on original-implementation preprocessing artifacts, and RICO label ordering must remain compatible with the released vocabulary.

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
uv run --package flex-dm --extra data models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra convert models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello
```

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/flex-dm/REPRODUCING.md) for the full original-implementation reference-generation and agreement-check workflow.

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

The original checkpoints were trained on data preprocessed by the original implementation for Crello and RICO. Crello uses [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello) as the ordinary dataset reference, while RICO uses [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico) with the semantic-annotation config after vocabulary compatibility is verified.

### Training Procedure

Training follows the original TensorFlow implementation with `latent_dim=256`, `num_blocks=4`, `block_type=deepsvg`, `seq_type=default`, `arch_type=oneshot`, and fixed seed 0 in the released `args.json` files.

## Evaluation

### Parity Results

| checkpoint    | dataset | original-implementation reference cases |               exact state tensors | forward comparison                                                                                 |
| ------------- | ------- | --------------------------------------: | --------------------------------: | -------------------------------------------------------------------------------------------------- |
| `ours-exp-ft` | Crello  |                                      10 | 98 / 98 tensors, 2,812,257 params | 10 cases / 25 steps; max logit abs diff `1.049041748e-5` against `atol=1.1e-5`, `rtol=0`; all pass |
| `ours-exp-ft` | RICO    |                                       6 | 88 / 88 tensors, 2,296,679 params | 6 cases / 15 steps; max logit abs diff `5.24520874e-6` against `atol=5.5e-6`, `rtol=0`; all pass   |

Forward comparisons load each saved original-implementation case, run every saved step, and compare all logits. The table records the maximum absolute logit difference for each task and iteration count; each case uses the unchanged dataset tolerance shown in the final column. Values are rounded to 9 or 10 significant digits.

| dataset | task | max diff, `num_iter=1` | max diff, `num_iter=4` | retained `atol`, `rtol=0` |
| ------- | ---- | ---------------------: | ---------------------: | ------------------------: |
| Crello  | elem |        `7.62939453e-6` |       `1.049041748e-5` |                  `1.1e-5` |
| Crello  | pos  |       `9.775161743e-6` |       `9.775161743e-6` |                  `1.1e-5` |
| Crello  | attr |        `1.00135803e-5` |        `1.00135803e-5` |                  `1.1e-5` |
| Crello  | img  |        `5.48362732e-6` |        `5.48362732e-6` |                  `1.1e-5` |
| Crello  | txt  |       `2.384185791e-6` |       `2.384185791e-6` |                  `1.1e-5` |
| RICO    | elem |       `3.814697266e-6` |       `4.291534424e-6` |                  `5.5e-6` |
| RICO    | pos  |       `2.384185791e-6` |       `2.861022949e-6` |                  `5.5e-6` |
| RICO    | attr |       `3.814697266e-6` |        `5.24520874e-6` |                  `5.5e-6` |

Agreement-check tests load the saved TF32-disabled reference inputs from `forward_cases/*.npz`, load the converted checkpoints with `from_pretrained`, and execute every saved original-implementation forward step inside pytest. This covers Crello's 10 task/iteration cases and RICO's 6 cases, including the four forward steps used by `num_iter=4`. Layer probes also use one original-implementation test batch after `preprocess_for_test` for every supported task (`elem`, `pos`, `attr`, plus Crello `img` and `txt`). The first divergent block-0 operation was attention-score `tf.matmul(q, k^T)`: TensorFlow 2.15 used TF32 on the GPU path, while the public PyTorch model used fp32. Direct q/k isolation showed the TF32 score differed from NumPy/PyTorch fp32 by `7.92e-3`, and PyTorch CUDA TF32 matched the TensorFlow score. Final agreement checks instead disable TensorFlow TF32 during original-implementation reference and probe generation with `NVIDIA_TF32_OVERRIDE=0` plus `--disable-tf32`, so the public PyTorch model keeps its ordinary fp32 attention path. GELU was not involved because the original-implementation MLP uses ReLU; Keras LayerNormalization epsilon is `1e-3` and is matched by `FlexDmConfig.layer_norm_epsilon`; the additive mask remains `-1e9` before softmax.

| dataset | task | max logit diff / retained `atol` | max encoder diff | max block-0 score diff | max block-0 attention-output diff |
| ------- | ---- | -------------------------------: | ---------------: | ---------------------: | --------------------------------: |
| Crello  | attr |       `1.00135803e-5` / `1.1e-5` |  `2.42143869e-7` |        `4.76837158e-6` |                    `5.0291419e-8` |
| Crello  | elem |         `7.62939453e-6` / `8e-6` |  `2.01165676e-7` |        `7.62939453e-6` |                   `5.58793545e-8` |
| Crello  | img  |         `5.48362732e-6` / `6e-6` |   `2.5331974e-7` |         `5.7220459e-6` |                    `3.3993274e-8` |
| Crello  | pos  |        `9.775161743e-6` / `1e-5` |  `2.38418579e-7` |        `7.62939453e-6` |                   `4.47034836e-8` |
| Crello  | txt  |      `2.384185791e-6` / `2.5e-6` |  `2.23517418e-7` |         `5.7220459e-6` |                   `4.63332981e-8` |
| RICO    | attr |        `3.814697266e-6` / `4e-6` |  `5.96046448e-8` |        `7.62939453e-6` |                   `5.21540642e-8` |
| RICO    | elem |        `3.814697266e-6` / `4e-6` |  `2.98023224e-8` |         `5.7220459e-6` |                    `5.0291419e-8` |
| RICO    | pos  |      `2.384185791e-6` / `2.5e-6` |  `1.49011612e-8` |        `7.62939453e-6` |                   `5.58793545e-8` |

Layer probes use one original-implementation test batch after `preprocess_for_test`. Their unchanged checks require encoder diff `<=3e-7`, block-0 score diff `<=8e-6`, and block-0 attention-output diff `<=8e-8`. The README tables above report all measured forward and layer maxima; no tolerance increase was needed.

The currently verified stack is NVIDIA A100-SXM4-80GB, driver 610.57.04 (driver CUDA 13.3), PyTorch 2.13.0+cu130 (CUDA 13.0), and TensorFlow 2.15.1 (built with CUDA 12.2). Reference generation used `CUDA_VISIBLE_DEVICES=0`, `NVIDIA_TF32_OVERRIDE=0`, and `--disable-tf32`.

## Reproducibility

Reproduce the original-implementation agreement checks with the walkthrough in [models/flex-dm/REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/flex-dm/REPRODUCING.md). It installs the `data` and `vendor-tf` extras and the `dev` dependency group; `vendor-tf` supplies TensorFlow for reference generation, and `dev` supplies pytest for the required parity suite.

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

<!-- --8<-- [end:card] -->
