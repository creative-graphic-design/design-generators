---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "canvas-vae"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
model-index:
  - name: "CanvasVAE"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/Rico"
          name: "RICO"
          config: "ui-screenshots-and-hierarchies-with-semantic-annotations"
          split: "original training stream, first 50 batches"
        metrics:
          - type: "training-parity"
            value: "see Parity Results"
            name: "Training agreement with the original implementation"
---

<!-- --8<-- [start:card] -->

# Model Card for CanvasVAE

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2108.01249&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2108.01249)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [CanvasVAE](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html), the ICCV 2021 variational autoencoder for vector graphic documents, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package that trains on RICO mobile UI layouts with [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). To assess whether its training reproduces the original TensorFlow trainer, configuration, fixed-batch forward outputs, optimizer steps, short trajectories, and data streams are compared through stages [S0–S4](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md). For RICO only, on CPU in fp32, S0–S2 and S4 pass; S3 is bounded: in the natural 50-step trajectory, the report-only relative loss difference first exceeds the one-step limit at step 3, peaks at 5.760138e-4 at step 23, and is 1.309086e-4 at step 50, while the synchronized diagnostic passes its asserted loss and update limits. The full-training comparison (stage [S5](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md)) has not been run, so trained-checkpoint reproduction is not claimed.

## Model Details

### Model Description

CanvasVAE encodes a layout, meaning a sequence of up to 50 UI elements, into a 256-dimensional latent code with a transformer encoder, and decodes a latent code back into the element count and every element field in one pass. Each RICO element has a position and size discretized into 64 bins over the canvas, a `component` label, an `icon` class, a `text_button` class, and a `clickable` flag. Sampling the latent code from a standard normal prior generates new layouts. The pipeline returns normalized center `xywh` boxes in `[0, 1]`, `component` ids as `labels`, a valid-element `mask`, and `id2label`.

- **Developed by:** Kota Yamaguchi.
- **Shared by:** creative-graphic-design.
- **Model type:** variational autoencoder for layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [CanvasVAE repository](https://github.com/CyberAgentAILab/canvas-vae)
- **Paper:** [ICCV 2021 paper page](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html)
- **Released weights:** none; this package trains its own checkpoints.

## Supported Checkpoints

| Checkpoint | Hub ID                                    | Status        |
| ---------- | ----------------------------------------- | ------------- |
| RICO       | `creative-graphic-design/canvas-vae-rico` | not-published |

Trained-checkpoint reproduction is not yet claimed: the full-run comparison against the original trainer (stage [S5](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md), described in [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md)) has not been run. Crello is not supported because it first needs a separately trained image encoder.

## Uses

### Direct Use

Use this package to train CanvasVAE on RICO, to generate UI layouts from the prior, and to reconstruct layouts through `CanvasVAEModel`.

| `condition_type`                                                                                                    | Required inputs | Effect                                   |
| ------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------- |
| `unconditional`                                                                                                     | none            | samples latent codes and decodes layouts |
| `label`, `label_size`, `completion`, `refinement`, `text`, `content_image`, `relation`, `hierarchical`, `retrieval` | not applicable  | raises `NotImplementedError`             |

### Downstream Use

Generated layouts may feed UI prototyping, layout evaluation, or retrieval by latent code after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as accessibility annotations, semantic screen understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The model learns from RICO Android screens collected in 2017; layouts of other platforms, languages, or design eras are outside its training distribution. Element geometry is quantized to 64 bins, so generated boxes snap to a coarse grid.

### Recommendations

Rerun the agreement checks in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) after changing the model, losses, optimizer, or data pipeline.

## How to Get Started with the Model

Install the package directly from this repository. The command includes the shared `laygen` package because it is not published on PyPI.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "canvas-vae @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/canvas-vae"
```

Clone this repository, install the workspace member, and prepare RICO and a checkpoint with [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) or [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md). The conversion step there creates `.cache/canvas-vae/converted/initial`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package canvas-vae
uv run --package canvas-vae python
```

```python
from canvas_vae import CanvasVAEPipeline

path = ".cache/canvas-vae/converted/initial"
# After Hub publication: from_pretrained("creative-graphic-design/canvas-vae-rico")
pipe = CanvasVAEPipeline.from_pretrained(path)
out = pipe(batch_size=4, seed=0)

print(out.bbox.shape)
print(out.labels.shape)
print(out.id2label[int(out.labels[out.mask][0])])
```

## Training Details

### Training Data

| Dataset | Dataset ID                                                                                     | Notes                                                                                                                                                                                                                                                        |
| ------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | Training reads the original [semantic-annotation archive](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip) because its content-hash split and `textButtonClass` field are not in the hosted configuration. |

### Training Procedure

[TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) documents the LightningCLI configs and the staged agreement evidence against the original trainer.

#### Preprocessing

`scripts/prepare_rico.py` deduplicates identical annotation files, flattens each view hierarchy in depth-first pre-order, drops screens with more than 50 elements, and assigns 10% validation and 10% test splits by an MD5 content hash. Lookup tables keep `component` values seen at least once and `icon` and `text_button` values seen at least 500 times.

#### Training Hyperparameters

- **Training regime:** fp32, batch size 1024, 500 epochs of 45 batches, Adam with learning rate 1e-3 and per-tensor gradient-norm clipping at 1.0, KL weight 16, and L2 weight 1e-6.

#### Speeds, Sizes, Times

The model has 1,558,435 trainable parameters. Full-run training time has not been measured yet.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Agreement checks use the first 50 training batches of 1024 RICO layouts from the original data stream, the original records of every split, and the original stream order. Generated references stay under `.cache/canvas-vae/` and are not committed.

#### Factors

Results are reported per training-reproduction stage on RICO.

#### Metrics

Training agreement compares forward activations, losses, gradients, optimizer state, and parameters with float tolerances, and data records and stream order exactly. Layout quality uses the original reconstruction score, layout mIoU, and random-generation histogram score.

### Parity Results

Training agreement with the original TensorFlow trainer on RICO, measured on CPU in fp32 with the original's posterior noise injected and dropout 0. The stage names S0–S4 follow the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md); [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) gives the full evidence.

| Check                                                     |                                                             Cases | Criterion                                                                                                                                                                                                                                                                                                  | Result                                                                                                                                                                |
| --------------------------------------------------------- | ----------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0 static configuration and checkpoint key map            |         1,558,435 trainable parameters; 71 trainable tensors/keys | exact                                                                                                                                                                                                                                                                                                      | equal                                                                                                                                                                 |
| S0 eval-mode forward with copied weights                  |                                           1 batch of 1024 layouts | maximum relative-to-max difference `<= 1e-5`                                                                                                                                                                                                                                                               | 1.266045e-6                                                                                                                                                           |
| S1 train-mode forward activations and logits              |                                           1 batch of 1024 layouts | maximum relative-to-max difference `<= 1e-5` (`rtol=1e-5`)                                                                                                                                                                                                                                                 | 1.867336e-6                                                                                                                                                           |
| S1 per-field losses, KL, L2, total loss                   |                                                        12 scalars | maximum relative-to-max difference `<= 1e-6`                                                                                                                                                                                                                                                               | Per-field maximum 1.434896e-7; total loss max absolute difference 1.220703e-4 and relative difference 1.520953e-7                                                     |
| S2 gradients, clipped gradients, and Adam moments         |                 69 trainable tensors | Relative L2 difference `<= 1.8e-4` for gradients, clipped gradients, and first moments; `<= 2e-4` for second moments                                                                                                                                                                                       | Maxima: gradient 1.481723e-4, clipped gradient 1.481435e-4, first moment 1.481434e-4, second moment 1.342112e-4                                                       |
| S2 first-step parameter updates                           |                 69 trainable tensors | Cross-system relative L2 difference is a report-only diagnostic; it is not a pass/fail criterion                                                                                                                                                                                                           | Maximum observed difference 1.261390e-2                                                                                                                               |
| S3 natural 50-step trajectory and synchronized diagnostic |                           50 natural steps; 49 synchronized steps | Natural-trajectory loss differences are diagnostic, not a pass/fail gate: they first exceed `1e-6` at step 3, peak at 5.760138e-4 at step 23, and are 1.309086e-4 at step 50. Synchronized limits are 1e-6 for loss, 3.5e-4 for the Keras-Adam rule, and 1.6e-3 for well-conditioned cross-system updates. | Natural trajectory is bounded; synchronized maxima are 3.107378e-7 loss difference, 2.984581e-4 Keras-Adam-rule error, and 1.353780e-3 relative L2 update difference. |
| S4 original records and production-loader stream replay   | 45,222 / 5,584 / 5,623 train/validation/test records; 102 batches | exact content-hash sets, vocabularies, lookup tables, sample IDs, encoded fields, masks, and padding widths                                                                                                                                                                                                | equal; validation wraps 560 screens and the single-pass test stream ends with a batch of 503                                                                          |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) for the commands that download RICO, generate original-code references, run agreement checks, convert checkpoints, and smoke-test local loading. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) for package-local LightningCLI training.

## Environmental Impact

The staged agreement checks run on CPU in minutes. Full-run training cost will be recorded with the full-run comparison.

## Technical Specifications

### Model Architecture and Objective

The encoder sums per-field embeddings with learned position embeddings, applies one pre-norm transformer block conditioned on the element-count embedding, pools valid elements, applies batch normalization, and predicts a diagonal Gaussian posterior. The decoder applies one conditioned transformer block to position embeddings and predicts every field with linear heads. Training minimizes per-field cross-entropy summed over elements, plus 16 times the KL divergence, plus an L2 penalty on dense and embedding weights.

### Compute Infrastructure

Agreement checks run on CPU. Full RICO training is intended for one GPU.

#### Hardware

CPU is sufficient for inference, unit tests, and the staged agreement checks.

#### Software

Use `uv run --package canvas-vae ...` from the repository root. The `training` extra installs Lightning, and the `vendor` extra installs TensorFlow 2.15.1 and Apache Beam for original-code references.

## License

Repository wrapper code is Apache-2.0. The original implementation is released under the Apache-2.0 license.

## Citation

```bibtex
@inproceedings{yamaguchi2021canvasvae,
  title = {{CanvasVAE: Learning To Generate Vector Graphic Documents}},
  author = {Kota Yamaguchi},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021},
  pages = {5481-5489}
}
```

<!-- --8<-- [end:card] -->
