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
          split: "RICO held-out S0-S4 parity confirmed; full-run RICO comparison (S5)"
        metrics:
          - type: "training-parity"
            value: "RICO S0-S4 held-out pass; S5 RICO metrics pass; see Reproducibility"
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

This package ports [CanvasVAE](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html), a variational autoencoder for vector graphic documents, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package for training and layout generation. RICO full-run parity is confirmed: an independent, from-scratch held-out run in normal mode passed all five staged checks under calibrated limits, from static initialization through training outputs and optimizer updates to production-loader replay ([S0–S4](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md)). The three-seed RICO evaluation comparison also passes all three primary metrics. Crello model, training, and evaluation support is implemented. Crello S0–S3 vendor model parity passes; full-run training has not been run, and no trained Crello checkpoint or quality result is claimed.

The original and package each completed three 500-epoch RICO runs and pass the predeclared comparison rule on 5,623 test layouts for the reconstruction score (Sreconst, mean BLEU-1 across fields and test layouts), layout mIoU (mean class intersection-over-union on rasterized component maps), and generated-layout score (Sgen, mean histogram intersection for generated-layout field and element-count distributions against RICO); each Sgen value averages three evaluation seeds. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) gives per-run results, the comparison rule, evaluation scope, calibration, and reproduction commands.

## Model Details

### Model Description

CanvasVAE encodes a layout, meaning a sequence of up to 50 UI elements, into a latent code with a transformer encoder, and decodes the element count and fields in one pass. The RICO model uses a 256-dimensional latent code; each element has geometry discretized into 64 bins, a `component` label, `icon` and `text_button` classes, and a `clickable` flag. Its pipeline returns normalized center `xywh` boxes in `[0, 1]`, `component` ids as `labels`, a valid-element `mask`, and `id2label`. The Crello model uses a 512-dimensional latent code, categorical context and element fields, a three-channel color head, and 256-dimensional image embeddings. It uses the same transformer blocks but has a separate config and model class.

- **Developed by:** Kota Yamaguchi.
- **Shared by:** creative-graphic-design.
- **Model type:** content-agnostic; task: single-task; conditioning: unconditional.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [CanvasVAE repository](https://github.com/CyberAgentAILab/canvas-vae)
- **Paper:** [ICCV 2021 paper page](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html)
- **Released weights:** none; this package trains its own checkpoints.

## Supported Checkpoints

| Checkpoint | Hub ID                                    | Status                |
| ---------- | ----------------------------------------- | --------------------- |
| RICO       | `creative-graphic-design/canvas-vae-rico` | not-published         |
| Crello     | —                                         | no trained checkpoint |

RICO full-run parity is complete: an independent held-out normal-mode run passed the calibrated S0–S4 checks, and the three-seed full-run comparison passed all primary metrics. The corrected package checkpoints passed the final-epoch gate at epoch 499 and global step 22,500; all package outputs from before that correction are excluded. Neither dataset has published weights. The Crello implementation requires prepared data and the verified 256-dimensional embedding fixture; its independent CPU S0–S3 vendor comparison passed, while full-run training has not been run. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) for the evidence and run status.

## Uses

### Direct Use

Use this package to train CanvasVAE on RICO or Crello. `CanvasVAEPipeline` generates and reconstructs RICO layouts; `CanvasVAECrelloModel` trains, reconstructs, and decodes Crello layouts from latent vectors. Crello requires the prepared data and embedding fixture described in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md).

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

Clone this repository, install the workspace member, and prepare RICO and a checkpoint with [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) or [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md). The conversion step there creates `.cache/canvas-vae/converted/initial`. The snippet below loads those untrained starting weights as a wiring check, so its layouts are not meaningful; a trained checkpoint needs the full training run in [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md).

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

```text
torch.Size([4, 49, 4])
torch.Size([4, 49])
Modal
```

## Training Details

### Training Data

| Dataset | Dataset ID / source                                                                                        | Notes                                                                                                                                                                                                                                                        |
| ------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)             | Training reads the original [semantic-annotation archive](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip) because its content-hash split and `textButtonClass` field are not in the hosted configuration. |
| Crello  | [Original Crello v1 archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip) | Training uses the canonical prepared splits and a verified posterior-mean fixture for every referenced image; the archive and derived assets stay private.                                                                                                   |

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

Agreement checks use data exported from the original trainer's RICO stream, with a different population per stage. S0's forward check, S1, and S2 use the first training batch of 1,024 layouts. S3 uses the first 50 training batches: the natural trajectory trains on all 50, and the synchronized diagnostic checks 49 steps. S4 compares all 45,222 / 5,584 / 5,623 train/validation/test records and replays 102 batches through the production loaders: 90 training, 6 validation, and 6 test batches. Generated references stay under `.cache/canvas-vae/` and are not committed.

#### Factors

Results are reported per training-reproduction stage on RICO.

#### Metrics

Training agreement compares forward activations, losses, gradients, optimizer state, and parameters with float tolerances, and data records and stream order exactly. Layout quality uses the original reconstruction score, layout mIoU, and random-generation histogram score.

### Parity Results

The table keeps RICO reference and calibration maxima separate from its independent held-out results. The held-out run used a fresh clone, initialized its submodule, prepared RICO, and regenerated references on an Intel Xeon CPU at 2.20 GHz without AVX-512. All five RICO S0–S4 commands passed; the documented suite reported 10 passed and 3 warnings in 3,591.26 seconds. See the [held-out evidence comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572). Crello CPU unit tests pass. Its independent-process S1–S3 vendor comparison passed on a High-RAM CPU using three calibration and three held-out processes; no full-run training or quality result is claimed.

| Check                                                                | Cases                                                                                                      | Criterion                                                                                                                                                                        | Historical calibration maximum                                                                                                                                                                                                                         | Held-out normal-mode result                                                                                                                                                                                                                         |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0 static configuration and checkpoint key map                       | 1,558,435 trainable parameters; 71 original trainable-parameter name keys                                  | Exact key-map equality                                                                                                                                                           | Reference and calibration reports retain equal topology.                                                                                                                                                                                               | Exact topology and key-map assertions passed.                                                                                                                                                                                                       |
| S0 evaluation forward                                                | 10 outputs from one batch of 1,024 layouts                                                                 | Maximum relative-to-max difference (`rtol=1e-5`)                                                                                                                                 | 1.266045e-6; reference run.                                                                                                                                                                                                                            | 1.139062e-6; limit 1e-5.                                                                                                                                                                                                                            |
| S1 activations and logits                                            | One exact encoder-context tensor and 14 relative-to-max comparisons from one batch                         | Maximum relative-to-max difference (`rtol=1e-5`)                                                                                                                                 | 2.916977e-6; independent rerun 1.                                                                                                                                                                                                                      | 2.087399e-6; limit 1e-5.                                                                                                                                                                                                                            |
| S1 per-field losses, KL, L2, and total loss                          | 12 scalars from one batch                                                                                  | Maximum relative-to-max difference (`rtol=1e-6`)                                                                                                                                 | 1.684800e-7; independent rerun 3.                                                                                                                                                                                                                      | 1.617341e-7; limit 1e-6.                                                                                                                                                                                                                            |
| S2 gradients, clipped gradients, and Adam moments                    | 69 trainable tensors per family; two key-projection bias tensors checked separately                        | Direct relative L2 `<= 3.5e-4` per family; float64 arbitration `<= 4.1e-3` if direct bound exceeded                                                                              | Gradient `2.278771e-4`, clipped gradient `2.277752e-4`, first moment `2.277754e-4` (calibration run 1); second moment `2.301702e-4` (independent rerun 2); max arbitration error `2.696621e-3` (calibration run 1).                                    | Raw/clipped/first-moment/second-moment maxima `6.333509e-5 / 6.333466e-5 / 6.333450e-5 / 6.393008e-5`; 276 direct, 0 arbitrated.                                                                                                                    |
| S2 optimizer update, excluded-bias gradient, and batch normalization | 69 updates per system; two excluded bias tensors × two systems; 256 moving-mean and moving-variance values | Adam-rule relative L2 `<= 3.5e-4`; well-conditioned update `<= 1.6e-3`; excluded-bias maximum absolute gradient `<= 1e-6`; moving-statistic relative-to-max difference `<= 1e-5` | Update-rule `2.829523e-5`, well-conditioned update `2.623599e-5`, excluded-bias gradient `6.600749e-8`, moving statistics `2.608958e-7`; historical maxima in Calibration Results.                                                                     | Update-rule `2.7879995e-5`; well-conditioned update `6.479192e-6`; excluded-bias gradient `7.178460e-8`; moving-mean difference `1.553235e-7`. The moving-variance gate also passed; its separate maximum was not included in the supplied summary. |
| S3 synchronized loss, gradients, and updates                         | 49 synchronized steps × 69 tensors = 3,381 gradient comparisons and per-tensor update checks               | Loss `<= 1.2e-6`; direct routing `3.5e-4`; shared float64 arbitration `4.1e-3`; Adam rule `5.1e-4`; well-conditioned update `6.1e-3`; running variance `1e-5`                    | Loss `7.353933e-7` (calibration run 3); arbitration `2.696621e-3` (calibration run 1); Adam rule `3.350313e-4` (independent rerun 3); well-conditioned update `4.037209e-3` (calibration run 1); running variance `8.221208e-8` (independent rerun 3). | Loss `3.044964e-7`; 3,268 direct and 113 arbitrated gradients, maximum arbitrated error `7.491022e-4`; Adam rule `2.973134e-4`; well-conditioned update `1.698453e-3`; running variance `8.067037e-8`.                                              |
| S4 original records and production-loader stream replay              | 45,222 / 5,584 / 5,623 train/validation/test records; record and stream gate counts recorded separately    | Exact content-hash sets, vocabularies, lookups, sample IDs, encoded fields, masks, and padding widths                                                                            | Calibration reports recorded exact equality.                                                                                                                                                                                                           | Exact equality; 24 record gates and 427 stream gates passed.                                                                                                                                                                                        |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) for commands to prepare RICO and Crello, generate original-code references, run the agreement checks, convert checkpoints, and load local models. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records the completed RICO S0–S5 evidence and the Crello package recipe and parity-run status. Full-run reproduction is claimed for RICO only.

## Environmental Impact

The staged agreement checks run on CPU in minutes. Full-run GPU telemetry is report-only and does not affect the S5 decision.

## Technical Specifications

### Model Architecture and Objective

The encoder sums per-field embeddings with learned position embeddings, applies one pre-norm transformer block conditioned on the element-count embedding, pools valid elements, applies batch normalization, and predicts a diagonal Gaussian posterior. The decoder applies one conditioned transformer block to position embeddings and predicts every field with linear heads. Training minimizes per-field cross-entropy summed over elements, plus 16 times the KL divergence, plus an L2 penalty on dense and embedding weights.

### Compute Infrastructure

Agreement checks run on CPU. Full RICO training used one A100 per training process, with training processes run sequentially per GPU.

#### Hardware

CPU is sufficient for inference, unit tests, and the staged agreement checks.

#### Software

Use `uv run --package canvas-vae ...` from the repository root. The `training` extra installs Lightning, and the `vendor` extra installs CUDA-enabled TensorFlow 2.15.1 and Apache Beam for original-code references. The separate `convert` extra installs CPU-only TensorFlow for original `initial.ckpt` and `final.ckpt` conversion. Package training keeps validation-best weights in `best.ckpt`; a separate unmonitored callback with `save_top_k: 1` retains one rolling checkpoint under Lightning’s epoch/step name and updates `last.ckpt` at configured checkpoint events. The final-checkpoint gate requires epoch 499 and global step 22,500 before evaluation.

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
