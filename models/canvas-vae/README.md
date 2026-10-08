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
          split: "S0-S4 calibrated limits pending held-out run; full-run RICO comparison (S5)"
        metrics:
          - type: "training-parity"
            value: "S0-S4 calibration complete; held-out pending; S5 RICO metrics pass; see Reproducibility"
            name: "Training agreement with the original implementation"
---

<!-- --8<-- [start:card] -->

# Model Card for CanvasVAE

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2108.01249&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2108.01249)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [CanvasVAE](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html), a variational autoencoder for vector graphic documents, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package for training and layout generation. Full-run parity remains pending: one independent held-out normal-mode run must pass all five staged checks, from static initialization through training outputs and optimizer updates to production-loader replay ([S0–S4](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md)).

Separately, the RICO evaluation comparison is complete. The original and package each completed three 500-epoch runs and pass the predeclared comparison rule on 5,623 test layouts for the reconstruction score (Sreconst, mean BLEU-1 across fields and test layouts), layout mIoU (mean class intersection-over-union on rasterized component maps), and generated-layout score (Sgen, mean histogram intersection for generated-layout field and element-count distributions against RICO); each Sgen value averages three evaluation seeds. Crello remains blocked and unclaimed. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) gives per-run results, the comparison rule, evaluation scope, calibration, and reproduction commands.

## Model Details

### Model Description

CanvasVAE encodes a layout, meaning a sequence of up to 50 UI elements, into a 256-dimensional latent code with a transformer encoder, and decodes a latent code back into the element count and every element field in one pass. Each RICO element has a position and size discretized into 64 bins over the canvas, a `component` label, an `icon` class, a `text_button` class, and a `clickable` flag. Sampling the latent code from a standard normal prior generates new layouts. The pipeline returns normalized center `xywh` boxes in `[0, 1]`, `component` ids as `labels`, a valid-element `mask`, and `id2label`.

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

| Checkpoint | Hub ID                                    | Status        |
| ---------- | ----------------------------------------- | ------------- |
| RICO       | `creative-graphic-design/canvas-vae-rico` | not-published |

The RICO S5 comparison is complete: all three primary metrics pass across three training seeds per system, and the corrected package checkpoints pass the final-epoch gate at epoch 499 and global step 22,500. The overall full-run parity claim remains pending one independent held-out normal-mode run of the recalibrated S0–S4 checks. Earlier package outputs from before the checkpoint correction are excluded. The model weights are not published. Crello remains blocked and unclaimed because it requires a separately trained image encoder. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) for the full evidence.

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

Agreement checks use data exported from the original trainer's RICO stream, with a different population per stage. S0's forward check, S1, and S2 use the first training batch of 1,024 layouts. S3 uses the first 50 training batches: the natural trajectory trains on all 50, and the synchronized diagnostic checks 49 steps. S4 compares all 45,222 / 5,584 / 5,623 train/validation/test records and replays 102 batches through the production loaders: 90 training, 6 validation, and 6 test batches. Generated references stay under `.cache/canvas-vae/` and are not committed.

#### Factors

Results are reported per training-reproduction stage on RICO.

#### Metrics

Training agreement compares forward activations, losses, gradients, optimizer state, and parameters with float tolerances, and data records and stream order exactly. Layout quality uses the original reconstruction score, layout mIoU, and random-generation histogram score.

### Parity Results

Calibration used the pre-registered rule linked from [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md). The reports cover the reference run, earlier independent runs for stages they completed, and three retained calibration runs; numeric reports from calibration run 2 were not retained. Calibration mode records measurements without enforcing the candidate limits. The vendor-parity badge refers to the required held-out normal-mode run, which has not run yet, so these calibration results do not establish S0–S4 parity. The three retained calibration reports record an Intel Xeon CPU at 2.20 GHz with AVX-512 absent. Maxima are rounded to seven significant digits; report-only values do not affect pass/fail decisions.

| Check                                                                |                                                                                                          Cases | Criterion                                                                                                                                                                                                                                                                                        | Result                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0 static configuration and checkpoint key map                       |                                      1,558,435 trainable parameters; 71 original trainable-parameter name keys | Exact key-map equality                                                                                                                                                                                                                                                                           | The reference and calibration reports retain equal topology; the held-out normal-mode run is pending.                                                                                                                                                                                                                                                                                |
| S0 evaluation forward                                                |                                                                     10 outputs from one batch of 1,024 layouts | Maximum relative-to-max difference (`rtol=1e-5`)                                                                                                                                                                                                                                                 | Calibration-set maximum: `1.266045e-6` in the reference run.                                                                                                                                                                                                                                                                                                                         |
| S1 activations and logits                                            |            One exact encoder-context tensor and 14 relative-to-max comparisons from one batch of 1,024 layouts | Maximum relative-to-max difference (`rtol=1e-5`)                                                                                                                                                                                                                                                 | Calibration-set maximum: `2.916977e-6` in independent rerun 1.                                                                                                                                                                                                                                                                                                                       |
| S1 per-field losses, KL, L2, and total loss                          |                                                                     12 scalars from one batch of 1,024 layouts | Maximum relative-to-max difference (`rtol=1e-6`)                                                                                                                                                                                                                                                 | Calibration-set maximum: `1.684800e-7` in independent rerun 3.                                                                                                                                                                                                                                                                                                                       |
| S2 gradients, clipped gradients, and Adam moments                    |    69 trainable tensors per family; two key-projection bias tensors checked separately for near-zero gradients | Direct relative L2 `<= 3.5e-4` for each family; if exceeded, both systems' fp32 values must be within relative L2 `<= 4.1e-3` of the same float64 reference. Clipped gradients and moments use the corresponding float64-gradient formulas.                                                      | Calibration maxima: gradient `2.278771e-4`, clipped gradient `2.277752e-4`, and first moment `2.277754e-4` in calibration run 1; second moment `2.301702e-4` in independent rerun 2. Maximum float64 arbitration error: `2.696621e-3` in calibration run 1. Direct/arbitrated counts are report-only.                                                                                |
| S2 optimizer update, excluded-bias gradient, and batch normalization | 69 per-tensor updates; two excluded bias tensors × two systems; 256 moving-mean and 256 moving-variance values | Per-system Keras 2 Adam update-rule relative L2 `<= 3.5e-4`; cross-system relative L2 `<= 1.6e-3` on elements where `sqrt(v) >= 1e-5`; excluded-bias maximum absolute gradient `<= 1e-6`; moving-statistic relative-to-max difference `<= 1e-5`                                                  | Calibration maxima: update-rule `2.829523e-5` (independent rerun 1), well-conditioned update `2.623599e-5` (independent rerun 2), excluded-bias gradient `6.600749e-8` (reference run), and moving statistics `2.608958e-7` (calibration run 1).                                                                                                                                     |
| S3 synchronized loss, gradients, and updates                         |                         49 synchronized steps × 69 tensors = 3,381 gradient comparisons and per-tensor updates | Total-loss relative difference `<= 1.2e-6`; direct gradient routing threshold `3.5e-4` inherited from S2, followed by float64 arbitration `<= 4.1e-3`; Adam-rule relative L2 `<= 5.1e-4`; well-conditioned update relative L2 `<= 6.1e-3`; running-variance relative-to-max difference `<= 1e-5` | Calibration maxima: loss `7.353933e-7` (calibration run 3), float64 error `2.696621e-3` (calibration run 1), Adam-rule error `3.350313e-4` (independent rerun 3), well-conditioned update difference `4.037209e-3` (calibration run 1), and running variance `8.221208e-8` (independent rerun 3). Direct/arbitrated counts and the observed direct-gradient maximum are report-only. |
| S4 original records and production-loader stream replay              |                                              45,222 / 5,584 / 5,623 train/validation/test records; 102 batches | Exact content-hash sets, vocabularies, lookups, sample IDs, encoded fields, masks, and padding widths                                                                                                                                                                                            | Calibration reports record exact equality; the held-out normal-mode run is pending.                                                                                                                                                                                                                                                                                                  |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) for commands to prepare RICO, generate original-code references, run agreement checks, convert checkpoints, and smoke-test loading. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records the completed RICO S5 measurements, recalibrated S0–S4 limits, and package-local LightningCLI training. The held-out run required for a full-run parity claim is pending; Crello remains unclaimed.

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
