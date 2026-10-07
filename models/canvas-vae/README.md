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
          split: "S0-S4 staged training checks and full-run RICO comparison (S5)"
        metrics:
          - type: "training-parity"
            value: "S0-S4 staged checks; S5 RICO PASS; see Reproducibility"
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

This package ports [CanvasVAE](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html), a variational autoencoder for vector graphic documents, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package for training and layout generation. Full-run training reproduction passes the predeclared rule (the absolute system-mean difference must be no greater than twice the pooled sample SD) for Sreconst (mean reconstruction BLEU-1 across fields and test layouts), layout mIoU (mean class intersection-over-union on rasterized component maps), and Sgen (mean histogram intersection for generated-layout field and element-count distributions against RICO) across three 500-epoch training seeds per system on the 5,623-layout RICO test split; each Sgen value averages three evaluation seeds. This is a RICO-only claim; Crello remains blocked and is not claimed. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) gives the per-run results, comparison rule, evaluation scope, and reproduction commands.

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

Full-run reproduction is complete for RICO, with three training seeds per system and all three primary S5 metrics passing. The corrected package checkpoints pass the final-epoch gate at epoch 499 and global step 22,500; earlier package outputs from before that correction are excluded. The model weights are not published. Crello remains blocked and unclaimed because it requires a separately trained image encoder. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) for the full evidence.

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

Training agreement with the original TensorFlow trainer on RICO, measured on CPU in fp32 with the original's posterior noise injected and dropout 0. The stage names S0–S4 follow the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md); [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) gives the full evidence. These are historical reference-run measurements; a fresh independent S0–S4 rerun under the current checks is pending. Each new S0–S4 report records the CPU model and AVX-512 flag presence. Floating-point maxima in this table are rounded to the digits shown; raw JSON reports retain full precision.

| Check                                                     |                                                                                                                                        Cases | Criterion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0 static configuration and checkpoint key map            |                                                                    1,558,435 trainable parameters; 71 original trainable-parameter name keys | exact key-map equality                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Asserted equal. Raw `static.json.trainable_keys` contains 71 original trainable-parameter name keys; the S2 relative-comparison population contains 69 trainable parameter tensors because it excludes the encoder and decoder attention key-projection bias tensors (the two-key difference; these are not the 512 non-trainable batch-norm statistic values). Their exact gradient is zero because softmax is invariant to a shared key bias; the S2 row reports the S2 test's separate near-zero check of their gradients.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| S0 eval-mode forward with copied weights                  |                                                                                                                      1 batch of 1024 layouts | maximum relative-to-max difference `<= 1e-5`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Asserted maximum 1.266045e-6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| S1 train-mode forward activations and logits              |                                                                                                                      1 batch of 1024 layouts | maximum relative-to-max difference `<= 1e-5` (`rtol=1e-5`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Asserted maximum 1.867336e-6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| S1 per-field losses, KL, L2, total loss                   |                                                                                                                                   12 scalars | maximum relative-to-max difference `<= 1e-6` (the S1 loss limit)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Asserted per-field maximum 1.434896e-7 relative (limit 1e-6). The total-loss scalar passes the asserted relative check at 1.520953e-7 (limit 1e-6); its absolute difference 0.0001220703125 is report-only, not an absolute-tolerance gate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| S2 gradients, clipped gradients, and Adam moments         | 69 trainable parameter tensors in the S2 relative-comparison set; two key-projection bias tensors checked separately for near-zero gradients | Direct relative L2 limits remain `<= 1.8e-4` for gradients, clipped gradients, and first moments and `<= 2e-4` for second moments. A comparison that exceeds its direct limit passes only when both systems’ fp32 values are within relative L2 `<= 2.7e-3` of the corresponding float64 reference; clipped gradients and Adam moments are derived from the float64 gradient using the same clipping and optimizer formulas.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Historical reference-run measurements over the 69 tensors: gradient 1.481723e-4, clipped gradient 1.481435e-4, first moment 1.481434e-4, and second moment 1.342112e-4. A three-run diagnostic found eight comparisons above a direct limit across three raw gradients; its float64 error measurements and report-only arbitration counts are in [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md). These values are measurements, not gates; the current S2 rerun is pending. The separate near-zero-gradient check for each excluded bias tensor had a maximum absolute value of 6.600749e-8 across four tensor/system values (limit 1e-6).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| S2 first-step parameter updates                           |                                                                                                      69 trainable parameter tensors compared | Asserted per-tensor gates: actual updates are compared with each system's Keras 2 Adam update recomputed in float64 (relative L2 <= 3.5e-4); the asserted cross-system relative L2 limit is 1.6e-3 on the well-conditioned elements of each tensor, those where `sqrt(v) >= 100 * eps` (`eps=1e-7`; `v` is the Adam second moment). Report-only population: cross-system relative L2 over all elements of each of the 69 tensors                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Asserted maxima: cross-system relative L2 on the well-conditioned elements (`sqrt(v) >= 100 * eps`) 1.218219e-5; per-system Keras 2 Adam update-rule relative L2 2.782154e-5. Report-only maximum per-tensor relative L2 across the full-tensor population is 1.261390e-2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| S3 natural 50-step trajectory and synchronized diagnostic |           50 natural steps; 49 synchronized steps × 69 trainable parameter tensors = 3,381 gradient comparisons and 3,381 per-tensor updates | Natural trajectory, report-only with no pass/fail limit: the per-step total-loss difference \|package − original\| / \|original\| is shown against the S1 loss limit (`1e-6`). Synchronized diagnostic, asserted at each step after reloading the original's weights and optimizer state: total-loss difference <= 1e-6; per-tensor gradient cross-system relative L2 <= 1.8e-4, or, for a comparison that fails it, float64 relative L2 error `‖g_fp32 − g_f64‖₂ / ‖g_f64‖₂` <= 2.7e-3 for both systems, where `g_f64` recomputes the same step in float64 from the same saved state ([plan amendment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5976356123)); per-system Keras 2 Adam update-rule relative L2 <= 3.5e-4; cross-system update relative L2 <= 1.6e-3 on the well-conditioned elements of each tensor (`sqrt(v) >= 100 * eps`, `eps=1e-7`, `v` the Adam second moment); batch-norm running variance (256 values) maximum relative-to-max difference <= 1e-5. | Reference-run measurements are historical; a fresh independent S0–S4 rerun under the current checks is pending. Natural trajectory: first exceeds 1e-6 at step 3, peaks at 5.760138e-4 at step 23, and is 1.309086e-4 at step 50. Rerunning the original trainer changes its total loss by at most 0.000244140625 absolute, at most 1.04e-6 relative to the smallest original total loss in the run (235.01874), so the peak drift (5.760138e-4 at step 23) is at least 550 times the largest run-to-run variation; the drift comes from compounding differences between the two frameworks rather than run-to-run noise. Synchronized asserted maxima: total-loss difference 3.107378e-7; each gradient comparison passes either the direct limit or float64 arbitration for both systems. The reference run measured 2,993 direct passes and 388 arbitrated passes with a maximum float64 relative L2 error of 2.419970e-3 (89.6% of the limit); these counts and maximum are report-only, not gates. Keras 2 Adam update-rule error 2.984581e-4; well-conditioned cross-system update difference 1.353780e-3; running-variance difference 7.760259e-8 over 49 per-step checks. |
| S4 original records and production-loader stream replay   |                                                                            45,222 / 5,584 / 5,623 train/validation/test records; 102 batches | exact content-hash sets, vocabularies, lookup tables, sample IDs, encoded fields, masks, and padding widths                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Asserted equal; validation wraps 560 screens and the single-pass test stream ends with a batch of 503                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md) for commands to prepare RICO, generate original-code references, run agreement checks, convert checkpoints, and smoke-test loading. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records the completed RICO S5 comparison and package-local LightningCLI training; full-run reproduction is not claimed for Crello.

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
