# CanvasVAE Training

Three 500-epoch training seeds per system were evaluated for RICO, and the Sreconst, layout mIoU, and Sgen comparisons pass the predeclared statistical rule. An independent, from-scratch held-out run in normal mode passed all five RICO stages, S0–S4, under the calibrated limits; full-run reproduction is claimed for RICO only. The Crello model, training, and evaluation paths use the original 512-dimensional latent and KL weight 32. Its independent-process S0–S3 vendor comparison passed on a High-RAM CPU using three calibration and three held-out processes; no Crello full-run result is claimed. See the [training-reproduction guide](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md), [Calibration Results](#calibration-results) for historical calibration values, and [Held-out Validation Results](#held-out-validation-results) for the independent RICO confirmation.

Run commands from the repository root. Keep generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts under `.cache/canvas-vae/`.

## Install

```bash
uv sync --package canvas-vae --extra training
```

Install the `vendor` extra for original-code references and training; it adds CUDA-enabled TensorFlow 2.15.1 and Apache Beam. For conversion, use the separate `convert` extra only with original TensorFlow checkpoint prefixes such as `initial.ckpt` and `final.ckpt`; it installs `tensorflow-cpu==2.15.1` without NVIDIA CUDA wheels. A Lightning `last.ckpt` converts without a TensorFlow extra. Keep `vendor` and `convert` in separate environments.

```bash
uv sync --package canvas-vae --extra training --extra vendor
```

## Data

| Dataset  | Source                                                                                                                                                                        | Config or path                                                                                                                                                                                                                                                                                                   |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rico`   | [RICO semantic-annotation archive](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip), MD5 `5dd3372e2d99b342958136e394d4de79` | `scripts/prepare_rico.py` writes `.cache/canvas-vae/data/rico/{train,val,test}.jsonl`, `vocabulary.json`, and `count.json`: both systems have 45,222 / 5,584 / 5,623 train/validation/test screens after content deduplication, filtering screens with more than 50 elements, and splitting by MD5 content hash. |
| `crello` | [Original Crello v1 archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip)                                                                    | Use the canonical prepared splits and verified 256-dimensional posterior-mean fixture described in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md); no full-run result is claimed.                                                     |

## Configs

Training configs live under `models/canvas-vae/configs/training`.

| Config                               | Dataset          | Seed mode                      | Purpose                                                                                                |
| ------------------------------------ | ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `canvas_vae_rico.yaml`               | `rico`           | default (`seed_everything: 0`) | Full run: 500 epochs of 45 batches of 1024, validation every 20 epochs.                                |
| `canvas_vae_rico_deterministic.yaml` | `rico`           | deterministic, dropout 0       | Bounded production-wiring run for step-level checks.                                                   |
| `canvas_vae_crello.yaml`             | `crello`         | default (`seed_everything: 0`) | Crello recipe: latent dimension 512, KL weight 32, and 500 epochs.                                     |
| `smoke.yaml`                         | prepared fixture | deterministic                  | CPU wiring smoke with a tiny model; pass `--model.init_args.data_dir` and `--data.init_args.data_dir`. |

## Crello Model Workflow

The Crello model consumes the source context fields, typed element fields, and verified 256-dimensional image embeddings. Its color head predicts three categorical channels, and conditional losses and metrics use the source type masks. The training config uses latent dimension 512 and KL weight 32, matching the original Crello command.

Prepare the data and embedding fixture using the Crello data contract and commands in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md), then run package training and evaluation with these commands. The 500-epoch command is the full training recipe; this change did not run it or claim a trained-model result.

```bash
uv run --package canvas-vae --extra training traingen fit --config models/canvas-vae/configs/training/canvas_vae_crello.yaml --trainer.accelerator=cpu --trainer.devices=1
uv run --package canvas-vae models/canvas-vae/scripts/convert_original_checkpoint.py --dataset crello --checkpoint .cache/canvas-vae/training-runs/crello/lightning_logs/version_0/checkpoints/last.ckpt --vocabulary .cache/canvas-vae/crello/package-run-1/vocabulary.json --output-dir .cache/canvas-vae/converted/crello-package
uv run --package canvas-vae --extra training models/canvas-vae/scripts/evaluate_crello.py --checkpoint .cache/canvas-vae/converted/crello-package --output .cache/canvas-vae/evaluation/crello-package.json
```

The evaluation script reports per-document masked reconstruction metrics on the Crello test split and prior-generation histogram scores for each requested seed. It loads only a package model directory written by `save_pretrained` or the conversion command above.

## Scheduler and Recipe Notes

The recipe follows the original trainer's effective behavior rather than generic Lightning defaults:

- Adam uses the Keras 2 update (`m += (g - m)(1 - beta1)`, epsilon `1e-7` added after `sqrt(v)` with bias correction folded into the step size) at a constant learning rate of `1e-3`.
- Every gradient tensor is clipped to norm 1.0 on its own (Keras `clipnorm`), inside `configure_gradient_clipping`. Lightning's `gradient_clip_val`, a global-norm clip, is rejected.
- For RICO, the loss is the per-field cross-entropy summed over valid elements and fields, averaged over the batch, plus 16 times the KL divergence averaged over batch and latent dimensions, plus `1e-6` times the squared norm of every dense and embedding weight. Crello uses the same reduction with KL weight 32.
- Batch normalization of the pooled encoder output uses the batch mean and biased batch variance with moving-average momentum 0.99. The unmodified TensorFlow 2.15 layer receives the `(batch, elements)` padding mask propagated from the pooled block and fails, while the original trainer was written for TensorFlow 2.3/2.4, whose batch normalization ignored masks.
- A batch-normalization compatibility control, `models/canvas-vae/scripts/compare_batch_norm_control.py`, checks that mask-free harness against the unmodified original code. It is not part of the parity suite and runs outside the workspace environment on TensorFlow 2.11.1, the last release before Keras `BatchNormalization.call` gained the `mask` argument in 2.12, so the original code runs there without edits: `uv run --no-project --python 3.10 --with "tensorflow-cpu==2.11.1" --with "numpy<2" --with pyyaml --with jaxtyping models/canvas-vae/scripts/compare_batch_norm_control.py`. It compares one training forward pass and one Adam step on the reference trace's first batch of 1,024 layouts, initial checkpoint, and posterior noise with the TensorFlow 2.15.1 trace, and writes `.cache/canvas-vae/reference/reports/batch_norm_control.json`. The forward and batch-normalization values are within the script's 1e-5 relative tolerance: the total-loss scalar (802.59094 on both sides; relative difference 0), the maximum relative difference across nine logits tensors (1.167085e-6), the moving-mean tensor (maximum relative difference 2.078424e-7), and the moving-variance tensor (relative difference 0). The script's strict check also requires every post-step model-state key to be within 1e-5, and only 44 of the 73 keys are, so after writing the report the script raises `AssertionError` as expected; full post-step model-state agreement is not claimed. The raw control report is retained with the campaign's archived evidence, and the result is summarized in the [S0–S4 evidence comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5963001129). Floating-point values in this summary are rounded to the digits shown; the raw report retains full precision.
- The control's report also contains a float64 update analysis that the script does not check against a limit. It covers all 71 matched trainable parameter tensors, including the two key-projection bias tensors omitted from the S2 relative-comparison population of 69 tensors. It compares each system's actual update with the Keras 2 Adam update recomputed in float64 (maximum per-system relative L2 error 2.7774e-5), and compares systems over the well-conditioned elements of each tensor, those where `sqrt(v) >= 100 * eps` (`eps=1e-7`, `v` is the Adam second moment; maximum relative L2 1.285e-5). The near-zero-gradient elements, those where `sqrt(v) < 100 * eps`, are 14.4967% of the elements of the 71 tensors and account for 99.999856% of their summed squared cross-system update difference.
- The shared S2/S3 float64-reference relative-L2 arbitration limit is `4.1e-3`. The synchronized direct comparison uses the recalibrated one-step raw-gradient limit of `3.5e-4` as its routing prefilter. The literal calibration formula would set this prefilter to `4.1e-3`, leaving no comparisons in the calibration set for arbitration; retaining `3.5e-4` is stricter because lowering the direct threshold can only reject more under the pass rule: a comparison passes when its direct error is at most the direct threshold or its arbitrated error is at most the shared limit. S3 data raised the shared limit to `4.1e-3`, while S2 arbitrated errors were at most `2.29e-4`. For any comparison above the direct threshold, both systems’ fp32 gradients must be within relative L2 `4.1e-3` of the same saved-state step recomputed in float64. Direct/arbitrated counts and maximum arbitration error are report-only.
- The direct one-step limits for raw gradients, per-tensor clipped gradients, first moments, and second moments are each `3.5e-4`. A comparison above its direct limit is checked against a float64 reference recomputed from the same one-step inputs; both systems’ fp32 values must each be within relative L2 `4.1e-3`. Clipped gradients and moments derive from the float64 gradient using the same per-tensor norm-1.0 clipping and Keras 2 Adam formulas. Direct/arbitrated counts and maximum arbitration error are report-only.
- S0–S4 use `num_workers: 0`; S5 uses `num_workers: 2` and one training process per A100. The asserted CPU full-epoch worker-stream check matched all 45 batch hashes over the 45,222-record training split. GPU utilization, board power, process CPU, and finite-window throughput are report-only diagnostics.
- Training batches come from one endless stream of shuffled passes with 45 full batches per epoch, so batches straddle pass boundaries. Validation scores 6 batches from an ordered stream, wrapping 560 screens; testing scores all 5,623 screens once in 6 batches, with 503 in the final batch. S4 replays the original exported order through the production loaders and does not claim that native shuffles match across frameworks.
- The training config keeps validation-best weights in `best.ckpt`. Its separate unmonitored checkpoint callback retains one rolling top-1 checkpoint under Lightning’s epoch/step name and uses `save_last: true` to update `last.ckpt` at each configured save event; the final full-run validation saves epoch 499 at global step 22,500. With `save_top_k: 0`, Lightning 2.6.5 would defer a never-written `last.ckpt` until `on_train_end`, when its epoch metadata is one past the final epoch. The [2.6.5 callback source](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.5/src/lightning/pytorch/callbacks/model_checkpoint.py#L449-L496) documents the save trigger and end behavior. The S5 executor checks the checkpoint metadata before copying or uploading it.

The protocol's [Stage Rules](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#stage-rules) apply only to components the training loop contains, so these rules are not applicable:

- Scheduler — not applicable: the training loop has no scheduler or warmup.
- Sampler — not applicable: the trainer has no timestep or importance sampler; `CrossEpochBatchSampler` only constructs the data stream.
- EMA — not applicable: training does not maintain exponential-moving-average weights.
- Automatic mixed precision (AMP) — not applicable: both systems run fp32.

- Run-completion gate — require a successful training exit and a valid checkpoint before conversion and evaluation. Package checkpoint metadata must report epoch 499 and global step 22,500; a missing checkpoint, invalid metadata, missing exit record, or nonzero exit stops the run.

## Seed Policy

The original code sets no seed; `scripts/train_original.py` calls `tf.keras.utils.set_random_seed(seed)` before the unmodified trainer, and the package configs set `seed_everything`. The [S1–S4 checks](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md) inject the original's posterior noise into the package and disable dropout on both sides, so they need no cross-framework RNG pairing. S5 targets `training-seed n=3` per system, not seed-paired across frameworks, with random-generation evaluation seeds 0, 1, and 2.

## Validation Stages

Floating-point maxima in this table are rounded to the digits shown; raw report artifacts retain full precision.

| Stage | Scope                                      | Purpose                                                                                                                                                                                                                                                                                                                                         |
| ----- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Parameter count, checkpoint key map, vocabularies, bin boundaries, initializers, optimizer and clipping constants, cadence, and effective-behavior facts of the original run.                                                                                                                                                                   |
| S1    | Fixed-batch pre-optimizer trace parity     | One real 1024-layout batch: encoder activations, posterior, logits, per-field losses, KL, L2, and total loss.                                                                                                                                                                                                                                   |
| S2    | One optimizer-step parity                  | Gradients, per-tensor clipped gradients, Adam moments, step count, post-step parameters, and batch-norm moving statistics.                                                                                                                                                                                                                      |
| S3    | Short deterministic multi-batch run        | The synchronized diagnostic checks 49 steps across 69 tensors, for 3,381 gradient comparisons and update checks. The independent held-out normal-mode run passed under calibrated limits; see [Held-out Validation Results](#held-out-validation-results). Natural-trajectory loss differences and direct/arbitrated counts remain report-only. |
| S4    | Deterministic loader stream                | Record-level equality, vocabulary/lookups, and production DataModule/sampler/loader replay of exported train, validation, and test IDs, encoded fields, masks, and padding widths. The check does not equate native shuffle order across frameworks.                                                                                            |
| S5    | Full-run statistical comparison            | RICO complete at training-seed n=3 per system; all three primary metrics pass the fixed comparison rule. Crello model code is available, but no full run or quality result is claimed.                                                                                                                                                          |

## Stage Evidence

An independent, from-scratch held-out run in normal mode passed all five S0–S4 stages on RICO after a fresh clone, submodule initialization, RICO download and preparation, and reference regeneration. It ran on an Intel Xeon CPU at 2.20 GHz without AVX-512; the documented full suite reported 10 passed and 3 warnings in 3,591.26 seconds (59 minutes 51 seconds). The read-back evidence bundle contained all 208 manifest entries. Its result is recorded in the [held-out evidence comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572). Crello CPU unit tests pass. The independent-process S0–S3 vendor comparison passed on a High-RAM CPU using three calibration and three held-out processes; no Crello full-run result is claimed.

| Stage | Command                                                                                                                                                                                                     | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Artifact                                                                                                                             |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s0 -rs` | The held-out run passed static configuration, initialized-state, and forward checks. The exact topology and key-map assertions passed; the 10-output forward maximum was 1.139062e-6 relative to each output maximum (limit 1e-5). The reference-run forward value remains historical in Calibration Results.                                                                                                                                                                                                                                                                                                                                                                                                                                        | https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572                                       |
| S1    | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s1 -rs` | The held-out run passed the fixed-batch trace. The maximum across activation/logit comparisons was 2.087399e-6 relative to tensor maxima (limit 1e-5); the maximum across 12 loss scalars was 1.617341e-7 (limit 1e-6). The reference-run values remain historical in Calibration Results.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572                                       |
| S2    | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s2 -rs` | The held-out one-step command passed. Across four 69-tensor gradient families, maxima for raw gradient, clipped gradient, first moment, and second moment were 6.333509e-5, 6.333466e-5, 6.333450e-5, and 6.393008e-5 (limit 3.5e-4 each; 276 direct and 0 arbitrated). Adam-rule error was 2.7879995e-5 / 3.5e-4; well-conditioned update error was 6.479192e-6 / 1.6e-3; excluded-bias maximum absolute gradient was 7.178460e-8 / 1e-6. Moving-mean error was 1.553235e-7 / 1e-5; the moving-variance assertion also passed at 1e-5, with no separate maximum included in the supplied run summary. See Held-out Validation Results.                                                                                                              | https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572                                       |
| S3    | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s3 -rs` | The held-out synchronized command passed. Across 49 steps, total-loss maximum was 3.044964e-7 / 1.2e-6; 3,268 of 3,381 gradient comparisons passed directly at 3.5e-4 and 113 were arbitrated, with maximum float64 error 7.491022e-4 / shared limit 4.1e-3. Adam-rule maximum was 2.973134e-4 / 5.1e-4; well-conditioned update maximum was 1.698453e-3 / 6.1e-3; running-variance maximum was 8.067037e-8 / 1e-5.                                                                                                                                                                                                                                                                                                                                  | https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572                                       |
| S4    | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s4 -rs` | The held-out command passed exact record and production-loader stream replay. It covered 45,222 / 5,584 / 5,623 train/validation/test records; 24 record gates and 427 stream gates were true. Exact equality remained the criterion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6052063572                                       |
| S5    | `uv run --package canvas-vae --extra training traingen fit --config models/canvas-vae/configs/training/canvas_vae_rico.yaml --trainer.devices=1`                                                            | The launch manifest records the corrected-run plan, not run completion. The completed comparison, source hashes, and run-log digests are recorded in `.cache/canvas-vae/full-run/rico/s5_statistical_comparison.v2.json` and `.cache/canvas-vae/full-run/rico/conversion-chain-review/SHA256SUMS`; the hash-verified segment-002 training, conversion, and evaluation logs record the package run chain and checkpoint gate. Evaluation-path parity passes; all three primary S5 metric decisions pass for RICO. The held-out run passed the first four model-parity stages and the data comparison stage, so full-run reproduction is claimed for RICO only. Crello S0–S3 model parity and S4 data equality pass; full-run results are not claimed. | `.cache/canvas-vae/full-run/rico/manifest.json; evaluation-path-parity: .cache/canvas-vae/full-run/rico/evaluation_path_parity.json` |

### Crello CPU S0–S3 model evidence and S4 data evidence

The CPU model comparison used package commit [`855d1a08ba2946d146a3b2703365334ec967ccfc`](https://github.com/creative-graphic-design/design-generators/commit/855d1a08ba2946d146a3b2703365334ec967ccfc) and vendor commit [`bc1e2072ba3a253f1b099e8b0c604f6051e787da`](https://github.com/CyberAgentAILab/canvas-vae/tree/bc1e2072ba3a253f1b099e8b0c604f6051e787da). S0 static configuration, field dimensions, parameter mapping, and initialized state passed. Three independent calibration processes and three independent held-out processes each exited 0. The calibration aggregate reports `calibration_complete: true`, `incomplete: false`, and no static errors; the held-out report has empty `static_errors` and `heldout_errors` arrays. The pre-run calibration plan fixed three distinct batches; each calibration process started from the prescribed initial state with one selected batch. Limits were frozen before canonical test batches were evaluated in independent held-out processes. The CPU rerun command and fixture digest inputs are in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md). Crello S5 full-run training and trained-model evaluation are not claimed.

The calibration rule is `limit = max(L, ceil2(1.5 × M))`, where `M` is the maximum for that metric across three independent calibration processes and `ceil2` rounds upward to two significant digits. For reconstruction and layout scores in `[0, 1]`, `L = 2 × 2^-24 = 1.1920928955078125e-7`; attention key-projection bias maximum-absolute gradients use `L = 1e-6`; all other metrics use `L = 0`. The table values are rounded to six significant digits; raw reports retain full precision.

| Metric | Report comparison | Frozen limit | Held-out maximum |
| --- | --- | ---: | ---: |
| `s1_eval_outputs` | `max_rel_to_max` | `1.8e-06` | `1.23375e-06` |
| `s1_eval_posterior_mean` | `max_rel_to_max` | `1.1e-06` | `6.58826e-07` |
| `s1_l2_regularization` | `max_rel_to_max` | `0` | `0` |
| `s1_layout_metrics/layout_acc` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_layout_metrics/layout_miou` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_posterior_statistics` | `max_rel_to_max` | `1.5e-06` | `1.08983e-06` |
| `s1_reconstruction_losses` | `max_rel_to_max` | `3.3e-07` | `1.61849e-07` |
| `s1_reconstruction_metrics/canvas_height` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/canvas_width` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/category` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/color` | `max_rel_to_max` | `1.19209e-07` | `1.49012e-08` |
| `s1_reconstruction_metrics/format` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/group` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/height` | `max_rel_to_max` | `1.19209e-07` | `5.16191e-08` |
| `s1_reconstruction_metrics/image_embedding` | `max_rel_to_max` | `1.19209e-07` | `0` |
| `s1_reconstruction_metrics/left` | `max_rel_to_max` | `1.19209e-07` | `3.15458e-08` |
| `s1_reconstruction_metrics/opacity` | `max_rel_to_max` | `1.19209e-07` | `5.38999e-09` |
| `s1_reconstruction_metrics/top` | `max_rel_to_max` | `1.19209e-07` | `5.16191e-08` |
| `s1_reconstruction_metrics/total` | `max_rel_to_max` | `2.5e-07` | `1.34394e-07` |
| `s1_reconstruction_metrics/type` | `max_rel_to_max` | `1.19209e-07` | `6.52936e-08` |
| `s1_reconstruction_metrics/width` | `max_rel_to_max` | `1.19209e-07` | `3.04904e-08` |
| `s1_training_outputs` | `max_rel_to_max` | `2.5e-06` | `2.2122e-06` |
| `s1_training_total_loss` | `max_rel_to_max` | `2.4e-07` | `2.25978e-07` |
| `s2_clipped_gradients` | `norm_rel` | `0.00065` | `0.000118793` |
| `s2_first_moments` | `norm_rel` | `0.00065` | `0.000118793` |
| `s2_gradients` | `norm_rel` | `0.00065` | `0.000118795` |
| `s2_second_moments` | `norm_rel` | `0.00074` | `0.000150907` |
| `s2_updated_parameters` | `norm_rel` | `0.02` | `0.000835994` |
| `s2_zero_gradients` | `max_abs` | `1.9e-06` | `8.98493e-07` |
| `s3_batch_norm_mean` | `max_rel_to_max` | `5.6e-07` | `2.54592e-07` |
| `s3_batch_norm_variance` | `max_rel_to_max` | `9e-08` | `5.96193e-08` |

The relative-gradient comparison excludes the two attention key-projection biases because softmax invariance makes their exact gradients zero. Their package and vendor maximum absolute gradients are each checked against the calibrated `1.9e-6` bound. Values below are maxima over the three independent processes in each phase; they are rounded to six significant digits.

| Tensor | Frozen limit | Calibration maximum (package / vendor) | Held-out maximum (package / vendor) |
| --- | ---: | ---: | ---: |
| `encoder.blocks.0.attention.k_proj.bias` | `1.9e-06` | `8.54221e-07` / `1.24029e-06` | `6.76489e-07` / `8.98493e-07` |
| `decoder.blocks.0.attention.k_proj.bias` | `1.9e-06` | `6.47306e-08` / `1.27591e-07` | `8.95961e-08` / `1.79745e-07` |

The data comparison passed exact equality for train, validation, and test at data commit `113e54392d18b5f14d106e8bb12a7162abc32c39` against vendor commit `bc1e2072ba3a253f1b099e8b0c604f6051e787da`. The canonical posterior-mean fixture is contiguous little-endian float32 with shape `[114007, 256]`; its array SHA-256 is `fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a` and its manifest SHA-256 is `769f372cf7baca6805c3e2a0e4c605ba7c0b47cc490acee62e736fbb45ec42f6`. The package data tar `package-run-1.tar` has SHA-256 `10babb7f8116cef51d97cb64a66b409a10e458e699862ac97d3024687adc0406`; the original reference tar `original-run-2.tar` has SHA-256 `1e3f4b0fa006aa135f35876950d5c7c5cf95ec8f7a5f74655e7ff4d025727596`. The public source archive `crello-dataset-v1.zip` has SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e`.

The accepted model report archive contains 13 report files and has SHA-256 `27312cd8d8568f08f555ebd57e4554617cb4c353b307e63d790d7a71d956ab2f`. Its frozen `limits.json` has SHA-256 `c27ab5abd5d79da6cba38edbc9116f5f9b637b9dffc2d6dc4e6ad40d9f8269ec`. The [PR evidence record](https://github.com/creative-graphic-design/design-generators/pull/458) contains the same run summary and hashes.

Earlier attempts remain historical and do not alter the accepted result:

- A shared-trajectory held-out attempt had 116 failures across batches 0/1/2 = 0/45/71 because all batches reused one changing model and optimizer state instead of independent processes.
- A standard-CPU attempt was killed for running out of memory and did not produce a result.
- A spliced opacity result mixed batches 0 and 1 from one run with batch 2 from another, so it was not one verifiable held-out run.
- The `8af2b26` relative comparison failed on key-projection biases whose exact gradients are zero by softmax invariance; those tensors are now checked with absolute gradients.
- The `65317ab` run's fixed `1e-6` key-bias bound failed calibration; the accepted run uses the calibrated maximum-absolute-gradient bound with the same floor.

## Calibration Results

Each recalibrated numerical threshold uses `max(old_limit, upward_round_to_two_significant_figures(1.5 × maximum_observed))`. The calibration set includes the archived reference run, earlier independent reruns for the stages they completed, and new independent from-scratch calibration runs 1, 3, and 4. New calibration run 2's numeric reports were not retained and no values from it were used. The three retained new reports record an Intel Xeon CPU at 2.20 GHz with AVX-512 absent. Values below are historical calibration measurements; the independent held-out normal-mode results and maxima are recorded in [Held-out Validation Results](#held-out-validation-results).

| Quantity                             | Population, unit, and gate status                                                                                                                    | Old limit | Maximum observed and run                                                                  | New limit                               |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------- | --------------------------------------- |
| S0 output forward                    | 10 outputs from one 1,024-layout batch per run; maximum relative-to-max difference; asserted                                                         | 1e-5      | 1.266045e-6; reference run                                                                | 1e-5                                    |
| S1 activations and logits            | One exact encoder-context tensor and 14 relative-to-max comparisons from one 1,024-layout batch per run; asserted                                    | 1e-5      | 2.916977e-6; independent rerun 1                                                          | 1e-5                                    |
| S1 scalar losses                     | 12 scalars from one 1,024-layout batch per run; maximum relative-to-max difference; asserted                                                         | 1e-6      | 1.684800e-7; independent rerun 3                                                          | 1e-6                                    |
| S2 raw gradients                     | 69 trainable tensors per one-step run; relative L2; asserted direct comparison                                                                       | 1.8e-4    | 2.278771e-4; calibration run 1                                                            | 3.5e-4                                  |
| S2 clipped gradients                 | 69 trainable tensors per one-step run; relative L2; asserted direct comparison                                                                       | 1.8e-4    | 2.277752e-4; calibration run 1                                                            | 3.5e-4                                  |
| S2 first Adam moments                | 69 trainable tensors per one-step run; relative L2; asserted direct comparison                                                                       | 1.8e-4    | 2.277754e-4; calibration run 1                                                            | 3.5e-4                                  |
| S2 second Adam moments               | 69 trainable tensors per one-step run; relative L2; asserted direct comparison                                                                       | 2e-4      | 2.301702e-4; independent rerun 2                                                          | 3.5e-4                                  |
| S2 Adam update rule                  | 69 tensor updates, checked for each system; relative L2; asserted                                                                                    | 3.5e-4    | 2.829523e-5; independent rerun 1                                                          | 3.5e-4                                  |
| S2 well-conditioned update           | 69 tensor updates; cross-system relative L2 on elements where `sqrt(v) >= 1e-5`; asserted                                                            | 1.6e-3    | 2.623599e-5; independent rerun 2                                                          | 1.6e-3                                  |
| S2 excluded-bias gradient            | Two key-projection bias tensors × two systems; maximum absolute gradient; asserted                                                                   | 1e-6      | 6.600749e-8; reference run                                                                | 1e-6                                    |
| S2 batch-normalization statistics    | Moving mean and variance, 256 values each; maximum relative-to-max difference; asserted                                                              | 1e-5      | 2.608958e-7; calibration run 1                                                            | 1e-5                                    |
| Shared S2/S3 float64 arbitration     | Only comparisons exceeding the pre-calibration direct limit; both systems compared with the same float64 step; relative L2; asserted secondary check | 2.7e-3    | 2.696621e-3; calibration run 1                                                            | 4.1e-3                                  |
| S3 synchronized total loss           | 49 scalar comparisons per complete synchronized run; relative difference; asserted                                                                   | 1e-6      | 7.353933e-7; calibration run 3                                                            | 1.2e-6                                  |
| S3 direct-gradient routing prefilter | 3,381 cross-system gradient comparisons per complete run; relative L2; asserted routing branch                                                       | 1.8e-4    | 2.702790e-3; calibration run 1 (observed maximum, not used to recalibrate this threshold) | 3.5e-4, inherited from S2 raw gradients |
| S3 Adam update rule                  | 3,381 tensor updates, checked for each system; relative L2; asserted                                                                                 | 3.5e-4    | 3.350313e-4; independent rerun 3                                                          | 5.1e-4                                  |
| S3 well-conditioned update           | 3,381 tensor updates; cross-system relative L2 on elements where `sqrt(v) >= 1e-5`; asserted                                                         | 1.6e-3    | 4.037209e-3; calibration run 1                                                            | 6.1e-3                                  |
| S3 running variance                  | 49 comparisons of 256 values per complete synchronized run; maximum relative-to-max difference; asserted                                             | 1e-5      | 8.221208e-8; independent rerun 3                                                          | 1e-5                                    |
| S2 learning-rate comparison          | One-step package vs original learning rate; relative error; explicitly excluded from recalibration and report-only                                   | 1e-7      | 4.749745e-8; calibration run 4                                                            | 1e-7 (unchanged)                        |
| S4 record and stream equality        | All RICO train, validation, and test records and production-loader replay batches; exact equality; asserted                                          | exact     | No numeric tolerance; calibration reports record equality                                 | exact (unchanged)                       |

The synchronized direct-gradient threshold uses the recalibrated one-step raw-gradient limit of `3.5e-4`. Under this pass rule, a comparison passes if its direct relative-L2 error is at most the direct threshold `D`, or if it exceeds `D` and both systems' errors against the same float64 step are at most the shared arbitration limit `R`. The literal formula would set this routing prefilter to `4.1e-3`, leaving no comparisons in the calibration set for arbitration; retaining `D = 3.5e-4` deviates from the formula in the stricter direction, because lowering `D` can only reject more. The observed `2.702790e-3` direct maximum is therefore report-only, and comparisons above `3.5e-4` must pass arbitration at `R = 4.1e-3`. This single arbitration limit is shared by S2 and S3: S3 data raised it to `4.1e-3`, while S2 arbitrated errors were at most `2.29e-4`.

Direct/arbitrated counts and the maximum float64 error are report-only, not additional gates. At the pre-calibration direct thresholds, the retained one-step reports recorded 263 direct / 13 arbitrated comparisons in calibration run 1, 276 / 0 in calibration run 3, and 276 / 0 in calibration run 4, out of 276 gradient-family comparisons. The synchronized reports recorded 2,993 / 388 in the reference run, 2,946 / 435 in independent rerun 3, 2,940 / 441 in calibration run 1, 2,773 / 608 in calibration run 3, and 2,842 / 539 in calibration run 4, out of 3,381 gradient comparisons. Earlier independent rerun 1 completed only the natural trajectory in the repeated-run stage; independent rerun 2 completed through the one-step stage. These counts describe the original routing threshold and are not expected to remain the same under the recalibrated threshold.

## Held-out Validation Results

The independent, from-scratch RICO run used normal mode on a fresh clone after submodule initialization, RICO download and preparation, and reference regeneration. It ran on an Intel Xeon CPU at 2.20 GHz without AVX-512. All five S0–S4 stage commands passed, and the documented full suite reported 10 passed and 3 warnings in 3,591.26 seconds (59 minutes 51 seconds). The read-back evidence bundle contained all 208 manifest entries. A later documentation/comment-only update did not change executable checks or limits, so this result applies to the current PR head. The table reports the held-out maximum and its asserted limit; reference and calibration-set maxima remain historical in [Calibration Results](#calibration-results).

| Quantity                                      | Population and unit                                                                                           | Held-out maximum or result                                         | Asserted limit                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------- |
| S0 static configuration and initialized state | Model topology, parameter count, checkpoint keys, vocabularies, bins, optimizer and cadence; exact assertions | Passed                                                             | Exact equality                           |
| S0 evaluation forward                         | 10 output tensors; maximum relative-to-max difference                                                         | 1.139062e-6                                                        | 1e-5                                     |
| S1 activations and logits                     | One exact context tensor and 14 relative-to-max comparisons from one 1,024-layout batch                       | 2.087399e-6                                                        | 1e-5                                     |
| S1 losses                                     | 12 scalar loss comparisons; relative-to-max difference                                                        | 1.617341e-7                                                        | 1e-6                                     |
| S2 raw gradients                              | 69 trainable tensors; relative L2; 276 total comparisons across four gradient families                        | 6.333509e-5; all 276 direct, 0 arbitrated                          | 3.5e-4                                   |
| S2 clipped gradients                          | 69 trainable tensors; relative L2                                                                             | 6.333466e-5                                                        | 3.5e-4                                   |
| S2 first Adam moments                         | 69 trainable tensors; relative L2                                                                             | 6.333450e-5                                                        | 3.5e-4                                   |
| S2 second Adam moments                        | 69 trainable tensors; relative L2                                                                             | 6.393008e-5                                                        | 3.5e-4                                   |
| S2 Adam update rule                           | 69 tensor updates per system; relative L2                                                                     | 2.7879995e-5                                                       | 3.5e-4                                   |
| S2 well-conditioned update                    | 69 tensors; cross-system relative L2 where `sqrt(v) >= 1e-5`                                                  | 6.479192e-6                                                        | 1.6e-3                                   |
| S2 excluded-bias gradient                     | Two key-projection bias tensors × two systems; maximum absolute gradient                                      | 7.178460e-8                                                        | 1e-6                                     |
| S2 moving mean                                | 256 values; relative-to-max difference                                                                        | 1.553235e-7                                                        | 1e-5                                     |
| S2 moving variance                            | 256 values; relative-to-max difference                                                                        | Passed; separate maximum not included in the supplied run summary  | 1e-5                                     |
| S3 total loss                                 | 49 scalar comparisons across the synchronized run; relative difference                                        | 3.044964e-7                                                        | 1.2e-6                                   |
| S3 synchronized gradients                     | 3,381 cross-system comparisons; direct relative L2 or float64 arbitration                                     | 3,268 direct; 113 arbitrated; maximum arbitrated error 7.491022e-4 | Direct 3.5e-4; shared arbitration 4.1e-3 |
| S3 Adam update rule                           | 3,381 tensor updates per system; relative L2                                                                  | 2.973134e-4                                                        | 5.1e-4                                   |
| S3 well-conditioned update                    | 3,381 tensors; cross-system relative L2 where `sqrt(v) >= 1e-5`                                               | 1.698453e-3                                                        | 6.1e-3                                   |
| S3 running variance                           | 49 comparisons of 256 values; relative-to-max difference                                                      | 8.067037e-8                                                        | 1e-5                                     |
| S4 record equality                            | 45,222 / 5,584 / 5,623 train/validation/test records; 24 gates                                                | Exact; all 24 gates passed                                         | Exact equality                           |
| S4 production-loader stream replay            | 427 stream gates across train, validation, and test replay; exact equality                                    | Exact; all 427 gates passed                                        | Exact equality                           |

The full-run reproduction claim covers RICO only. The independent-process Crello S0–S3 vendor comparison passed on a High-RAM CPU using three calibration and three held-out processes; no Crello full-run training result is claimed.

## Reproduction Results

The results below record the completed RICO S5 evaluation and statistical comparisons. Together with the independent held-out normal-mode S0–S4 pass documented in [Held-out Validation Results](#held-out-validation-results), they support a full-run reproduction claim for RICO only. [Calibration Results](#calibration-results) preserves the historical calibration measurements and bounds. The independent-process Crello S0–S3 vendor comparison passed on a High-RAM CPU using three calibration and three held-out processes; no Crello full-run result is claimed.

| Dataset  | System   | Status                                        | Seed scope          | Primary metrics                                                                                     | Loss evidence                                                                                       | Artifact summary                                                    |
| -------- | -------- | --------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `rico`   | original | `s5-practical-reproduction`                   | `training-seed n=3` | Sreconst 93.049884 ± 0.560324 pp; layout mIoU 56.103125 ± 1.480075 pp; Sgen 92.852189 ± 0.020325 pp | Logged loss history for completed epochs 20–500 is report-only; see the run-level S5 evidence JSON. | `.cache/canvas-vae/full-run/rico/s5_statistical_comparison.v2.json` |
| `rico`   | package  | `s5-practical-reproduction`                   | `training-seed n=3` | Sreconst 93.086390 ± 0.173882 pp; layout mIoU 56.174088 ± 0.460258 pp; Sgen 92.740042 ± 0.155047 pp | Logged loss history for completed epochs 20–500 is report-only; see the run-level S5 evidence JSON. | `.cache/canvas-vae/full-run/rico/s5_statistical_comparison.v2.json` |
| `crello` | original | `not-yet-run (S0–S4 passed; S5 not claimed)` | none                | not claimed                                                                                         | not applicable                                                                                      | not applicable                                                      |
| `crello` | package  | `not-yet-run (S0–S4 passed; S5 not claimed)` | none                | not claimed                                                                                         | not applicable                                                                                      | not applicable                                                      |

The S5 population is one final 500-epoch checkpoint per training seed, with n=3 per system. Reconstruction and layout mIoU use 5,623 RICO test layouts per checkpoint. Each Sgen run-level value is the arithmetic mean of `random_total` over evaluation seeds 0, 1, and 2, each evaluated on 5,623 generated layouts. Primary metrics are score fractions multiplied by 100 and reported in percentage points (pp). The run-level values below are the asserted inputs to the decision rule.

| System   | Training seed | Sreconst (pp) | layout mIoU (pp) | Sgen (pp) |
| -------- | ------------- | ------------- | ---------------- | --------- |
| original | 0             | 92.428809     | 54.412419        | 92.843439 |
| original | 1             | 93.203380     | 56.732244        | 92.837704 |
| original | 2             | 93.517462     | 57.164711        | 92.875423 |
| package  | 0             | 92.937155     | 55.741682        | 92.838477 |
| package  | 1             | 93.044681     | 56.122697        | 92.561316 |
| package  | 2             | 93.277333     | 56.657885        | 92.820334 |

The per-system sample standard deviation is `s = sqrt(sum((x_i - mean)^2) / (n - 1))`, with n=3 training seeds. The pooled standard deviation is `sqrt(((n_o - 1)s_o^2 + (n_p - 1)s_p^2) / (n_o + n_p - 2) )`. PASS when `|mean_p - mean_o| <= 2 × pooled SD`; otherwise CHECK. This rule was fixed before the corrected package results existed. For background on that timing, see the [CanvasVAE checkpoint and S5 decision history](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6020167348). Means, sample SDs, pooled SDs, differences, and bounds below use percentage-point units.

| Primary metric | Original mean ± sample SD (pp) | Package mean ± sample SD (pp) | Pooled SD (pp) | Absolute difference (pp) | 2 × pooled SD (pp) | Decision |
| -------------- | ------------------------------ | ----------------------------- | -------------- | ------------------------ | ------------------ | -------- |
| Sreconst       | 93.049884 ± 0.560324           | 93.086390 ± 0.173882          | 0.414848       | 0.036506                 | 0.829696           | PASS     |
| layout mIoU    | 56.103125 ± 1.480075           | 56.174088 ± 0.460258          | 1.096006       | 0.070963                 | 2.192012           | PASS     |
| Sgen           | 92.852189 ± 0.020325           | 92.740042 ± 0.155047          | 0.110573       | 0.112147                 | 0.221146           | PASS     |

The three corrected package logs each record the asserted final-checkpoint gate `epoch=499` and `global_step=22,500`; checkpoint metadata confirms the final completed epoch. The asserted checkpoint-to-conversion check has an intact hash chain through evaluation; an independent review compared all 73 converted model tensors, measured in parameter-value units, and found a maximum absolute difference of 0. Across the three corrected package runs, 25 validation observations per run placed the best-checkpoint epoch indices at 399, 439, and 479; these epoch indices are report-only selection metadata, and none of those best-validation checkpoints enters S5. All pre-correction package outputs are excluded: the earlier best-validation checkpoint could be copied into a final-checkpoint slot, so the training setup now saves best-validation and latest checkpoints separately and rejects package checkpoints unless the final epoch and update count match. For background on these changes, see the [checkpoint and S5 decision history](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6020167348).

### Report-only Metrics

The epoch-500 `val_total_score` is a report-only validation score over 6,144 presentations per checkpoint (5,584 unique layouts and 560 repeated layouts), in native score-fraction units on [0, 1]. Logged training-loss values at completed epochs 20–500 are report-only; the JSON records each run and the per-system mean and sample SD across three training seeds in logged loss units. Secondary evaluation metrics are also report-only: all 19 secondary fields are recorded for each of the six runs and as per-system mean and sample SD in the S5 evidence JSON. Score-valued secondary metrics use native fractions on [0, 1]; other values retain their evaluator-native units.

| System   | Training seed        | Epoch-500 `val_total_score` |
| -------- | -------------------- | --------------------------- |
| original | 0                    | 0.927000                    |
| original | 1                    | 0.933300                    |
| original | 2                    | 0.937100                    |
| original | n=3 mean ± sample SD | 0.932467 ± 0.005101         |
| package  | 0                    | 0.931557                    |
| package  | 1                    | 0.931643                    |
| package  | 2                    | 0.935319                    |
| package  | n=3 mean ± sample SD | 0.932840 ± 0.002148         |

The original `test_results.json` reconstruction values provide an asserted evaluator cross-check: 12 dimensionless reconstruction metrics for each of three original training seeds, 36 scalar comparisons total, all pass `abs(package evaluator value − original value) <= 1e-6 + 1e-5 × abs(original value)`. The maximum absolute gap across those 36 values is 1.2611e-7. The original unseeded random-generation values are report-only: 10 random metrics per original seed, 30 comparisons total, with no shared latents and no tolerance applied.

### Comparison Scope

Comparison Scope records the evaluation setup for each dataset comparison. The evaluator, test split, checkpoint-selection rule, and sample count below apply to both systems.

| Dataset  | System | Evaluator                                                                                 | Test split  | Checkpoint-selection rule                                                                                                                                                                                                         | Sample count                                                                             |
| -------- | ------ | ----------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `rico`   | both   | `models/canvas-vae/scripts/evaluate.py` (CPU, batch size 1,024, evaluation seeds 0, 1, 2) | RICO test   | Train through epoch 500. For package runs, require checkpoint metadata at epoch 499 and global step 22,500; for original runs, use the final checkpoint. Convert the selected checkpoint and evaluate it with the same evaluator. | 5,623 reconstruction layouts per checkpoint; 5,623 generated layouts per evaluation seed |
| `crello` | both   | `models/canvas-vae/scripts/evaluate_crello.py` (CPU, configurable batch size and seeds)   | Crello test | no selected checkpoint; no full-run result is claimed                                                                                                                                                                             | no evaluation result is claimed                                                          |

The Comparison Scope `Evaluator` cell names the evaluator that the manifest's `evaluator_command` runs. The `Checkpoint-selection rule` cell summarizes the manifest's rule in portable terms.

### Paper Table 2 Context

These values are from Table 2 of the [CanvasVAE ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html); they are context only and are not part of the asserted S5 comparison. They are reported in percentage points as shown in the paper.

| Metric      | Paper Table 2 (pp) |
| ----------- | ------------------ |
| Sreconst    | 94.35              |
| layout mIoU | 60.42              |
| Sgen        | 93.90              |

### S5 Evidence Hashes

The SHA-256 values for all 27 source inputs are recorded per file in `.cache/canvas-vae/full-run/rico/s5_statistical_comparison.v2.json`: six evaluation JSON files, six artifact-hash lists, three original test-result files, six training logs, three package checkpoint-gate logs, and three package metric CSV files. Those 27 file hashes were verified against the local inputs. The JSON also records the comparison script hash.

| Evidence artifact               | SHA-256                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| S5 comparison script            | `ce4e5f1ff0d9ea44e1306af95f4881eeb7a25a6631733d5c7eb703c0b9a78401` |
| S5 result JSON                  | `35f43f18c5bef9ed40c0e58b1eeba6ff0dbdb0b8d542a925aa4c9a4b70664cd6` |
| Launch manifest                 | `b52d889bfc1b950237a19a512a8501b2aaac49c9b438e6f46c0cfcc8e9bcad61` |
| Evaluation-path parity artifact | `2934dac2a4427d001956eea548359e4466c10e3306b9f05eb2575df8ffac6c2b` |

## Regeneration Metadata

The complete-data original-code references were generated at commit [`374e56e`](https://github.com/creative-graphic-design/design-generators/commit/374e56ed7f23364cacad17d9648021631d5e9b55) from the public RICO semantic-annotation archive. Original preprocessing used Apache Beam 2.76.0 with `FnApiRunner`, one worker, and `in_memory` mode, which retained the full data set. The reference generation and agreement checks ran on Google Colab CPU with Python 3.11, TensorFlow 2.15.1, PyTorch 2.13.0, Lightning 2.6.5, and Transformers 5.13.0, using fp32 and `NVIDIA_TF32_OVERRIDE=0`; the original reference used TensorFlow seed 0 with op determinism and dropout 0, and the package side injected the original posterior noise. The S2 and S3 gradient checks use a direct relative-L2 threshold of `3.5e-4`; above it, both systems must be within `4.1e-3` relative L2 of the same saved-state float64 step. The final S0-S4 run and TF 2.11.1 control result are summarized in the [S0–S4 evidence comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5963001129). The earlier [S0–S2 summary](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5948477063) and [S3–S4 summary](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5950282344) provide optional background on the initial evidence.

```text
.cache/canvas-vae/data/semantic_annotations.zip      RICO archive
.cache/canvas-vae/data/rico/                         package splits and vocabulary counts
.cache/canvas-vae/original/data/rico/                original Beam TFRecords, vocabulary.json, count.json
.cache/canvas-vae/reference/trace/                   static.json, initial weights, step-0 trace, 50-step trajectory, per-step synchronized state
.cache/canvas-vae/reference/trace-repeat/            repeated original trajectory for the run-to-run envelope
.cache/canvas-vae/reference/stream/                  original records and stream orders
.cache/canvas-vae/reference/reports/                 measured differences per stage
```

The original record function emits all eligible screens, but the multi-threaded Beam runner can drop a run-dependent handful. Generating the original TFRecords with one worker in `in_memory` mode retains all eligible records; S4 confirms exact content-hash and per-field equality with the package for all three splits.

## Training Commands

Run the staged parity checks after generating references with [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/REPRODUCING.md).

```bash
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -rs
```

Train on RICO.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package canvas-vae --extra training traingen fit --config models/canvas-vae/configs/training/canvas_vae_rico.yaml --trainer.devices=1
```

Train the original code with a seed.

Original GPU training and evaluation set `TF_ENABLE_ONEDNN_OPTS=0` and `NVIDIA_TF32_OVERRIDE=0`. The S0–S4 references ran on CPU with oneDNN at its default. With the default setting, the original A100 smoke stopped before its first batch with `NotFoundError: No registered '_MklLayerNorm' OpKernel for 'GPU' devices compatible with node {{node custom_model/layer_normalization/add}}` ([TensorFlow issue #62607](https://github.com/tensorflow/tensorflow/issues/62607)); with `TF_ENABLE_ONEDNN_OPTS=0`, one 1,024-layout training batch ran on `/device:GPU:0`. Inference: oneDNN selects CPU kernels, so this setting should not change GPU-side calculations. We did not verify that with a paired GPU numerical comparison because the default-setting run failed before training.

```bash
TF_ENABLE_ONEDNN_OPTS=0 NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/train_original.py --seed 0 --job-dir .cache/canvas-vae/original/jobs/rico-seed0
```

Convert an original TensorFlow checkpoint prefix. Original `final.ckpt` and reference `initial.ckpt` are TensorFlow prefixes and need the `convert` extra.

```bash
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/original/jobs/rico-seed0/checkpoints/final.ckpt --vocabulary .cache/canvas-vae/original/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted-trained/rico-original-seed0
```

Convert a package Lightning checkpoint. A `last.ckpt` generated by the current configs uses PyTorch loading and needs no TensorFlow extra; for S5, use only a file that passes the final-checkpoint gate.

```bash
uv run --package canvas-vae models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/training-runs/rico/lightning_logs/version_0/checkpoints/last.ckpt --vocabulary .cache/canvas-vae/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted-trained/rico-package
```

Evaluate and smoke-test local loading.

```bash
uv run --package canvas-vae models/canvas-vae/scripts/evaluate.py --checkpoint .cache/canvas-vae/converted-trained/rico --output .cache/canvas-vae/evaluation/rico-package-seed0.json
uv run --package canvas-vae models/canvas-vae/scripts/smoke_from_pretrained.py --checkpoint .cache/canvas-vae/converted-trained/rico
```
