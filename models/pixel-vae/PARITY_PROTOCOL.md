# PixelVAE CPU parity protocol

This protocol defines the numerical comparisons before calibration. It covers the untrained PixelVAE state and does not claim trained embedding quality.

## Fixed inputs and state

Use the original TensorFlow PixelVAE at vendor commit `bc1e2072ba3a253f1b099e8b0c604f6051e787da`, configured for 256 × 256 RGBA uint8 input, a 256-dimensional posterior, quantization factor 4, KL weight 100, L2 weight 1e-6, and float32. Set TensorFlow seed 0 and deterministic CPU execution. Copy the same deterministic untrained state into PyTorch through the strict converter and record the sorted encoder-state SHA-256. Do not change the state, examples, metric definitions, or comparison code between repeat runs.

Sort documents by UTF-8 bytewise `document_id` within each split and preserve source element order. S1 covers the first two unique document-stage IDs of any element type; S2 and S3 use the first six unique IDs selected by the original PixelVAE training filter (`imageElement`, `maskElement`, and `svgElement`). S1 and S2 use one fixed batch of two, and S3 uses three consecutive batches of two. Calibration uses train records; the held-out run uses the corresponding first six test IDs, excluded from calibration. Both implementations receive the same PNG bytes decoded to exact-size RGBA uint8 values; PyTorch receives those pixels scaled to float32 `[0, 1]` in channels-first order.

## Metrics and aggregation

`max_abs` is the largest absolute elementwise difference across every compared tensor and example for that metric in one run. Scalar losses use absolute scalar difference. Each tensor must have matching shape and finite values before its difference is measured.

| Stage | Metric key                          | Compared values                                                            |
| ----- | ----------------------------------- | -------------------------------------------------------------------------- |
| S1    | `s1.posterior_mean.max_abs`         | Encoder posterior means.                                                   |
| S1    | `s1.posterior_log_variance.max_abs` | Encoder posterior log variances.                                           |
| S1    | `s1.decoder_logits.max_abs`         | Decoder categorical logits.                                                |
| S1    | `s1.reconstruction_loss.abs`        | Alpha-masked RGB and alpha reconstruction loss.                            |
| S1    | `s1.kl_loss.abs`                    | Unweighted KL divergence.                                                  |
| S1    | `s1.total_loss.abs`                 | Reconstruction, weighted KL, and encoder-head L2 sum.                      |
| S2    | `s2.loss.abs`                       | Fixed-noise training loss before the optimizer update.                     |
| S2    | `s2.gradients.max_abs`              | Every mapped trainable gradient.                                           |
| S2    | `s2.adam_first_moment.max_abs`      | First-moment optimizer state after one update.                             |
| S2    | `s2.adam_second_moment.max_abs`     | Second-moment optimizer state after one update.                            |
| S2    | `s2.parameters.max_abs`             | Every trainable parameter after one update.                                |
| S2    | `s2.batch_norm_state.max_abs`       | Every moving mean and variance after the training forward.                 |
| S3    | `s3.step_losses.max_abs`            | Losses from each fixed short-trace step.                                   |
| S3    | `s3.posterior_means.max_abs`        | Posterior means from each fixed short-trace step.                          |
| S3    | `s3.parameters.max_abs`             | Every trainable parameter after each short-trace step.                     |
| S3    | `s3.batch_norm_state.max_abs`       | Every moving mean and variance after each short-trace step.                |
| S4    | `s4.posterior_means.max_abs`        | Posterior means for one source-ordered PNG of every document element type. |

S0 requires exact configuration values, a complete one-to-one key map, and bitwise equality between each TensorFlow state array and its converted PyTorch tensor after the declared layout transform. PixelVAE S4 verifies the pinned archive digest and document counts, decodes every PNG to report malformed or non-256 × 256 examples by element type, compares exact RGBA pixels and posterior means for one canonical first-occurrence example of each type, and checks the encoder save/load output. The Crello data package owns record identities, categorical fields, masks, full record and manifest order, and fixture row/file hashes; its S4 validates those fields against the same [embedding contract](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/EMBEDDING_CONTRACT.md). The package encoder handles every document-stage type, while the original three-type filter applies only to PixelVAE training records.

## Calibration and held-out pass rule

Run three independent CPU calibration processes from the fixed state. Each process writes one finite, non-negative maximum absolute discrepancy per metric above and the same encoder-state digest. For each metric, freeze `ceil2(1.5 × max(repeat 1, repeat 2, repeat 3))`, where `ceil2` rounds upward to two significant figures. A zero maximum produces a zero bound. The implementation in [`testing.py`](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/src/pixel_vae/testing.py) is the comparison code for this rule.

Before calibration, retain the three raw run records and the resulting frozen limits under `.cache/pixel-vae/parity/calibration/`. Then start a new normal-mode CPU process on held-out test examples and the full v1 source path. The held-out run passes only if all stage metrics are at or below their frozen bounds and every S0/S4 exact check passes. A failed calibration or held-out run cannot be repaired by changing the limits; code changes require three new calibration repeats and a fresh held-out run.

All numerical comparisons use float32. Data images, manifests, embedding fixtures, checkpoints, and raw calibration tensors remain private and are not committed or published publicly. Store required run evidence in the project's private artifact repository. The package-level report may record aggregate counts, digests, limits, and maximum differences.
