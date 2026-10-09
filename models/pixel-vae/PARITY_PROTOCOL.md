# PixelVAE CPU parity protocol

This protocol defines the numerical comparisons before calibration. It covers the untrained PixelVAE state and does not claim trained embedding quality.

## Fixed inputs and state

Use the original TensorFlow PixelVAE at vendor commit `bc1e2072ba3a253f1b099e8b0c604f6051e787da`, configured for 256 × 256 RGBA uint8 input, a 256-dimensional posterior, quantization factor 4, KL weight 100, L2 weight 1e-6, and float32. Set TensorFlow seed 0 and deterministic CPU execution. Copy the same deterministic untrained state into PyTorch through the strict converter and record the sorted encoder-state SHA-256. Do not change the state, examples, metric definitions, or comparison code between repeat runs.

Sort documents by UTF-8 bytewise `document_id` within each split, preserve source element order, and deduplicate exact PNG bytes by SHA-256. The diagnostic report uses the canonical held-out S1 pair and the first eight distinct train pairs; it is report-only and its values do not set limits. Calibration repeat `r` uses document-stage PNG indexes `[2(r−1), 2r)` for S1 and filtered training-stage PNG indexes `[6(r−1), 6r)` for S2/S3, with zero-based indexes over their respective canonical train lists. For S4, repeat `r` uses the `r`th unique canonical train PNG of each element type. Thus every calibration run uses different train inputs for each numerical stage. S1 and S2 use a batch of two, and S3 uses three consecutive batches of two. Held-out validation uses the first two canonical test document-stage IDs for S1, the first six test training-filter IDs for S2/S3, and the first canonical test PNG of each type for S4. Both implementations receive the same PNG bytes decoded to exact-size RGBA uint8 values; PyTorch receives those pixels scaled to float32 `[0, 1]` in channels-first order.

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

The registered calibration ran three CPU processes on the same two train images. CPU execution is deterministic, so the three runs were bit-identical (S1 posterior mean `6.355e-11`, log variance `5.270e-11`) and the maximum was a single sample. The held-out check on canonical test inputs then exceeded the frozen limits on two S1 metrics, identically at two commits (`2b0c16e` and `b755f92`): posterior mean `9.879e-11` vs `9.6e-11` and log variance `8.310e-11` vs `8.0e-11`. Both runs stay recorded as history.

From now on, each of three fresh CPU calibration processes uses different train inputs. Take the canonical train PNG list (exact-byte deduplicated, sorted by UTF-8 `document_id`, source element order kept). Repeat `r` uses document-stage indexes `[2(r-1), 2r)` for its S1 pair and training-filter indexes `[6(r-1), 6r)` for its S2/S3 examples. Every selected image ID is recorded before outputs are compared. All runs use the same deterministic untrained TensorFlow state copied to PyTorch.

For each metric, the new limit is `max(L, ceil2(1.5 × M))`. `L` is the originally registered floor, not a limit frozen by the failed runs. `M` is the maximum over the three distinct-input runs, and `ceil2` rounds up to two significant figures. Diagnostics are report-only. Each calibration process writes its selected IDs to `calibration/repeat-r.inputs.json` before comparison; the limit utility verifies those IDs against the completed metric record. Limits are frozen after calibration, then one fresh normal-mode CPU check runs on canonical test inputs. A held-out failure requires an implementation fix and a new calibration plus held-out sequence; it never authorizes editing limits.

The runner reads `L` from [`parity-limit-floors.json`](./parity-limit-floors.json), records its provenance and the historical failures in `calibration/limits.json`, and applies the formula in [`testing.py`](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/src/pixel_vae/testing.py).

Run the sequence in one CPU job with separate processes for the diagnostic, each calibration repeat, and held-out validation. Retain the report-only diagnostic, three raw calibration records, and frozen limits under `.cache/pixel-vae/parity/`. Freeze the limits before starting a fresh normal-mode CPU process on held-out test examples and the full v1 source path. The held-out run passes only if all stage metrics are at or below their frozen bounds and every S0/S4 exact check passes. A failed held-out run cannot be repaired by changing limits; a code change requires three new distinct-input calibration repeats and a fresh held-out run. Earlier failed runs remain historical evidence and are not overwritten.

All numerical comparisons use float32. Data images, manifests, embedding fixtures, checkpoints, and raw calibration tensors remain private and are not committed or published publicly. Store required run evidence in the project's private artifact repository. The package-level report may record aggregate counts, digests, limits, and maximum differences.
