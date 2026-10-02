# CanvasVAE Training

This document records package-local training commands, staged reproduction evidence, and per-dataset full-run status for `canvas-vae`. The package trains CanvasVAE on RICO with LightningCLI and checks it against the [original TensorFlow trainer](https://github.com/CyberAgentAILab/canvas-vae) with the six ordered stages S0-S5 of the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md): S0 static configuration and topology, S1 a fixed-batch forward trace, S2 one optimizer step, S3 a multi-step trajectory, S4 the data-loader stream, and S5 a full-run statistical comparison. S0-S4 pass for RICO; S5 has not been run, so trained-checkpoint reproduction is not yet claimed.

Run commands from the repository root. Keep generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts under `.cache/canvas-vae/`.

## Install

```bash
uv sync --package canvas-vae --extra training
```

Install the `vendor` extra only when rerunning original-code parity checks. It adds TensorFlow 2.15.1 and Apache Beam.

```bash
uv sync --package canvas-vae --extra training --extra vendor
```

## Data

| Dataset  | Source                                                                                                                                                                        | Config or path                                                                                                                                                                                                                                                             |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rico`   | [RICO semantic-annotation archive](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip), MD5 `5dd3372e2d99b342958136e394d4de79` | `scripts/prepare_rico.py` writes `.cache/canvas-vae/data/rico/{train,val,test}.jsonl`, `vocabulary.json`, and `count.json`: 45,222 / 5,584 / 5,623 screens after deduplication by content, dropping screens with more than 50 elements, and splitting by MD5 content hash. |
| `crello` | not used                                                                                                                                                                      | Not claimed: Crello needs a separately trained PixelVAE image encoder first.                                                                                                                                                                                               |

## Configs

Training configs live under `models/canvas-vae/configs/training`.

| Config                               | Dataset          | Seed mode                      | Purpose                                                                                                |
| ------------------------------------ | ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `canvas_vae_rico.yaml`               | `rico`           | default (`seed_everything: 0`) | Full run: 500 epochs of 45 batches of 1024, validation every 20 epochs.                                |
| `canvas_vae_rico_deterministic.yaml` | `rico`           | deterministic, dropout 0       | Bounded production-wiring run for step-level checks.                                                   |
| `smoke.yaml`                         | prepared fixture | deterministic                  | CPU wiring smoke with a tiny model; pass `--model.init_args.data_dir` and `--data.init_args.data_dir`. |

## Scheduler and Recipe Notes

The recipe follows the original trainer's effective behavior rather than generic Lightning defaults:

- Adam uses the Keras 2 update (`m += (g - m)(1 - beta1)`, epsilon `1e-7` added after `sqrt(v)` with bias correction folded into the step size) at a constant learning rate of `1e-3`; there is no scheduler, warmup, EMA, or mixed precision.
- Every gradient tensor is clipped to norm 1.0 on its own (Keras `clipnorm`), inside `configure_gradient_clipping`. Lightning's `gradient_clip_val`, a global-norm clip, is rejected.
- The loss is the per-field cross-entropy summed over valid elements and fields, averaged over the batch, plus 16 times the KL divergence averaged over batch and latent dimensions, plus `1e-6` times the squared norm of every dense and embedding weight.
- Batch normalization of the pooled encoder output uses the batch mean and the biased batch variance with moving-average momentum 0.99. Under TensorFlow 2.15 the original `BatchNormalization` would receive the `(batch, elements)` padding mask propagated from the pooled block and fail; the original code was written for TensorFlow 2.3/2.4, whose batch normalization ignored masks. The reference scripts restore that mask-free layer, and S0 records the TensorFlow 2.15 failure.
- Training batches come from one endless stream of shuffled passes with 45 full batches per epoch, so batches straddle pass boundaries. Validation scores 6 full batches of a repeated ordered pass (560 screens twice), and testing scores one ordered pass.
- The checkpoint callback keeps `best.ckpt` by `val/total_score` and `last.ckpt`. The original `evaluate()` scores the final in-memory weights, so the final checkpoint is the one compared.

## Seed Policy

The original code sets no seed; `scripts/train_original.py` calls `tf.keras.utils.set_random_seed(seed)` before the unmodified trainer, and the package configs set `seed_everything`. S1-S4 inject the original's posterior noise into the package and disable dropout on both sides, so they need no cross-framework RNG pairing. S5 targets `training-seed n=3` per system, not seed-paired across frameworks, with random-generation evaluation seeds 0, 1, and 2.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                                                                                       |
| ----- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Parameter count, checkpoint key map, vocabularies, bin boundaries, initializers, optimizer and clipping constants, cadence, and effective-behavior facts of the original run. |
| S1    | Fixed-batch pre-optimizer trace parity     | One real 1024-layout batch: encoder activations, posterior, logits, per-field losses, KL, L2, and total loss.                                                                 |
| S2    | One optimizer-step parity                  | Gradients, per-tensor clipped gradients, Adam moments, step count, post-step parameters, and batch-norm moving statistics.                                                    |
| S3    | Short deterministic multi-batch run        | Natural 50-step trajectory with run-to-run envelopes, plus a bounded `traingen fit` production-wiring run.                                                                    |
| S4    | Deterministic loader stream                | Record-level equality with the original TFRecords, vocabulary counts, and replayed train, validation, and test streams.                                                       |
| S5    | Full-run statistical comparison            | Not run yet.                                                                                                                                                                      |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                     | Artifact                                             | Result                                                                                                                   |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| S0 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s0 -rs` | `.cache/canvas-vae/reference/reports/s0.json` | Pass. 1,558,435 trainable parameters and 512 batch-norm statistics on both sides; the checkpoint key map has no missing or extra keys (`norm3` absent on both sides); lookup sizes, empty-label id 6, float32 bin boundaries, initializers, Adam and clipping constants, 61 L2-regularized tensors, cadence, and steps per epoch (45 / 6 / 6) are equal; source facts hold (final weights scored, no seed, no schedule, warmup, or EMA, cross-epoch batches, wrapping validation, unpatched TensorFlow 2.15 batch norm raises). Eval-mode forward differs by at most 1.2e-6 of each tensor's largest magnitude. |
| S1 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s1 -rs` | `.cache/canvas-vae/reference/reports/s1.json` | Pass. One original 1024-layout batch with injected posterior noise: length embedding exact; encoder block, batch norm, posterior, decoder block, and all logits within 1.3e-6 of each tensor's largest magnitude (limit 1e-5); per-field losses, KL, and L2 within 1.4e-7 relative (limit 1e-6); total loss bit-identical. |
| S2 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s2 -rs` | `.cache/canvas-vae/reference/reports/s2.json` | Pass. Dense gradients, per-tensor clipped gradients, and Adam first moments within 4.9e-5 relative L2 norm (limit 1e-4); second moments within 4.8e-5 (limit 2e-4); first-step updates within 1.2e-2 (limit 5e-2); batch-norm moving mean within 2.1e-7 and moving variance exact; step count 1 and learning rate equal. Attention key-projection bias gradients are zero in exact arithmetic and stay below 4.7e-8 on both sides. |
| S3 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s3 -rs` | `.cache/canvas-vae/reference/reports/s3_synchronized.json` | Bounded numerical pass. The natural 50-step trajectory leaves the one-step loss limit at step 2 and reaches a 3.6e-4 relative loss difference by step 50 (`s3.json`; the original's repeated run differs from itself by up to 1.5e-4 in absolute loss). Synchronized at every optimizer boundary from the original's weights and Adam state, all 49 steps keep the loss within 9.5e-7 relative, updates within 4.1e-2, and gradients within 1e-4 or, for later-step tensors with cancelling sums, both systems within 1.6e-3 of the same step recomputed in float64 (limit 5e-3). Production wiring through `traingen fit` (`s3_wiring.json`): `KerasAdam`, per-tensor clipping, no global clipping, validation logging `val/total_score`, and `best.ckpt` plus `last.ckpt`. |
| S4 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -k s4 -rs` | `.cache/canvas-vae/reference/reports/s4_stream.json` | Pass. Every record of the original TFRecords (45,196 / 5,580 / 5,619) equals the package record with the same content hash field by field, the package additionally keeps the 34 screens the original Beam job dropped, and value counts over the common screens equal the original `vocabulary.json`; the three lookup tables are identical. Replaying the original record order gives byte-identical prepared batches and padding widths for 90 training, 6 validation (560 screens scored twice), and 6 test batches (last batch 499); the original order is reproducible under op determinism. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package canvas-vae --extra training traingen fit --config models/canvas-vae/configs/training/canvas_vae_rico.yaml --seed_everything=<seed> --trainer.devices=1` | `.cache/canvas-vae/full-run/rico/manifest.json` | Not claimed yet: the three original and three package full runs and the real-scale lockstep probe need separately approved GPU time. |

## Reproduction Results

RICO training reproduction is not yet claimed: S0-S2 and S4 pass on CPU in fp32, S3 passes as a bounded numerical result through its synchronized layer, and the S5 full-run comparison has not been run. Crello is not claimed.

| Dataset  | System   | Status                                                        | Seed scope                  | Primary metrics                          | Loss evidence                | Artifact summary                       |
| -------- | -------- | ------------------------------------------------------------- | --------------------------- | ---------------------------------------- | ---------------------------- | -------------------------------------- |
| `rico`   | original | `not-yet-run (creative-graphic-design/design-generators#31)`                                           | `training-seed n=3` planned | Sreconst, layout mIoU, Sgen not measured | 50-step trajectory only (S3) | `.cache/canvas-vae/reference/trace/`   |
| `rico`   | package  | `not-yet-run (creative-graphic-design/design-generators#31)`                                           | `training-seed n=3` planned | Sreconst, layout mIoU, Sgen not measured | 50-step trajectory only (S3) | `.cache/canvas-vae/reference/reports/` |
| `crello` | original | `blocked (needs a separately trained PixelVAE image encoder)` | none                        | not claimed                              | none                         | none                                   |
| `crello` | package  | `blocked (needs a separately trained PixelVAE image encoder)` | none                        | not claimed                              | none                         | none                                   |

### Comparison Scope

Comparison Scope records the evaluation setup for each dataset comparison. Use one row with `System` set to `both` when the evaluator, test split, checkpoint-selection rule, and sample count apply to both systems. Use separate `package` and `original` rows when any of these values differs. The four scope fields state the evaluator used, the evaluated test split, the rule that selects the checkpoint entering evaluation, and the sample count. Sample count is the metric denominator for the reported metrics.

| Dataset | System | Evaluator                                                                                                | Test split                    | Checkpoint-selection rule | Sample count                                                               |
| ------- | ------ | -------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `rico`  | both   | package `scripts/evaluate.py` (reconstruction and random-generation scores of the original `evaluate()`) | CanvasVAE MD5-hash test split | final epoch-500 weights   | 5,623 layouts for reconstruction; 5,623 random latents per evaluation seed |
| `crello` | both | not claimed | not claimed | not claimed | not claimed |

The Comparison Scope `Evaluator` cell names the evaluator that the manifest's `evaluator_command` runs, and the `Checkpoint-selection rule` cell equals the manifest's `checkpoint_rule`.

## Regeneration Metadata

The S0-S4 evidence was generated with commit [`787bf87`](https://github.com/creative-graphic-design/design-generators/commit/787bf87cb74c82009057fdd56a274e1c1af4d7d1) on a Google Colab CPU runtime (2 vCPUs) with Python 3.11, TensorFlow 2.15.1, Apache Beam 2.76.0 (`FnApiRunner`, `multi_threading`), PyTorch 2.13.0, Lightning 2.6.5, and Transformers 5.13.0, all in fp32 with `NVIDIA_TF32_OVERRIDE=0` and `CUDA_VISIBLE_DEVICES=""`. The original-code reference uses TensorFlow global seed 0 with op determinism and dropout 0; the package side injects the original's posterior noise. Issue evidence: [S0-S2 summary](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-5948477063).

```text
.cache/canvas-vae/data/semantic_annotations.zip      RICO archive
.cache/canvas-vae/data/rico/                         package splits and vocabulary counts
.cache/canvas-vae/original/data/rico/                original Beam TFRecords, vocabulary.json, count.json
.cache/canvas-vae/reference/trace/                   static.json, initial weights, step-0 trace, 50-step trajectory, per-step synchronized state
.cache/canvas-vae/reference/trace-repeat/            repeated original trajectory for the run-to-run envelope
.cache/canvas-vae/reference/stream/                  original records and stream orders
.cache/canvas-vae/reference/reports/                 measured differences per stage
```

The original Beam preprocessing is not deterministic in which screens it keeps: the reference run wrote 45,196 / 5,580 / 5,619 records and an earlier run 45,221 / 5,584 / 5,622, although the original record function emits every dropped screen when called directly. The package keeps all 45,222 / 5,584 / 5,623 screens; S4 compares the screens both keep.

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

```bash
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/train_original.py --seed 0 --job-dir .cache/canvas-vae/original/jobs/rico-seed0
```

Convert a trained checkpoint; use the original run's `final.ckpt` prefix and its data's `vocabulary.json` for original runs.

```bash
uv run --package canvas-vae models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/training-runs/rico/lightning_logs/version_0/checkpoints/last.ckpt --vocabulary .cache/canvas-vae/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted-trained/rico
```

Evaluate and smoke-test local loading.

```bash
uv run --package canvas-vae models/canvas-vae/scripts/evaluate.py --checkpoint .cache/canvas-vae/converted-trained/rico --output .cache/canvas-vae/evaluation/rico-package-seed0.json
uv run --package canvas-vae models/canvas-vae/scripts/smoke_from_pretrained.py --checkpoint .cache/canvas-vae/converted-trained/rico
```
