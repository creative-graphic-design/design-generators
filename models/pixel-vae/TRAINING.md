---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# PixelVAE Training

The original TensorFlow PixelVAE and the package PixelVAE each finished 250 Crello v1 epochs; the package run used training seed 0 and ended at `global_step=350000`. Its selected validation best is `193.7491` at epoch 249, compared with the original run’s `184.70` at epoch 250. The package `last.ckpt` supplied its encoder export. PixelVAE S5 is not claimed because the protocol defines no standalone PixelVAE test metric. The downstream CanvasVAE comparison is described separately; it uses an untrained original PixelVAE cache and does not measure end-to-end PixelVAE-to-CanvasVAE reproduction.

Run commands from the repository root. Keep source records, logs, checkpoints, conversion outputs, and embeddings under `.cache/pixel-vae/`; these artifacts are private and are not committed.

## Install

```bash
uv sync --package pixel-vae --extra training
```

The training extra installs Lightning, LightningCLI support, the TFRecord reader, and the shared `traingen` optimizer. Add `vendor` and `parity` only when running original TensorFlow checks.

```bash
uv sync --package pixel-vae --extra training --extra vendor --extra parity
```

## Data

| Dataset   | Source                                                                                                                                                                                                       | Config or path                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Crello v1 | [Pinned original GCS archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip), 2,989,732,284 bytes, SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e` | Filtered image-record export under `.cache/pixel-vae/data/crello-image/`; private source data. |

The data module reads `count.json` and `train-*.tfrecord`/`val-*.tfrecord` records emitted by the original image-record transform. That transform selects `imageElement`, `maskElement`, and `svgElement` and deduplicates identical PNG bytes within each split. Each TFRecord example contains one `image` byte feature. The source archive and derived image records remain private.

The data module indexes each unique image payload's TFRecord offset without decoding PNGs, then the sampler determines the source-compatible shuffle order in the main process. A map-style PyTorch dataset reads and decodes records in that order; PyTorch's `DataLoader` preserves it when `num_workers` is greater than zero. The Crello recipe uses six workers and pinned memory; persistent workers are enabled whenever workers are active. Worker seeds use a separate CPU generator, while the training module preserves the previous zero-worker loader's CPU RNG draw at each train and validation epoch boundary.

A report-only loader benchmark on an eight-vCPU High-RAM Colab CPU measured 7.754, 10.676, 18.213, and 22.009 batches per second at 0, 2, 4, and 6 workers, respectively, after 20 order-check batches and 128 measured batches per setting. Six workers were fastest among the tested settings. The throughput probe used `pin_memory: false`; the retained regression test checks the recipe's pinned-memory setting and matches the legacy tensor order for the first 20 batches of epochs 1 and 11. These CPU measurements do not establish A100 throughput.

## Configs

Training configs live under `models/pixel-vae/configs/training/`.

| Config                  | Dataset   | Seed mode            | Purpose                             |
| ----------------------- | --------- | -------------------- | ----------------------------------- |
| `pixel_vae_crello.yaml` | Crello v1 | deterministic seed 0 | Original 250-epoch training recipe. |

## Scheduler and Recipe Notes

The package uses batch size 64 for 250 epochs, `KerasAdam` at a constant learning rate of `1e-4`, KL weight 100, L2 weight `1e-6`, float32, and validation after every epoch. The shared PixelVAE model computes the alpha-masked RGB and alpha reconstruction objective, weighted KL divergence, and variational-head L2 penalty. The best checkpoint minimizes `val_reconstruction` and stores weights only; a separate full-state `last.ckpt` stores optimizer and epoch state at every epoch boundary for resuming segmented runs.

The training config includes an epoch timing callback. Set `PIXELVAE_EPOCH_TIMING_FILE` to write one JSONL record per epoch with the train-and-validation wall duration and UTC start/end timestamps; the segment runner uses those timestamps to summarize GPU utilization during each epoch.

The original image-record transform selects `imageElement`, `maskElement`, and `svgElement` images, then deduplicates identical PNG bytes within each split. [The Crello amendment records this selection.](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6065470695)

## Segment Acceptance

Accept a completed segment only after reading its `last.ckpt` back from the Hugging Face Hub: its SHA-256 must equal `checkpoints.last_sha256` in the segment report, and its stored `global_step` must equal `segment.global_step` in that report. A successful training exit code alone does not confirm that every artifact reached the Hub. The best checkpoint stores weights only and cannot resume optimizer state; continue segmented training from the verified full-state `last.ckpt`.

Scheduler cadence — not applicable: the source and package use a constant learning rate. Importance sampler — not applicable: the training loop has no timestep or importance sampler. EMA — not applicable: the training loop has no exponential moving average. AMP — not applicable: the recipe uses float32. Multi-worker loader — applied: the sampler fixes record order in the main process, and the six-worker, persistent-worker, pinned-memory loader preserves the same first 20 batch tensor hashes at epochs 1 and 11 in the committed CPU regression test.

## Completed Crello v1 PixelVAE Run

The package resumed from the epoch-227 `last.ckpt` produced by its preceding run and finished at epoch 250 with `global_step=350000`. The preceding run's global best had `val_reconstruction=194.7901153564453` and SHA-256 `02b64603f4343c0367ddf5fd0e3d73639012e230dbff4e8133f45984f4476799`; the continuation's best was `193.74908447265625` at epoch 249 with SHA-256 `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`. The continuation checkpoint is the global best because `val_reconstruction` is minimized. The final epoch-250 `last.ckpt` has SHA-256 `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87`; the encoder export used this final checkpoint, as the CanvasVAE recipe requires. The exported `model.safetensors` SHA-256 is `dd8ed8e8a85bae96eeba8a2cbf5cfdad7b5e23a5456f856039db63ef279a53f2`, and `PixelVAEEncoder.from_pretrained` loaded it with 2,880,032 parameters. The package embedding cache contains 114,007 × 256 float32 rows with array SHA-256 `33f1d1757bc8ae1a6e46d0dd8bd1230c21b133d881a65dfcf1f012ff9738bb3e` and encoder-state SHA-256 `1c3345e75e57add7fc529dd02fee01a5550f48ac09b8e798e51ca570fa9ac62e`.

The first PixelVAE run covered epochs 1–227 on an A100 with Torch 2.13.0 and the CUDA 13.0 build. The continuation used one Tesla V100-SXM2-32GB and Torch `2.13.0+cu126` because the server's CUDA 12.9 driver could not initialize the locked CUDA 13.0 build. Its `pip freeze` SHA-256 is `67127ccc31819828846e2cd41bada86d37800801436cb47461aff4d4a248f149`; compared with the environment record for the first run (`5edbe286b3b2851230c36f3b92c659e95923e8e3f7780810f5d52c2e97cc274a`), all 18 differing package records are Torch or CUDA runtime distributions.

The final resumed segment used `--trainer.max_epochs=-1 --trainer.max_steps=350000` from the unchanged epoch-245 `last.ckpt` at `global_step=343000`, with sampler start epoch 245. Lightning increments `fit_loop.epoch_progress.processed` at train-epoch end before the boundary checkpoint is saved, so each resumed segment restored an epoch counter one ahead of the sampler; by epoch 245 the accumulated offset was five epochs. The step limit ended the final checkpoint at epoch 250 and step 350000 without editing checkpoint metadata. Use an absolute `max_steps` target for resumed segments.

The recipe used `precision=32-true` and did not override Torch's TF32 backend flags. With Torch 2.13.0 defaults, matmul TF32 was disabled and cuDNN TF32 was enabled; `32-true` disables autocast but does not disable cuDNN TF32. The A100 used by the first run could therefore use TF32 for convolution, while the V100 has no TF32 hardware support.

## Seed Policy

The checked-in recipe fixes the training seed at 0. This seed is the default for one PixelVAE run per system in the planned campaign; it is not a multi-seed result. CPU parity uses a deterministic seed-0 untrained TensorFlow state copied to PyTorch and is not training-seed evidence.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                |
| ----- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| S0    | Static config and initialized state parity | Check source config and one-to-one TensorFlow weight mapping.                                          |
| S1    | Fixed-batch pre-optimizer trace parity     | Compare decoder output, posterior statistics, and source loss terms.                                   |
| S2    | One optimizer-step parity                  | Compare gradients, Keras Adam moments, and updated parameters.                                         |
| S3    | Short deterministic multi-batch run        | Compare the existing three-step CPU model trace.                                                       |
| S4    | Crello source and loader checks            | Check source selection, PNG decoding, record filtering and order, and posterior-mean save/load.        |
| S5    | Full-run statistical comparison            | No standalone PixelVAE test metric is defined. Downstream CanvasVAE results are documented separately. |

## Stage Evidence

| Stage | Command                                                                                                                               | Artifact                                                                                     | Result                                                                                                                                                                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh`               | `input-selection-plan.json`, source state record, calibration input sidecars                 | Pass; exact config, complete key map, and bitwise state tensors for one deterministic untrained state.                                                                                                                                                                              |
| S1    | Same ordered CPU acceptance command                                                                                                   | `diagnostic/s1-diagnostic.json`, S0 input selections, `calibration/repeat-1..3.json`         | Pass; all 6 held-out forward and loss metrics met their frozen limits.                                                                                                                                                                                                              |
| S2    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                               | Pass; all 6 held-out gradient and optimizer metrics met their frozen limits.                                                                                                                                                                                                        |
| S3    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                               | Pass; all 4 held-out metrics in the existing three-step model trace met their frozen limits.                                                                                                                                                                                        |
| S4    | `PARITY_REQUIRE=1 uv run --package pixel-vae --extra training --group dev pytest models/pixel-vae/tests/test_training.py -m training` | `models/pixel-vae/tests/test_training.py`                                                    | Pass on CPU: the 41-PNG fixture first 20 batch tensors (batch size 2) match the legacy stream at epochs 1 and 11 with six workers, persistent workers, and pinned memory; the dedupe fixture asserts 89,545 train/11,690 val records and 1,400/183 batch-size-64 batches per epoch. |
| S5    | Not claimed for PixelVAE                                                                                                              | [CanvasVAE S5 record](https://github.com/creative-graphic-design/design-generators/pull/458) | No standalone PixelVAE test metric is defined here. The linked pull request records a separate downstream CanvasVAE comparison.                                                                                                                                                     |

## Reproduction Results

PixelVAE S5 is not claimed because the protocol has no standalone PixelVAE test-set metric. The original and package PixelVAE runs each finished 250 epochs. The original best validation reconstruction was `184.70`; the package best was `193.7491` at epoch 249. The downstream CanvasVAE comparison is documented separately and does not evaluate the package PixelVAE cache.

The table marks PixelVAE S5 as not claimed because no standalone PixelVAE test metric is defined; this does not mean either 250-epoch training run is missing. CanvasVAE training-seed metrics are not PixelVAE results.

| Dataset   | System   | Status                                                | Seed scope                                    | Primary metrics                                                            | Loss evidence                                                                  | Artifact summary                                                                                                                                                                                                                                                                                        |
| --------- | -------- | ----------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crello v1 | original | not-yet-run (no standalone PixelVAE test metric is defined; downstream CanvasVAE results are separate) | PixelVAE training seed 0; CanvasVAE seeds 0–2 | No PixelVAE test metric; validation reconstruction `184.70` at epoch 250   | Original PixelVAE training finished; downstream CanvasVAE metrics are separate | Original PixelVAE training completed; no standalone S5 test artifact exists.                                                                                                                                                                                                                             |
| Crello v1 | package  | not-yet-run (no standalone PixelVAE test metric is defined; downstream CanvasVAE results are separate) | PixelVAE training seed 0; CanvasVAE seeds 0–2 | No PixelVAE test metric; validation reconstruction `193.7491` at epoch 249 | Package PixelVAE training finished at epoch 250 / step 350,000                 | Best `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`; last `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87`; encoder `dd8ed8e8a85bae96eeba8a2cbf5cfdad7b5e23a5456f856039db63ef279a53f2`; cache `33f1d1757bc8ae1a6e46d0dd8bd1230c21b133d881a65dfcf1f012ff9738bb3e`. |

### Downstream CanvasVAE Comparison (Separate Scope)

The CanvasVAE package and original each used the same full-size cache, array SHA-256 `fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a`, produced by untrained, seed-0-initialized original PixelVAE weights. Their three-seed ranges intersect for Sreconst, layout mIoU, KL, and Sgen. This downstream comparison evaluates CanvasVAE training on identical inputs; it does not evaluate the package PixelVAE cache or end-to-end PixelVAE-to-CanvasVAE reproduction. Per-seed results, commands, and artifact hashes are recorded in [CanvasVAE PR 458](https://github.com/creative-graphic-design/design-generators/pull/458).

### Comparison Scope

| Dataset   | System | Evaluator                                                          | Test split | Checkpoint-selection rule                                                             | Sample count                                                           |
| --------- | ------ | ------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Crello v1 | both   | CanvasVAE evaluator, reported separately; not a PixelVAE S5 metric | test       | CanvasVAE final checkpoints; both arms use the same untrained-original-PixelVAE cache | 2,278 documents per CanvasVAE evaluation; no PixelVAE test denominator |

## Regeneration Metadata

The original archive digest, source commit, seed, input IDs, TensorFlow version, original artifact hashes, and encoder-state digest are retained with the run evidence. Keep parity reports, archive audits, checkpoints, and derived embeddings outside Git; record the artifact hashes and aggregate metrics needed to identify results without copying private artifact locations.

```text
.cache/pixel-vae/data/crello-image/
.cache/pixel-vae/parity/
.cache/pixel-vae/training-runs/crello/
```

## Training Commands

Run the existing CPU model and vendor parity checks:

```bash
PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> \
  bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh
```

Train the package with the checked-in Crello recipe. The data directory must contain the filtered image-record export described above.

```bash
uv run --package pixel-vae --extra training traingen fit \
  --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
```

For a Colab GPU run, use the [segmented Crello job script](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/scripts/run_crello_training_segment.sh) through `colab-gpu-run`. Name each job `pixelvae-<40-hex-source-sha>-e<3-digit-epoch>-a<unique-attempt-id>`. The original A100 campaign targeted epochs 010, 078, 146, 214, and 250: the first segment runs 10 epochs and later segments run up to 68. The first production launch runs the 300-step record-only lockstep. After a training-path change, update the pinned attribution-report SHA-256 and set `PIXELVAE_RUN_LOCKSTEP=1` to run it again; otherwise later segments verify the pinned report SHA-256 and skip the lockstep. Each later segment downloads both `last.ckpt` and `best.ckpt` from the preceding attempt so Lightning can resume optimizer state and keep the best-checkpoint selection history. Declare the commit-pinned source archive as `--hub-input source=...` and pass the data export as one staged tar through `--hub-input data_tar=...`; the tar root contains `count.json` and the train/validation TFRecord shards. The script keeps raw training output out of upload paths until it passes a scan for Hugging Face token-shaped strings (`hf_` followed by at least 20 letters or digits); its wrapper log contains only the checked summary and is scanned before final upload.

The direct `traingen fit` command above runs the full 250-epoch config in one process. The completed segmented run selected the checkpoint with the lowest `val_reconstruction` across the first run and its continuation; the final `last.ckpt`, rather than that selected best checkpoint, supplied the package PixelVAE encoder. PixelVAE S5 remains unclaimed because there is no standalone PixelVAE test metric; the separate fixed-cache CanvasVAE comparison does not change that status.
