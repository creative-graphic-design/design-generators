---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# PixelVAE Training

PixelVAE training is available through a package-local Lightning module, data module, and `traingen` YAML recipe. The recipe uses the original batch size, objective, optimizer, epoch count, validation metric, and best-checkpoint rule. Existing CPU parity verifies the model objective and Keras Adam update on one deterministic untrained state; no trained checkpoint or full-run result is claimed.

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

Scheduler cadence — not applicable: the source and package use a constant learning rate. Importance sampler — not applicable: the training loop has no timestep or importance sampler. EMA — not applicable: the training loop has no exponential moving average. AMP — not applicable: the recipe uses float32. Multi-worker loader — applied: the sampler fixes record order in the main process, and the six-worker, persistent-worker, pinned-memory loader preserves the same first 20 batch tensor hashes at epochs 1 and 11 in the committed CPU regression test.

## Seed Policy

The checked-in recipe fixes the training seed at 0. This seed is the default for one PixelVAE run per system in the planned campaign; it is not a multi-seed result. CPU parity uses a deterministic seed-0 untrained TensorFlow state copied to PyTorch and is not training-seed evidence.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                         |
| ----- | ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Check source config and one-to-one TensorFlow weight mapping.                                   |
| S1    | Fixed-batch pre-optimizer trace parity     | Compare decoder output, posterior statistics, and source loss terms.                            |
| S2    | One optimizer-step parity                  | Compare gradients, Keras Adam moments, and updated parameters.                                  |
| S3    | Short deterministic multi-batch run        | Compare the existing three-step CPU model trace.                                                |
| S4    | Crello source and loader checks            | Check source selection, PNG decoding, record filtering and order, and posterior-mean save/load. |
| S5    | Full-run statistical comparison            | Not run; full training and trained-checkpoint evaluation have not been performed.               |

## Stage Evidence

| Stage | Command                                                                                                                               | Artifact                                                                             | Result                                                                                                                                                                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh`               | `input-selection-plan.json`, source state record, calibration input sidecars         | Pass; exact config, complete key map, and bitwise state tensors for one deterministic untrained state.                                                                                                                                                                              |
| S1    | Same ordered CPU acceptance command                                                                                                   | `diagnostic/s1-diagnostic.json`, S0 input selections, `calibration/repeat-1..3.json` | Pass; all 6 held-out forward and loss metrics met their frozen limits.                                                                                                                                                                                                              |
| S2    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                       | Pass; all 6 held-out gradient and optimizer metrics met their frozen limits.                                                                                                                                                                                                        |
| S3    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                       | Pass; all 4 held-out metrics in the existing three-step model trace met their frozen limits.                                                                                                                                                                                        |
| S4    | `PARITY_REQUIRE=1 uv run --package pixel-vae --extra training --group dev pytest models/pixel-vae/tests/test_training.py -m training` | `models/pixel-vae/tests/test_training.py`                                            | Pass on CPU: the 41-PNG fixture first 20 batch tensors (batch size 2) match the legacy stream at epochs 1 and 11 with six workers, persistent workers, and pinned memory; the dedupe fixture asserts 89,545 train/11,690 val records and 1,400/183 batch-size-64 batches per epoch. |
| S5    | Not run                                                                                                                               | N/A                                                                                  | Not run; there is no trained checkpoint or full-run evaluation, so trained-checkpoint reproduction is not claimed.                                                                                                                                                                  |

## Reproduction Results

No trained-checkpoint reproduction is claimed for Crello v1. Existing CPU conversion and model parity passed on a deterministic seed-0 untrained state; the package training interface is implemented, but neither system has a completed training run in this record.

| Dataset   | System   | Status                                                                                      | Seed scope                   | Primary metrics                                       | Loss evidence        | Artifact summary    |
| --------- | -------- | ------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------- | -------------------- | ------------------- |
| Crello v1 | original | not-yet-run ([#31](https://github.com/creative-graphic-design/design-generators/issues/31)) | seed-0 untrained state only  | CPU model parity passed; no trained-checkpoint metric | No training loss run | `.cache/pixel-vae/` |
| Crello v1 | package  | not-yet-run ([#31](https://github.com/creative-graphic-design/design-generators/issues/31)) | training seed 0; not yet run | CPU model parity passed; no trained-checkpoint metric | No training loss run | `.cache/pixel-vae/` |

### Comparison Scope

| Dataset   | System | Evaluator           | Test split | Checkpoint-selection rule                              | Sample count   |
| --------- | ------ | ------------------- | ---------- | ------------------------------------------------------ | -------------- |
| Crello v1 | both   | not run; S5 pending | test       | best validation reconstruction; no checkpoint selected | not applicable |

## Regeneration Metadata

The original archive digest, source commit, seed, input IDs, TensorFlow version, and encoder-state digest are recorded in private reports. Keep parity reports, archive audits, checkpoints, and derived embeddings outside Git.

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

For a full GPU run, use the [segmented Crello job script](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/scripts/run_crello_training_segment.sh) through `colab-gpu-run`. Name each job `pixelvae-<40-hex-source-sha>-e<3-digit-epoch>-a<unique-attempt-id>`. For the current A100 campaign, target epochs are 010, 078, 146, 214, and 250: the first segment runs 10 epochs and later segments run up to 68. The first production launch runs the 300-step record-only lockstep. After a training-path change, update the pinned attribution-report SHA-256 and set `PIXELVAE_RUN_LOCKSTEP=1` to run it again; otherwise later segments verify the pinned report SHA-256 and skip the lockstep. Each later segment downloads both `last.ckpt` and `best.ckpt` from the preceding attempt so Lightning can resume optimizer state and keep the best-checkpoint selection history. Declare the commit-pinned source archive as `--hub-input source=...` and pass the data export as one staged tar through `--hub-input data_tar=...`; the tar root contains `count.json` and the train/validation TFRecord shards. The script keeps raw training output out of upload paths until it passes a scan for Hugging Face token-shaped strings (`hf_` followed by at least 20 letters or digits); its wrapper log contains only the checked summary and is scanned before final upload.

The direct `traingen fit` command above runs the full 250-epoch config in one process; its selected best checkpoint is saved under `.cache/pixel-vae/training-runs/crello/`. Full training and S5 evaluation are pending, so trained-checkpoint reproduction is not claimed.
