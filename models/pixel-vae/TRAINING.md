---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# PixelVAE Training

The original TensorFlow PixelVAE and package PixelVAE each completed one 250-epoch Crello v1 run; the package used training seed 0 and ended at `global_step=350000`. Original validation reconstruction was `184.70` at epoch 250 using deterministic posterior means. The package value `193.7491` at epoch 249 came from sampled-posterior validation in the recorded run and is not comparable. Both values are references only. Current `validation_step` passes `sample=False`, so package validation uses deterministic posterior means. The recorded run predates this fix and the resumed-sampler fix described below; rerunning the documented command would not reproduce `193.7491` exactly.

S5 (full-run training and test evaluation in the [protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md)) has not run. PixelVAE S5 is not claimed, and PixelVAE training reproduction is not verified. Both PixelVAE trainings were deferred because the campaign estimate exceeded the available compute limit. The [issue #31 compute-budget comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806) records that decision. The shared-cache CanvasVAE comparison uses embeddings generated from untrained original PixelVAE weights and does not evaluate package PixelVAE training.

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

| Dataset   | Source                                                                                                                                                                                                       | Config or path                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Crello v1 | [Pinned original GCS archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip), 2,989,732,284 bytes, SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e` | Filtered image-record export under `.cache/pixel-vae/data/crello-image/`; derived records stay local and uncommitted. |

The data module reads `count.json` and `train-*.tfrecord`/`val-*.tfrecord` records emitted by the original image-record transform. That transform selects `imageElement`, `maskElement`, and `svgElement` and deduplicates identical PNG bytes within each split. Each TFRecord example contains one `image` byte feature. The pinned GCS source archive is public; keep derived image records local and uncommitted.

The data module indexes each unique image payload's TFRecord offset without decoding PNGs, then the sampler determines the package-deterministic shuffle order in the main process. A map-style PyTorch dataset reads and decodes records in that order; PyTorch's `DataLoader` preserves it when `num_workers` is greater than zero. The Crello recipe uses six workers and pinned memory; persistent workers are enabled whenever workers are active. Worker seeds use a separate CPU generator. Each `DataLoader` iterator consumes one CPU int64 base-seed value from PyTorch's default RNG. The training module makes one matching draw at each training and validation epoch boundary, keeping later CPU random values aligned.

A report-only loader benchmark on an eight-vCPU High-RAM Colab CPU measured 7.754, 10.676, 18.213, and 22.009 batches per second at 0, 2, 4, and 6 workers, respectively, after 20 order-check batches and 128 measured batches per setting. Six workers were fastest among the tested settings. The throughput probe used `pin_memory: false`. The retained regression test checks the recipe's pinned-memory setting and matches the first 20 package-deterministic batches at epochs 1 and 11. These CPU measurements do not establish A100 throughput.

## Configs

Training configs live under `models/pixel-vae/configs/training/`.

| Config                  | Dataset   | Seed mode            | Purpose                             |
| ----------------------- | --------- | -------------------- | ----------------------------------- |
| `pixel_vae_crello.yaml` | Crello v1 | deterministic seed 0 | Original 250-epoch training recipe. |

## Scheduler and Recipe Notes

The package uses batch size 64 for 250 epochs, `KerasAdam` at a constant learning rate of `1e-4`, KL weight 100, L2 weight `1e-6`, float32, and validation after every epoch. The shared PixelVAE model computes the alpha-masked RGB and alpha reconstruction objective, weighted KL divergence, and variational-head L2 penalty. The best checkpoint minimizes `val_reconstruction` and stores weights only; a separate full-state `last.ckpt` stores optimizer and epoch state at every epoch boundary for resuming segmented runs.

The training config includes an epoch timing callback. Set `PIXELVAE_EPOCH_TIMING_FILE` to write one JSONL record per epoch. Each record stores train-and-validation duration and UTC start/end timestamps; the segment runner summarizes GPU use from those timestamps.

The original image-record transform selects `imageElement`, `maskElement`, and `svgElement` images, then deduplicates identical PNG bytes within each split. [The Crello amendment records this selection.](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6065470695)

## Segment Acceptance

Accept a completed segment only after reading its `last.ckpt` back from the Hugging Face Hub: its SHA-256 must equal `checkpoints.last_sha256` in the segment report, and its stored `global_step` must equal `segment.global_step` in that report. A successful training exit code alone does not confirm that every artifact reached the Hub. The best checkpoint stores weights only and cannot resume optimizer state; continue segmented training from the verified full-state `last.ckpt`.

Scheduler cadence — not applicable: the source and package use a constant learning rate. Importance sampler — not applicable: the loop has no timestep or importance sampler. EMA — not applicable: the loop has no exponential moving average. AMP — not applicable: the recipe uses float32. Multi-worker loader — applied: the sampler fixes record order in the main process. The six-worker persistent loader with pinned memory preserved the first 20 batch tensor hashes at epochs 1 and 11 in the CPU regression test.

## Completed Crello v1 PixelVAE Run

The package checkpoint rule selects minimum `val_reconstruction` (`mode=min`, top-k 1). In the recorded continuation, `best.ckpt` was selected at epoch 249 (SHA-256 `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`). It was selected using sampled-posterior validation, a deviation from the original's deterministic-mean selection; the exported encoder depends on that choice. The encoder was re-exported from the selected checkpoint on CPU. The exported `model.safetensors` SHA-256 is `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`; `PixelVAEEncoder.from_pretrained` loaded it with 2,880,032 parameters. Export with:

```bash
uv run --package pixel-vae --extra training models/pixel-vae/scripts/export_trained_encoder.py \
  --checkpoint /path/to/epoch-249-best.ckpt \
  --output-dir .cache/pixel-vae/encoder-best-e249-20261108
```

The final epoch-250 `last.ckpt` SHA-256 is `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87` and stores the final training state. The existing cache contains 114,007 × 256 float32 rows. It was generated from the epoch-250 `last.ckpt` encoder file (SHA-256 `dd8ed8e8a85bae96eeba8a2cbf5cfdad7b5e23a5456f856039db63ef279a53f2`), not the re-exported epoch-249 best encoder. Its array SHA-256 is `33f1d1757bc8ae1a6e46d0dd8bd1230c21b133d881a65dfcf1f012ff9738bb3e`; its encoder-state SHA-256 is `1c3345e75e57add7fc529dd02fee01a5550f48ac09b8e798e51ca570fa9ac62e`. No cache was regenerated from the best-checkpoint encoder. CanvasVAE uses the separate untrained-original cache `fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a`, so its comparison is unaffected.

The first PixelVAE run covered epochs 1–227 on an A100 with Torch 2.13.0 and the CUDA 13.0 build. The continuation used Torch `2.13.0+cu126` on a Tesla V100-SXM2-32GB because the available driver could not initialize the locked CUDA 13.0 build. Its `pip freeze` SHA-256 is `67127ccc31819828846e2cd41bada86d37800801436cb47461aff4d4a248f149`. Eighteen package records differed from the first run's environment record (`5edbe286b3b2851230c36f3b92c659e95923e8e3f7780810f5d52c2e97cc274a`). All differences are Torch or CUDA runtime distributions.

The final resumed segment ran with `--trainer.max_epochs=-1 --trainer.max_steps=350000` from the unchanged epoch-245 `last.ckpt` at `global_step=343000`, with sampler start epoch 245. Lightning increments `fit_loop.epoch_progress.processed` at train-epoch end before the boundary checkpoint is saved, so each resumed segment restored an epoch counter one ahead of the sampler; by epoch 245 the accumulated offset was five epochs. The step limit ended the final checkpoint at epoch 250 and step 350000 without editing checkpoint metadata. The segmented script at this head also passes `--trainer.max_epochs=-1`, so the absolute `max_steps` target controls completion.
The first training epochs after recorded resumes were 11, 79, 147, 194, 228, and 246. A100 segment records show resumes after boundaries 10, 78, and 146 at epochs 11, 79, and 147. The segment targeting boundary 214 stopped at absolute epoch 194; the retained evidence does not record why. The next A100 segment resumed from sampler epoch 193, so its first resumed epoch was 194 rather than 215. V100 segments resumed at epochs 228 and 246. The recorded configs for epochs 11, 79, 194, 228, and 246 used six workers; the epoch-147 segment's config was not retained, but its pinned source config set `num_workers: 6`. No resumed segment is recorded with zero workers. Workers prefetched before the callback set the configured epoch, so each first resumed epoch consumed epoch-0 order. The retained timing JSONL files show inaccurate `sampler_epoch` values at resumed epochs 79, 147, 194, 228, and 246. The epoch-11 timing file has no `sampler_epoch` field; its Lightning-counted `epoch` is 12 for absolute epoch 11. The retained V100 production timing file contains 18 full-epoch records for absolute epochs 228–245; its first row records `sampler_epoch=227` and 1,019 seconds. Earlier recorded segment endpoints were 10, 78, 146, 194, 227, and 245. None of those records identifies the epoch cap as the stopping reason. The retained launch records do not preserve every segment's effective override.

Current `PixelVAEDataModule.setup()` sets the configured sampler epoch before a `DataLoader` iterator is created and workers prefetch. `test_trainer_resume_starts_with_the_configured_sampler_epoch` verifies the resumed order with a real two-worker Trainer resume. The recorded run predates this fix.

The recipe used `precision=32-true` and did not override Torch's TF32 backend flags. With Torch 2.13.0 defaults, matmul TF32 was disabled and cuDNN TF32 was enabled; `32-true` disables autocast but does not disable cuDNN TF32. The A100 used by the first run could therefore use TF32 for convolution, while the V100 has no TF32 hardware support.

## Seed Policy

The checked-in recipe fixes the training seed at 0. It describes one run, not the planned S5 comparison. S5 uses three full training runs per system. CPU parity uses a deterministic seed-0 untrained TensorFlow state copied to PyTorch and is not training-seed evidence.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                                                                     |
| ----- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Check source config and one-to-one TensorFlow weight mapping.                                                                                               |
| S1    | Fixed-batch pre-optimizer trace parity     | Compare decoder output, posterior statistics, and source loss terms.                                                                                        |
| S2    | One optimizer-step parity                  | Compare gradients, Keras Adam moments, and updated parameters.                                                                                              |
| S3    | Short deterministic multi-batch run        | Compare the existing three-step CPU model trace.                                                                                                            |
| S4    | Crello source and loader checks            | Check source selection, PNG decoding, record filtering and order, and posterior-mean save/load.                                                             |
| S5    | Full-run test evaluation                   | The original evaluates test reconstruction, KL, and total loss from the validation-selected best checkpoint; the planned three-seed comparison has not run. |

## Stage Evidence

| Stage | Command                                                                                                                               | Artifact                                                                             | Result                                                                                                                                                                                                                                                                                                                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh`               | `input-selection-plan.json`, source state record, calibration input sidecars         | Pass; exact config, complete key map, and bitwise state tensors for one deterministic untrained state.                                                                                                                                                                                                                                                                    |
| S1    | Same ordered CPU acceptance command                                                                                                   | `diagnostic/s1-diagnostic.json`, S0 input selections, `calibration/repeat-1..3.json` | Pass; all 6 held-out forward and loss metrics met their frozen limits.                                                                                                                                                                                                                                                                                                    |
| S2    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                       | Pass; all 6 held-out gradient and optimizer metrics met their frozen limits.                                                                                                                                                                                                                                                                                              |
| S3    | Same ordered CPU acceptance command                                                                                                   | `calibration/repeat-1..3.json`                                                       | Pass; all 4 held-out metrics in the existing three-step model trace met their frozen limits.                                                                                                                                                                                                                                                                              |
| S4    | `PARITY_REQUIRE=1 uv run --package pixel-vae --extra training --group dev pytest models/pixel-vae/tests/test_training.py -m training` | `models/pixel-vae/tests/test_training.py`                                            | Pass on CPU. The S4 legacy-order expectation is package-derived: the 41-PNG fixture's first 20 batch tensors (batch size 2) match the package-deterministic stream at epochs 1 and 11 with six workers, persistent workers, and pinned memory. The dedupe fixture asserts 89,545 train and 11,690 validation records, plus 1,400 and 183 batch-size-64 batches per epoch. |
| S5    | `uv run --package pixel-vae --extra training traingen fit --config models/pixel-vae/configs/training/pixel_vae_crello.yaml`           | None                                                                                 | Not run; see Reproduction Results.                                                                                                                                                                                                                                                                                                                                        |

## Reproduction Results

The recorded original validation reconstruction `184.70` at epoch 250 used deterministic posterior means. The recorded package value `193.7491` at epoch 249 came from sampled-posterior validation and is not comparable. Current package validation instead uses deterministic posterior means. Both values are references only, not S5 results.

PixelVAE S5 is not claimed. Its status is `not-yet-run (PixelVAE trainings deferred for compute budget: https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806)`. The validation values are references only. The original and current package code use deterministic posterior means; the recorded package value came from sampled-posterior validation and is not comparable. PixelVAE training reproduction is not verified.

| Dataset   | System   | Status                                                                                                                                                         | Seed scope                                                  | Primary metrics                                                                                  | Loss evidence                                                                                | Artifact summary                                                                                                                                                                                                                                                                                         |
| --------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crello v1 | original | `not-yet-run (PixelVAE trainings deferred for compute budget: https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806)` | PixelVAE training seed 0; no PixelVAE evaluation seed scope | Deterministic validation reference only: `184.70` at epoch 250                                   | Original training finished; the planned PixelVAE test evaluation was not run.                | No PixelVAE S5 test artifact.                                                                                                                                                                                                                                                                            |
| Crello v1 | package  | `not-yet-run (PixelVAE trainings deferred for compute budget: https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806)` | PixelVAE training seed 0; no PixelVAE evaluation seed scope | Sampled-posterior validation reference only, not comparable to original: `193.7491` at epoch 249 | Package training finished at epoch 250 / step 350,000; PixelVAE test evaluation was not run. | Best `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`; last `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87`; best encoder `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`; cache from epoch-250 last encoder; no cache from best (hashes above). |

### Downstream CanvasVAE Comparison (Separate Scope)

The CanvasVAE systems shared the full-size cache (array SHA-256: `fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a`). Untrained original PixelVAE weights with seed 0 produced it. Their three-seed ranges intersect for Sreconst, layout mIoU, KL, and Sgen. This comparison evaluates CanvasVAE training on shared inputs; it does not evaluate package PixelVAE training. Per-seed results, commands, and artifact hashes are recorded in [CanvasVAE PR 458](https://github.com/creative-graphic-design/design-generators/pull/458).

### Comparison Scope

| Dataset   | System | Evaluator                                                   | Test split | Checkpoint-selection rule                                                                | Sample count                                                               |
| --------- | ------ | ----------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Crello v1 | both   | CanvasVAE evaluator, separate from PixelVAE test evaluation | test       | CanvasVAE final checkpoints; both systems use the same untrained-original-PixelVAE cache | 2,278 documents per CanvasVAE evaluation; PixelVAE test evaluation not run |

## Regeneration Metadata

The original archive digest, source commit, seed, input IDs, TensorFlow version, original artifact hashes, and encoder-state digest are retained with the run evidence. Keep parity reports, archive audits, checkpoints, and derived embeddings outside Git. Record the artifact hashes and aggregate metrics needed to identify results without copying private artifact locations.

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

### Reader-Reproducible Training

The reader-reproducible path is the direct `traingen fit` command. First put the filtered image-record export described in Data at `.cache/pixel-vae/data/crello-image/`. The command uses the checked-in 250-epoch Crello recipe and needs no Colab launcher or Hub inputs.

```bash
uv run --package pixel-vae --extra training traingen fit \
  --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
```

### Recorded Segmented Campaign

Recorded segments used the [runner](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/scripts/run_crello_training_segment.sh). `colab-gpu-run` is its private Colab job launcher. The segmented route needs that launcher and private staged Hub inputs; readers without them should use the direct command above. The runner requires `uv 0.12.22` and syncs the locked training dependencies. Production runs use batch size 64, one visible CUDA device, and absolute epoch endpoints 010, 078, 146, 214, and 250. The first segment covers 10 epochs and later segments cover at most 68. The runner sets `max_epochs=-1` and stops at the absolute `max_steps` target.

The runner expects exactly one `<job-name>/logs/job.log` beneath `COLAB_JOB_WORK_ROOT`. If unset, that directory is `$PIXELVAE_WORK_ROOT/colab-jobs`; `PIXELVAE_WORK_ROOT` defaults to the current directory. The job name is `pixelvae-<40-hex-source-sha>-e<3-digit-end-epoch>-a<unique-attempt-id>`. The runner checks that its commit matches the source archive and reads the epoch endpoint from the name.

| `--hub-input` name  | Required when                                                                                   | Purpose                                                                                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source`            | Every invocation                                                                                | HTTPS archive for the exact 40-character source commit.                                                                                                           |
| `data_tar`          | Every production run and normal CPU rehearsal; omitted only when preparing the tiny CPU fixture | Private archive of filtered Crello image records.                                                                                                                 |
| `attribution`       | Every production run and normal CPU rehearsal; omitted only when preparing the tiny CPU fixture | Private lockstep attribution report; the runner verifies its SHA-256 against the pinned value `99030e6a1b53303898851707cb1f31b67a681e59b295f20408635b89738d1b17`. |
| `resume` and `best` | Together on every resumed segment; omit both for the first segment                              | Previous full-state `last.ckpt` for optimizer resume and validation-best `best.ckpt` to preserve checkpoint selection.                                            |

The data archive keeps records under `original-run-2/crello-image/`, including `count.json` and the train, validation, and test TFRecord shards. The runner strips those two leading path components, leaving `count.json` and the records directly in its data directory. A resumed job downloads both checkpoint inputs from the preceding attempt; it requires the full-state checkpoint to end at an epoch boundary.

The first production segment runs a 300-step lockstep, which records matched updates from both systems without a pass/fail gate. After changing the training path, update the pinned attribution-report SHA-256 and set `PIXELVAE_RUN_LOCKSTEP=1`. Later segments check that hash and skip the lockstep unless this setting requests another run. The runner scans its detailed log and evidence files for token-shaped strings before reporting completion; it does not upload Hub files itself.

The direct command runs the full 250-epoch config in one process. The completed segmented run selected epoch 249 by minimum `val_reconstruction`. The encoder was re-exported on CPU from that best checkpoint; its `model.safetensors` SHA-256 is `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`. For S5 status, see Reproduction Results.
