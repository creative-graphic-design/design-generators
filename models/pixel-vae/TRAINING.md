---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# PixelVAE Training

The original TensorFlow PixelVAE and package PixelVAE each completed one 250-epoch Crello v1 run; the package used training seed 0 and ended at `global_step=350000`. Original validation reconstruction was `184.70` at epoch 250 using deterministic posterior means. The package value `193.7491` at epoch 249 came from sampled-posterior validation in the recorded run and is not comparable. Both values are references only. Current `validation_step` passes `sample=False`, so package validation uses deterministic posterior means. The recorded run predates this fix and the resumed-sampler fix described below; rerunning the documented command would not reproduce `193.7491` exactly.

S5 (full-run training and test evaluation in the [protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md)) has not run. PixelVAE S5 is not claimed, and PixelVAE training reproduction is not verified. The planned S5 three-seed trainings for both systems were deferred because the campaign estimate exceeded the available compute limit. The [issue #31 compute-budget comment](https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806) records this deferral. The shared-cache CanvasVAE comparison uses embeddings generated from untrained original PixelVAE weights and does not evaluate package PixelVAE training.

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

### Build the Filtered Crello Image Records

The [pinned shell entry point](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/bin/generate_crello_image.sh) invokes `python -m preprocess crello-image`. The command is implemented by [the image transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_image.py). It reads raw split `SequenceExample` shards and writes image-only `Example` shards. Use Python 3.11, TensorFlow 2.15.1, and Apache Beam 2.76.0 from the locked `canvas-vae` vendor extra. Run the DirectRunner on CPU with these commands from the repository root:

```bash
git submodule update --init vendor/canvas-vae
uv sync --frozen --package canvas-vae --extra vendor
mkdir -p .cache/pixel-vae/crello-v1-source .cache/pixel-vae/data
curl --fail --location \
  https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip \
  --output .cache/pixel-vae/crello-dataset-v1.zip
echo 'f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e  .cache/pixel-vae/crello-dataset-v1.zip' | sha256sum --check
unzip -q .cache/pixel-vae/crello-dataset-v1.zip -d .cache/pixel-vae/crello-v1-source
PYTHONPATH=vendor/canvas-vae/src/preprocess CUDA_VISIBLE_DEVICES='' TF_ENABLE_ONEDNN_OPTS=0 \
  uv run --frozen --no-sync --package canvas-vae --extra vendor python -m preprocess crello-image \
  --input-path .cache/pixel-vae/crello-v1-source \
  --output-path .cache/pixel-vae/data/crello-image \
  --runner DirectRunner --direct_num_workers 0 --direct_running_mode multi_threading
```

The zip contains raw `train-*`, `val-*`, and `test-*` TFRecords at its root. The transform writes `count.json` and one or more matching split shards under `.cache/pixel-vae/data/crello-image/`; each count is the number of unique selected image payloads. This is the layout consumed by `PixelVAEDataModule`.

Beam 2.76.0 defines `DirectRunner` as `SwitchingDirectRunner`; supported pipelines run with Prism. The versioned [Beam runner source](https://beam.apache.org/releases/pydoc/2.76.0/_modules/apache_beam/runners/direct/direct_runner.html) confirms this selection. The tiny run with the documented flags logged `PrismRunner`.

The data module indexes each unique image payload's TFRecord offset without decoding PNGs, then the sampler determines the package-deterministic shuffle order in the main process. A map-style PyTorch dataset reads and decodes records in that order; PyTorch's `DataLoader` preserves it when `num_workers` is greater than zero. The Crello recipe uses six workers and pinned memory; persistent workers are enabled whenever workers are active. Worker seeds use a separate CPU generator. Each `DataLoader` iterator consumes one CPU int64 base-seed value from PyTorch's default RNG. The training module makes one matching draw at each training and validation epoch boundary, keeping later CPU random values aligned.

A report-only loader benchmark on an eight-vCPU High-RAM Colab CPU measured 7.754, 10.676, 18.213, and 22.009 batches per second at 0, 2, 4, and 6 workers, respectively, after 20 order-check batches and 128 measured batches per setting. Six workers were fastest among the tested settings. The throughput probe used `pin_memory: false`. The retained regression test checks the recipe's pinned-memory setting and matches the first 20 package-deterministic batches at epochs 1 and 11. These CPU measurements do not establish GPU throughput.

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

Use `scripts/audit_hub_checkpoint.py` to audit a completed segment's `last.ckpt` after reading it back from the Hugging Face Hub. The runner writes `reports/segment-eNNN.json`; the audit script does not read it. The script pins one Hub revision and checks metadata against downloaded checkpoint bytes. It checks `--expected-sha256` when supplied, requires `global_step > --minimum-step`, and requires CUDA to be unavailable. Manually compare the audit's `checkpoint_sha256` with `checkpoints.last_sha256` and its `global_step` with `segment.global_step` in the runner report. A successful training exit code alone does not confirm that every artifact reached the Hub. The best checkpoint stores weights only and cannot resume optimizer state; continue segmented training from the verified full-state `last.ckpt`.

Scheduler cadence — not applicable: the source and package use a constant learning rate. Importance sampler — not applicable: the loop has no timestep or importance sampler. EMA — not applicable: the loop has no exponential moving average. AMP — not applicable: the recipe uses float32. Multi-worker loader — applied: the sampler fixes record order in the main process. The six-worker persistent loader with pinned memory preserved the first 20 batch tensor hashes at epochs 1 and 11 in the CPU regression test.

## Completed Crello v1 PixelVAE Run

The package checkpoint rule selects minimum `val_reconstruction` (`mode=min`, top-k 1). In the recorded continuation, `best.ckpt` was selected at epoch 249 (SHA-256 `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`). It was selected using sampled-posterior validation, a deviation from the original's deterministic-mean selection; the exported encoder depends on that choice. The encoder was re-exported from the selected checkpoint on CPU. The exported `model.safetensors` SHA-256 is `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`; `PixelVAEEncoder.from_pretrained` loaded it with 2,880,032 parameters. Export with:

```bash
uv run --package pixel-vae --extra training models/pixel-vae/scripts/export_trained_encoder.py \
  --checkpoint /path/to/epoch-249-best.ckpt \
  --output-dir .cache/pixel-vae/encoder-best-e249-20261108
```

The final epoch-250 `last.ckpt` SHA-256 is `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87` and stores the final training state. The existing cache contains 114,007 × 256 float32 rows. It was generated from the epoch-250 `last.ckpt` encoder file (SHA-256 `dd8ed8e8a85bae96eeba8a2cbf5cfdad7b5e23a5456f856039db63ef279a53f2`), not the re-exported epoch-249 best encoder. Its array SHA-256 is `33f1d1757bc8ae1a6e46d0dd8bd1230c21b133d881a65dfcf1f012ff9738bb3e`; its encoder-state SHA-256 is `1c3345e75e57add7fc529dd02fee01a5550f48ac09b8e798e51ca570fa9ac62e`. No cache was regenerated from the best-checkpoint encoder. CanvasVAE uses the separate untrained-original cache `fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a`, so its comparison is unaffected.

The first PixelVAE run covered epochs 1–227 using Torch 2.13.0 with the CUDA 13.0 build. The continuation used Torch `2.13.0+cu126` because the available driver could not initialize the locked CUDA 13.0 build. Its `pip freeze` SHA-256 is `67127ccc31819828846e2cd41bada86d37800801436cb47461aff4d4a248f149`. Eighteen package records differed from the first run's environment record (`5edbe286b3b2851230c36f3b92c659e95923e8e3f7780810f5d52c2e97cc274a`). All differences are Torch or CUDA runtime distributions.

The final resumed segment ran with `--trainer.max_epochs=-1 --trainer.max_steps=350000` from the epoch-245 `last.ckpt` at `global_step=343000`, with sampler start epoch 245. `reports/segment-e250.json` records steps 343,000 to 350,000. The 7,000-step difference is five epochs at 1,400 steps each, advancing boundary 245 to 250. Lightning increments `fit_loop.epoch_progress.processed` at train-epoch end before saving the boundary checkpoint. The epoch-250 `last.ckpt` records `epoch_progress.completed=250`, `started=255`, and `processed=256`. Restored Lightning counters can run ahead of absolute epochs, so the segmented script passes `--trainer.max_epochs=-1` and uses the absolute `max_steps` target for completion.
The retained full-size report `reports/segment-e227.json` records steps 270,200 to 317,800. The 47,600-step difference is 34 epochs at 1,400 steps each, advancing boundary 193 to 227. The attempt planned for endpoint 214 has no retained ending report, so its final global step and stop point are unknown. The next segment resumed from boundary 193, making absolute epoch 194 its first training epoch, not a saved endpoint. No completed epoch is evidenced as repeated. The reports do not show whether the stopped process made unsaved updates during epoch 194; any such updates would have been replayed from step 270,200. Reports for planned endpoints 10, 78, and 146 are not retained here, so their saved step boundaries are not established.
Timing and configuration records identify first resumed absolute epochs 11, 79, 147, 194, 228, and 246. The first resumed timing/config records at 11, 79, 147, and 228 are not substitutes for saved checkpoint boundaries. The retained configs for resumes at 11, 79, 194, 228, and 246 used six workers; the epoch-147 launch config is not retained, but its pinned source config sets `num_workers: 6`. No resumed segment is recorded with zero workers. Workers prefetched before the callback set the configured epoch, so each first resumed epoch consumed epoch-0 order. At resumed epochs 79, 147, 194, 228, and 246, the logged `sampler_epoch` values do not reflect the epoch-0 order the loader actually used. The epoch-11 timing file has no `sampler_epoch` field; its Lightning-counted `epoch` is 12 for absolute epoch 11. The current timing callback writes `epoch` as `sampler_epoch + 1`, unlike the recorded epoch-11 file. The retained production timing file contains 18 full-epoch records for absolute epochs 228–245; its first row records `sampler_epoch=227` and 1,019 seconds. The retained reports establish saved boundaries 193, 227, 245, and 250; reports for the earlier planned endpoints and the attempt targeting 214 are not retained here.

Current `PixelVAEDataModule.setup()` sets the configured sampler epoch before a `DataLoader` iterator is created and workers prefetch. `test_trainer_resume_starts_with_the_configured_sampler_epoch` verifies the resumed order with a real two-worker Trainer resume. The recorded run predates this fix.

The recipe used `precision=32-true` and did not override Torch's TF32 backend flags. With Torch 2.13.0 defaults, matmul TF32 was disabled and cuDNN TF32 was enabled; `32-true` disables autocast but does not disable cuDNN TF32. The GPU used by the first run could therefore use TF32 for convolution, while the continuation GPU has no TF32 hardware support.

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

| Dataset   | System   | Status                                                                                                                                                                     | Seed scope                                                  | Primary metrics                                                                                  | Loss evidence                                                                                | Artifact summary                                                                                                                                                                                                                                                                                         |
| --------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crello v1 | original | `not-yet-run (planned S5 three-seed trainings deferred for compute limit: https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806)` | PixelVAE training seed 0; no PixelVAE evaluation seed scope | Deterministic validation reference only: `184.70` at epoch 250                                   | Original training finished; the planned PixelVAE test evaluation was not run.                | No PixelVAE S5 test artifact.                                                                                                                                                                                                                                                                            |
| Crello v1 | package  | `not-yet-run (planned S5 three-seed trainings deferred for compute limit: https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806)` | PixelVAE training seed 0; no PixelVAE evaluation seed scope | Sampled-posterior validation reference only, not comparable to original: `193.7491` at epoch 249 | Package training finished at epoch 250 / step 350,000; PixelVAE test evaluation was not run. | Best `1f2db7a9013ad296307ec1413701392596884dcd80c49e863e901dfba2d27c65`; last `ab2c0249f08302c5fd81e5fdaf01cf1d16e9c866e2877fe69311077e32eaae87`; best encoder `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`; cache from epoch-250 last encoder; no cache from best (hashes above). |

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

The reader-reproducible path is the direct `traingen fit` command. Build `.cache/pixel-vae/data/crello-image/` from the public archive with the commands in [Build the Filtered Crello Image Records](#build-the-filtered-crello-image-records). The training command uses the checked-in 250-epoch Crello recipe and needs no Colab launcher or Hub inputs.

```bash
uv run --package pixel-vae --extra training traingen fit \
  --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
```

The config sets `default_root_dir` to `.cache/pixel-vae/training-runs/crello` and leaves the checkpoint callback's `dirpath` unset. Lightning writes `best.ckpt` under `lightning_logs/version_N/checkpoints/` beneath that root; use that file as the input to the encoder-export command above.

### Recorded Segmented Campaign

Recorded segments used the [runner](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/scripts/run_crello_training_segment.sh). `colab-gpu-run` is the private Colab job launcher used for the recorded campaign. The segmented route needs that launcher and private staged Hub inputs; readers without them should use the direct command above. The runner requires `uv 0.12.22` and syncs the locked training dependencies. Production runs use batch size 64 and one visible CUDA device.

The planned cumulative epoch endpoints are 010, 078, 146, 214, and 250. `reports/segment-e227.json` records steps 270,200 to 317,800. The difference is 47,600 steps, or 34 epochs, advancing boundary 193 to 227. `reports/segment-e250.json` records steps 343,000 to 350,000. The difference is 7,000 steps, or five epochs, advancing boundary 245 to 250. The attempt targeting 214 has no retained ending report, so its final step and stop point are unknown. The next segment resumed from boundary 193, making absolute epoch 194 a resumed training epoch, not a saved endpoint. No completed epoch is evidenced as repeated. Unsaved updates during epoch 194, if any, would have been replayed from step 270,200. Reports for planned endpoints 010, 078, and 146 are not retained here. The runner reads the endpoint from the job name. It checks only that the endpoint follows the resume epoch and is at most 250. It does not enforce the planned schedule or the runner script's header comment (at most 68 epochs per segment). The runner sets `max_epochs=-1` and stops at the absolute `max_steps` target.

The runner expects exactly one `<job-name>/logs/job.log` beneath `COLAB_JOB_WORK_ROOT`. If unset, that directory is `$PIXELVAE_WORK_ROOT/colab-jobs`; `PIXELVAE_WORK_ROOT` defaults to the current directory. The job name is `pixelvae-<40-hex-source-sha>-e<3-digit-end-epoch>-a<unique-attempt-id>`. The runner checks that its commit matches the source archive and reads the epoch endpoint from the name. A token file must exist at `HF_TOKEN_PATH`; the runner defaults that path under `PIXELVAE_WORK_ROOT`, and `PIXELVAE_HF_TOKEN_PATH` can set another path.

| `--hub-input` name  | Required when                                                                                   | Purpose                                                                                                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source`            | Every invocation                                                                                | HTTPS archive for the exact 40-character source commit.                                                                                                                                       |
| `data_tar`          | Every production run and normal CPU rehearsal; omitted only when preparing the tiny CPU fixture | `hf://models/<owner>/<repo>/<file>` archive of filtered Crello image records.                                                                                                                 |
| `attribution`       | Every production run and normal CPU rehearsal; omitted only when preparing the tiny CPU fixture | `hf://models/<owner>/<repo>/<file>` lockstep attribution report; the runner verifies its SHA-256 against the pinned value `99030e6a1b53303898851707cb1f31b67a681e59b295f20408635b89738d1b17`. |
| `resume` and `best` | Together on every resumed segment; omit both for the first segment                              | `hf://models/<owner>/<repo>/<file>` full-state `last.ckpt` and validation-best `best.ckpt`; local fallbacks are `PIXELVAE_RESUME_FROM` and `PIXELVAE_BEST_FROM`, which must both exist.       |

The source input is the only `--hub-input` value that is not an `hf://` model-repository URI. The data archive keeps records under `original-run-2/crello-image/`, including `count.json` and the train, validation, and test TFRecord shards. The runner strips those two leading path components, leaving `count.json` and the records directly in its data directory. A resumed job downloads both checkpoint inputs from the preceding attempt or reads both local fallback files; it requires the full-state checkpoint to end at an epoch boundary.

The first production segment runs a 300-step lockstep, which records matched updates from both systems without a pass/fail gate. After changing the training path, update the pinned attribution-report SHA-256 and set `PIXELVAE_RUN_LOCKSTEP=1`. Later segments check that hash and skip the lockstep unless this setting requests another run. The runner scans its detailed log and evidence files for token-shaped strings before reporting completion; it does not upload Hub files itself.

The direct command runs the full 250-epoch config in one process. The completed segmented run selected epoch 249 by minimum `val_reconstruction`. The encoder was re-exported on CPU from that best checkpoint; its `model.safetensors` SHA-256 is `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`. For S5 status, see Reproduction Results.
