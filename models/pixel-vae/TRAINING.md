# PixelVAE Training

CPU conversion and S0–S4 parity passed against the original TensorFlow PixelVAE on the Crello v1 archive, using one deterministic seed-0 untrained state copied to PyTorch. All 17 held-out metrics met their frozen limits, and exact state-mapping and decode checks passed. This package does not train or evaluate trained checkpoints; GPU full training and S5 are out of scope. See [PARITY_PROTOCOL.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/PARITY_PROTOCOL.md) for the fixed CPU metrics and [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/REPRODUCING.md) for commands.

Run commands from the repository root. Keep Crello data, traces, checkpoints, conversion outputs, and embeddings under `.cache/pixel-vae/` during execution; they remain private and are not committed. Persist required run evidence only in the project's private artifact repository.

## Install

```bash
uv sync --package pixel-vae --extra vendor --extra parity
```

The `vendor` extra provides TensorFlow CPU 2.15.1 for original-code conversion and parity. The package provides conversion and encoder inference but no package-local training loop, so it has no training extra.

## Data

| Dataset   | Source                                                                                                                                                                                                       | Config or path                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Crello v1 | [Pinned original GCS archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip), 2,989,732,284 bytes, SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e` | Private archive and extracted records under `.cache/pixel-vae/`; train/val/test document counts are 18,768/2,315/2,278. |

## Configs

There is no package-local training config in this CPU port. `PixelVAEConfig` records the production architecture and objective parameters for conversion and inference.

| Config             | Dataset   | Seed mode                                                        | Purpose                                                    |
| ------------------ | --------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| `PixelVAEConfig()` | Crello v1 | deterministic source initialization seed 0 for conversion parity | 256 × 256 RGBA model with 256-dimensional latent posterior |

## Scheduler and Recipe Notes

The original PixelVAE model uses MobileNetV2, batch size 64, 250 epochs, Adam at 1e-4, KL weight 100, and L2 weight 1e-6. This package does not implement the original training loop. Scheduler cadence — not applicable: this package has no training scheduler. Sampler — not applicable: this package has no timestep or importance sampler. EMA — not applicable: this package has no EMA path. AMP — not applicable: CPU parity uses float32 and this package has no AMP path. Multi-worker loader — not applicable: this package has no package-local training loader.

## Seed Policy

CPU parity uses deterministic TensorFlow initialization seed 0 for one untrained state copied to PyTorch. It is not a training-seed result or an evaluation-seed comparison.

## Validation Stages

| Stage | Scope                                          | Purpose                                                                                       |
| ----- | ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity     | Verify source config and strict one-to-one TensorFlow weight mapping.                         |
| S1    | Fixed-batch pre-optimizer trace parity         | Compare exact decode, posterior statistics, decoder logits, and loss terms.                   |
| S2    | One optimizer-step parity                      | Compare gradients, Keras Adam moments, and updated parameters.                                |
| S3    | Short deterministic multi-batch run            | Compare a fixed three-step CPU trace.                                                         |
| S4    | Crello source and encoder serialization checks | Audit all element PNG dimensions, compare preprocessing, and verify posterior-mean save/load. |
| S5    | Full-run statistical comparison                | Not run; full training and S5 are out of scope for this package.                              |

## Stage Evidence

| Stage | Command                                                                                                                 | Artifact                                                                             | Result                                                                                                                                 |
| ----- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh` | `input-selection-plan.json`, source state record, calibration input sidecars         | Pass; exact config, complete key map, and bitwise state tensors for one untrained state.                                               |
| S1    | Same ordered CPU acceptance command                                                                                     | `diagnostic/s1-diagnostic.json`, S0 input selections, `calibration/repeat-1..3.json` | Pass; all 6 held-out forward and loss metrics met their frozen limits.                                                                 |
| S2    | Same ordered CPU acceptance command                                                                                     | `calibration/repeat-1..3.json`                                                       | Pass; all 6 held-out gradient and optimizer metrics met their frozen limits.                                                           |
| S3    | Same ordered CPU acceptance command                                                                                     | `calibration/repeat-1..3.json`                                                       | Pass; all 4 held-out trace metrics met their frozen limits.                                                                            |
| S4    | Same ordered CPU acceptance command                                                                                     | `heldout.json`, archive audit, encoder round-trip report                             | Pass; all 23,361 documents audited, exact RGBA decode and save/load checks passed, and the posterior-mean metric met its frozen limit. |
| S5    | Not run; this package has no training loop                                                                              | N/A                                                                                  | Not run; full training and S5 are out of scope, so no trained-checkpoint result is claimed.                                            |

## Reproduction Results

CPU conversion and parity passed on the Crello v1 source using one deterministic seed-0 untrained TensorFlow state copied to PyTorch. Three distinct canonical train-input groups set the metric bounds, and all 17 held-out metrics passed; trained-checkpoint reproduction, full-run evaluation, and embedding quality are not claimed.

| Dataset   | System   | Status                                                             | Seed scope                               | Primary metrics                                                                | Loss evidence        | Artifact summary           |
| --------- | -------- | ------------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------ | -------------------- | -------------------------- |
| Crello v1 | original | not-yet-run (no training run is in scope)                          | seed-0 untrained state only              | Paired CPU parity passed: 17/17 held-out metrics; no trained-checkpoint metric | No training loss run | `.cache/pixel-vae/parity/` |
| Crello v1 | package  | not-yet-run (no package-local training loop or trained checkpoint) | same seed-0 state copied from TensorFlow | Paired CPU parity passed: 17/17 held-out metrics; no trained-checkpoint metric | No training loss run | `.cache/pixel-vae/parity/` |

### Comparison Scope

| Dataset   | System | Evaluator                   | Test split | Checkpoint-selection rule      | Sample count   |
| --------- | ------ | --------------------------- | ---------- | ------------------------------ | -------------- |
| Crello v1 | both   | not applicable; S5 excluded | test       | no trained checkpoint selected | not applicable |

## Regeneration Metadata

The original archive SHA-256, source commit, seed, input IDs, TensorFlow version, and encoder-state digest are recorded in private reports. Keep the three calibration records, frozen limits, audit report, and fresh held-out report outside Git and in the project's private artifact repository; never store source images or derived embeddings in a public repository.

```text
.cache/pixel-vae/crello-dataset-v1.zip
.cache/pixel-vae/crello-v1-tfrecords/
.cache/pixel-vae/crello-v1-dimensions.json
.cache/pixel-vae/converted/
.cache/pixel-vae/parity/calibration/
.cache/pixel-vae/parity/heldout.json
```

## Training Commands

Run the report-only diagnostic, three distinct-input CPU calibration processes, limit freeze, and fresh held-out check in one named CPU job.

```bash
PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-sha> \
  bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh
```

Convert an original TensorFlow checkpoint or the deterministic untrained initialization.

```bash
uv run --package pixel-vae --extra vendor \
  models/pixel-vae/scripts/convert_original_checkpoint.py \
  --output-dir .cache/pixel-vae/converted
uv run --package pixel-vae python -c 'from pixel_vae import PixelVAEEncoder; encoder = PixelVAEEncoder.from_pretrained(".cache/pixel-vae/converted/encoder"); print(encoder.config.latent_dim)'
```
