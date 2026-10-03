---
icon: lucide:dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

This document records the LayoutGAN++ package-local training path and the S0-S4 evidence required before a full training run. The staged result is exact agreement with the pinned Const-layout implementation for the Magazine fixture; no S5 full-run claim is made.

Run commands from the repository root. Training data, logs, checkpoints, and staged evidence stay under `.cache/layoutganpp/`.

## Install

The lockfile environment is the source of truth for CPU-only checks and tests.

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
```

Install the package extras in the lockfile environment when preparing data or running CPU-only checks.

```bash
UV_FROZEN=1 uv sync --package layoutganpp --extra training --extra vendor --frozen
```

GPU evidence uses an audited runtime because the locked `torch 2.13.0+cu130` wheel cannot initialize on the host's CUDA 12.9-era driver. The audited runtime uses Python 3.11.15, `torch-2.8.0+cu128` (the nearest available audited-GPU-capable cu128 wheel; exact locked version has no cu128 wheel), and `torchvision-0.23.0+cu128`; all other dependencies are pinned to the lockfile. Its `pip freeze` hash is `b16da6f64a1b1b8f058437e2c1f2c2f4f5b32383355a515a2016cbed2afb5d61`.

Create the audited runtime outside tracked repository files and install this worktree into it as an editable package. The package's own tests must pass before GPU evidence is used.

```bash
UV_FROZEN=1 uv venv --python 3.11 <PACKAGE_AUDIT_VENV>
UV_FROZEN=1 uv export --frozen --package layoutganpp --extra training --extra vendor --format requirements-txt --output-file .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python <PACKAGE_AUDIT_VENV>/bin/python .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python <PACKAGE_AUDIT_VENV>/bin/python --no-deps <torch-2.8.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <PACKAGE_AUDIT_VENV>/bin/python --no-deps <torchvision-0.23.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <PACKAGE_AUDIT_VENV>/bin/python --index-url https://pypi.org/simple pytest==9.1.1 pytest-cov==7.1.0 pytest-xdist==3.8.0 beartype==0.22.9
CUDA_VISIBLE_DEVICES="" <PACKAGE_AUDIT_VENV>/bin/python -m pytest models/layoutganpp/tests -q
```

The audited-runtime package test result was `37 passed, 3 skipped`; the skips are the pre-existing original-weight parity tests without released checkpoint fixtures. One idle GPU was selected and verified with `nvidia-smi` before each GPU run.

## Data

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. It is a plain PyTorch GAN with batch size 64, 200,000 iterations, latent size 4, learning rate `1e-5`, generator/discriminator `d_model=256`, four attention heads, and eight transformer layers each. The generator update precedes the discriminator update. There is no EMA or scheduler in the vendor recipe.

Datasets are processed in the agreed order: Magazine, RICO13, and PubLayNet last. PubLayNet is approximately 107 GB, so check free disk before any transfer and do not download it during ordinary checks.

| Dataset   | Approved source                                                                                          | Config or path                                                           | Evidence scope                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Magazine  | [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine)   | `.cache/layoutganpp/data/magazine`                                       | Local HF cache, 3,919 source rows; vendor and package splits are train 3,331, val 196, test 392. |
| RICO13    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)           | `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration | Approved source; materialization deferred until Magazine and the S0-S4 path are complete.        |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default configuration                                                    | Approved source; last dataset by plan and not transferred for this S0-S4 run.                    |

Materialize the cached Magazine annotations without decoding unused image columns:

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra training models/layoutganpp/scripts/prepare_training_data.py --dataset magazine --source-arrow-dir .cache/layoutganpp/source/magazine-arrow --output-dir .cache/layoutganpp/data/magazine --source-id creative-graphic-design/magazine
```

The source manifest at `.cache/layoutganpp/data/magazine/source-manifest.json` records the four source Arrow SHA-256 hashes and the polygon-to-box rule. Magazine is exposed by the approved source as a train split; the vendor creates train, val, and test partitions from that source. The vendor loader converts polygon extrema to normalized center `xywh` for every partition, not only train.

## Configs

Training configs live under `models/layoutganpp/configs/training`.

| Config                       | Dataset            | Seed mode       | Purpose                                                  |
| ---------------------------- | ------------------ | --------------- | -------------------------------------------------------- |
| `layoutganpp_magazine.yaml`  | Magazine           | `default`       | Package-local recipe values and bounded training wiring. |
| `layoutganpp_rico13.yaml`    | RICO13             | `default`       | Package-local recipe values and bounded training wiring. |
| `layoutganpp_publaynet.yaml` | PubLayNet          | `default`       | Package-local recipe values and bounded training wiring. |
| `smoke.yaml`                 | Magazine synthetic | `deterministic` | CPU-only CLI wiring check.                               |

## Scheduler and Recipe Notes

The package training module uses Adam with learning rate `1e-5`, one generator update followed by one discriminator update, the vendor's four-dimensional latent draw per valid element, and no scheduler or EMA. Condition labels and valid-element masks come from the loader batch; the vendor path has no label or condition sampler. S1 and S2 records name the latent noise, condition labels, condition mask, padding mask, generator/discriminator losses, gradient norms, and update-order tensors.

S3 begins with a natural unsynchronized trajectory. The recorded 300-step natural trajectory stayed within the S0-S2 exact comparison tolerance, so a synchronized diagnostic was not needed. If a future change leaves that tolerance, the natural record must be retained and a synchronized layer rerun before any broader claim.

## Seed Policy

The package configs use seed `42975` for regular and smoke wiring. S1-S4 evidence uses the explicit fixed latent seed `999` for S1/S2 and the deterministic staged data/latent stream recorded in each run record. No training-seed statistical claim is made. Evaluation-path parity uses the same weights and fixed per-batch latent tensors for both systems.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                                                      |
| ----- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Confirms topology, initialized state, optimizer defaults, and vendor entry-point facts.                                                      |
| S1    | Fixed-batch pre-optimizer trace parity     | Confirms forward outputs and losses, including named GAN-specific draws and update order.                                                    |
| S2    | One optimizer-step parity                  | Confirms gradients, losses, optimizer mutation, and post-step state.                                                                         |
| S3    | Natural 300-step lockstep                  | Confirms the multi-step trajectory; a synchronized layer is conditional on natural divergence.                                               |
| S4    | Loader stream and evaluation-path parity   | Confirms train/val/test preprocessing, TEST output coordinate frame, predictions, metrics, and per-system counts.                            |
| S5    | Full-run statistical comparison            | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`; outside this task and requires a separate decision. |

## Stage Evidence

All stage records below were produced from source commit `18d78714cafe1f60c34ffe7d3498af08256fc104`; each JSON record stores its own source commit and audited-runtime metadata. Exact means bitwise equality in the recorded tensors; tolerance means the explicit `1e-6` S3 bound; approximate means an observed runtime only, not a parity claim.

| Stage | Command                                                                                                                                                              | Artifact                                                                                                                                 | Result                                                                                                                                                       |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                         | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                               | PASS; exact topology/state and first divergence `null`.                                                                                                      |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                    | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`, `.cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt`             | PASS; exact named tensor trace and first divergence `null`.                                                                                                  |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                       | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`, `.cache/layoutganpp/stage-evidence/s2-one-step/trace.pt`                   | PASS; exact one-step losses, draws, gradients, update order, and post-step state.                                                                            |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                           | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`, `.cache/layoutganpp/stage-evidence/s3-lockstep/natural.json`               | PASS; natural 300 steps, first divergence `null`, synchronized layer `not-needed`; runtime approximately 167 seconds on the audited GPU.                     |
| S4    | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval` | `.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json`, `.cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json` | PASS; all three loader splits exact, deterministic full TEST evaluation predictions max difference `0.0`, metrics identical, and 392 predictions per system. |
| S5    | —                                                                                                                                                                    | `.cache/layoutganpp/full-run/`                                                                                                           | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`.                                                                     |

## Reproduction Results

S0-S4 training reproduction is complete for the local Magazine source and synthetic fixed-batch lockstep: the package and the original implementation agree exactly on topology, fixed-batch GAN draws/losses, one optimizer step, a natural 300-step trajectory, all Magazine train/val/test loader rows, and the TEST evaluation path. RICO13 and PubLayNet have approved sources and recipe configs but no dataset transfer or full-run claim; S5 is outside this task.

| Dataset   | System   | Status                                                                                  | Seed scope                          | Primary metrics                        | Loss evidence                              | Artifact summary                     |
| --------- | -------- | --------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------------------- | ------------------------------------------ | ------------------------------------ |
| Magazine  | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged fixed seed; no S5 seed scope | S0-S4 exact; TEST prediction count 392 | S1/S2 exact GAN losses; S3 300-step losses | `.cache/layoutganpp/stage-evidence/` |
| Magazine  | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged fixed seed; no S5 seed scope | S0-S4 exact; TEST prediction count 392 | S1/S2 exact GAN losses; S3 300-step losses | `.cache/layoutganpp/stage-evidence/` |
| RICO13    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no S5 seed scope                    | not measured                           | not measured                               | `.cache/layoutganpp/`                |
| RICO13    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no S5 seed scope                    | not measured                           | not measured                               | `.cache/layoutganpp/`                |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no S5 seed scope                    | not measured                           | not measured                               | `.cache/layoutganpp/`                |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no S5 seed scope                    | not measured                           | not measured                               | `.cache/layoutganpp/`                |

The S4 TEST-split finding is first-class: the vendor and package both convert polygon extrema to normalized `xywh` for train, val, and test, so no train-only transform difference was found. The shared approved Magazine source itself provides only a train split, and the vendor's deterministic split creates the checked TEST partition.

### Comparison Scope

| Dataset   | System | Evaluator                                                                    | Test split                | Checkpoint-selection rule                                 | Sample count                      |
| --------- | ------ | ---------------------------------------------------------------------------- | ------------------------- | --------------------------------------------------------- | --------------------------------- |
| Magazine  | both   | vendor `eval.py:main` metric path and package `LayoutGANPPPipeline.__call__` | vendor-derived test split | same initialized generator weights; no trained checkpoint | 392 layouts per evaluation system |
| RICO13    | both   | not run                                                                      | not run                   | no checkpoint                                             | 0 layouts; S5 not run             |
| PubLayNet | both   | not run                                                                      | not run                   | no checkpoint                                             | 0 layouts; S5 not run             |

The evaluation artifact records normalized `xywh` output, per-system prediction counts, maximum absolute prediction difference, and metrics computed by the vendor functions `vendor/const-layout/metric.py:compute_alignment` and `vendor/const-layout/metric.py:compute_overlap`, through the evaluation loop in `vendor/const-layout/eval.py:main`. The deterministic full TEST run has 392 predictions per system; both systems report alignment `0.0024827037816104` and overlap `5.560081288522603`, with maximum absolute prediction difference `0.0`. The fixed generator initialization seed is `4242`, and both state dictionaries have SHA-256 `9cc2c23e8117867baa1bec024d43559b67dd98395466d7a14c67843fffc94915`. The latent-noise seed is `4243`; the TEST input hash is `b848437f62ea385b4364b45c1edadb9982fc773d6d56b4f85c67802cb96d43ea`, the latent-noise hash is `6d8c9f90911942662abd2f9cf53607d4b52b28530994973cad786976e6b841b5`, and the vendor/package prediction hashes are both `47f8d5b2b9931bf2585df23d0d39b3ef38f0945e9a185c9cc15dcfdad6da3135`. Both systems use `test`, `batch_size=2`, `shuffle=False`, latent size 4, and `eval` mode. These metrics are exact for this same-weights parity artifact; they are not full-run quality claims.

The converted original Magazine checkpoint was not available locally. The documented download command `UV_FROZEN=1 uv run --package layoutganpp models/layoutganpp/scripts/download_original_weights.py --output-dir .cache/layoutganpp/original --dataset magazine` failed at `https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar` with a TLS connection reset; an HTTP probe returned `503 Service Unavailable`, and no `.cache/layoutganpp/converted/layoutganpp-magazine` directory exists locally. The package inference parity tests therefore skip their absent local checkpoint fixture. Because the trained checkpoint could not be obtained in this environment, `full-392-seeded-003` remains the authoritative S4 evaluation-path record as a fixed-seed, same-random-weights parity result; it is not trained-checkpoint quality evidence. If the documented checkpoint becomes available, that later rerun must supersede this fallback record.

### S4 Attempt Comparison

The two earlier full-TEST attempts remain recorded because they are valid attempts but not replayable parity measurements. Their 392-row sample ordering, batch size, `shuffle=False` setting, normalized-`xywh` metric inputs, vendor metric functions, and package/vendor comparison were unchanged. Each invocation instead created fresh generator weights and fresh latent noise because neither random source was seeded or hashed. That is why the metric values moved even though the evaluator metadata and comparison logic did not change.

| Attempt                                      | Weight state                                                                                                                                                                    | Latent noise                                                                                  | Rows and ordering                                                                                                 | Metric inputs and evaluator                                                                                                               | Alignment              | Overlap             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------- |
| `full-392-unseeded-001`                      | fresh, unrecorded                                                                                                                                                               | fresh, unrecorded                                                                             | same 392 TEST rows and ordering                                                                                   | same normalized `xywh`; vendor metric functions; batch 2; `shuffle=False`; `eval`                                                         | `0.002390173413482908` | `5.59936015459956`  |
| `full-392-unseeded-002`                      | fresh, unrecorded                                                                                                                                                               | fresh, unrecorded                                                                             | same 392 TEST rows and ordering                                                                                   | same normalized `xywh`; vendor metric functions; batch 2; `shuffle=False`; `eval`                                                         | `0.005550596680055902` | `5.485093088706537` |
| `full-392-seeded-003` authoritative fallback | fixed random state SHA-256 `9cc2c23e8117867baa1bec024d43559b67dd98395466d7a14c67843fffc94915` from initialization seed `4242`; converted trained checkpoint unavailable locally | fixed seed `4243`, SHA-256 `6d8c9f90911942662abd2f9cf53607d4b52b28530994973cad786976e6b841b5` | same 392 TEST rows and ordering; input SHA-256 `b848437f62ea385b4364b45c1edadb9982fc773d6d56b4f85c67802cb96d43ea` | `vendor/const-layout/eval.py:main`; vendor `compute_alignment` and `compute_overlap`; normalized `xywh`; batch 2; `shuffle=False`; `eval` | `0.0024827037816104`   | `5.560081288522603` |

The seeded record is the reproducible S4 evaluation-path artifact: it has 392 predictions per system, identical prediction SHA-256 `47f8d5b2b9931bf2585df23d0d39b3ef38f0945e9a185c9cc15dcfdad6da3135`, and maximum absolute prediction difference `0.0`. The vendor's metric computation is used for both systems, so the identical metric values are an evaluation-path parity result, not a claim about trained-checkpoint quality.

## Regeneration Metadata

Evidence locations and source/data metadata:

```text
.cache/layoutganpp/runtime/pip-freeze.txt
.cache/layoutganpp/data/magazine/source-manifest.json
.cache/layoutganpp/stage-evidence/s0-static/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt
.cache/layoutganpp/stage-evidence/s2-one-step/summary.json
.cache/layoutganpp/stage-evidence/s2-one-step/trace.pt
.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json
.cache/layoutganpp/stage-evidence/s3-lockstep/natural.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-unseeded-001.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-unseeded-002.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-seeded-003.json
```

The free-disk check before any PubLayNet transfer reported approximately 2.3 TB available. No RICO13 or PubLayNet transfer was performed. Every stage record includes the vendor commit, package source commit, Python version, torch wheel/CUDA tag, and pip-freeze hash.

The attempt ledger is retained here because it affects replay: the locked cu130 runtime failed CUDA initialization on the host driver; cu128 torch 2.11.0 initialized but had no compatible kernel image and was not used; the audited cu128 torch 2.8.0 runtime passed the package tests and GPU smoke. The first Magazine materialization attempt decoded unused image columns and was interrupted; removing that column produced the recorded manifest. The first S4 script attempt lacked the vendor module path and produced no evidence. The documented Magazine checkpoint download then failed with a TLS connection reset, and the converted checkpoint was absent locally. Two subsequent full-TEST attempts used the same 392 rows, ordering, model architecture, vendor metric functions, and package/vendor comparison, but did not seed model initialization or latent noise; their metrics moved from alignment `0.002390173413482908`, overlap `5.59936015459956` to alignment `0.005550596680055902`, overlap `5.485093088706537` because each invocation drew different random generator weights and latent tensors. Both still had exact package/vendor predictions within their own run, but their inputs were not replayable and are preserved as `full-392-unseeded-001.json` and `full-392-unseeded-002.json`. The authoritative fallback fixes weight seed `4242` and latent seed `4243` and records all input, weight, latent, evaluator, checkpoint-availability, and prediction metadata in `evaluation-path.json`; it reports alignment `0.0024827037816104` and overlap `5.560081288522603`. These attempts are not full-run quality claims.

## Training Commands

Run package tests in the lockfile environment:

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package layoutganpp --extra training --extra vendor --with pytest pytest models/layoutganpp/tests -q
```

Prepare the cached Magazine source:

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra training models/layoutganpp/scripts/prepare_training_data.py --dataset magazine --source-arrow-dir .cache/layoutganpp/source/magazine-arrow --output-dir .cache/layoutganpp/data/magazine --source-id creative-graphic-design/magazine
```

Run the staged evidence with the audited runtime on one explicitly selected idle GPU only:

```bash
LAYOUTGANPP_AUDIT_VENV=<PACKAGE_AUDIT_VENV>
CUDA_VISIBLE_DEVICES=<gpu> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch
CUDA_VISIBLE_DEVICES=<gpu> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step
CUDA_VISIBLE_DEVICES=<gpu> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=<gpu> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval
```

Run the bounded package-local training wiring check without making an S5 claim:

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/smoke.yaml
```

Full-run training, conversion of trained checkpoints, and statistical evaluation are intentionally omitted until the separate S5 decision tracked by [issue 424](https://github.com/creative-graphic-design/design-generators/issues/424).
