---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LACE
---

# LACE Training

This document records the package-local LACE training recipe and the completed S0-S4 reproduction evidence for PubLayNet and RICO25. S5 full-run statistical reproduction is outside this issue's scope and has not been run.

Run commands from the repository root. Generated data, logs, checkpoints, runtime metadata, and evaluation artifacts stay under `.cache/lace/` and are not committed.

## Install

CPU-only checks and repository tests use the lockfile environment:

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
```

The vendor checkout is used only by the parity harness. GPU evidence uses the package-local audited runtime described below; it does not modify `uv.lock` or `pyproject.toml`.

## Data

The approved sources are [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) and the `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration of [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico), as specified by [the repository data-source policy](https://github.com/creative-graphic-design/design-generators/blob/main/docs/data-sources.md). LACE's own processed stream is the reproduction input for S1-S4:

```text
.cache/lace/data/<dataset>-max25/processed/{train,val,test}.pt
```

The stream is produced by LACE's `InMemoryDataset` preprocessing from the approved source. The LayoutDM processed files are a compatibility cross-check only; no S1-S4 stage consumes them.

| Dataset   |                 LACE processed layouts |                    LACE processed elements | S1-S4 stream                                                     |
| --------- | -------------------------------------: | -----------------------------------------: | ---------------------------------------------------------------- |
| PubLayNet | train 315,757; val 16,619; test 11,142 | train 3,033,717; val 159,541; test 119,402 | `.cache/lace/data/publaynet-max25/processed/{train,val,test}.pt` |
| RICO25    |   train 358,510; val 2,109; test 4,218 |   train 3,954,250; val 23,650; test 47,129 | `.cache/lace/data/rico25-max25/processed/{train,val,test}.pt`    |

S0 compared record ids and exact `(bbox, label)` element tuples as well as file SHA-256 values. Every compared split file hash differs from the LayoutDM copy. PubLayNet train, val, and test and RICO25 val and test contain the same records in the same order and the same element tuples; their differences are serialization-only. RICO25 train is different in content: LACE repeats the approved-source train stream 10 times, while the LayoutDM copy contains one source pass. The LACE stream above is therefore the sole reproduction stream, including its deliberate RICO25 train duplication.

## Configs

Training configs live under `models/lace/configs/training`.

| Config                              | Dataset   | Seed mode     | Purpose                                                             |
| ----------------------------------- | --------- | ------------- | ------------------------------------------------------------------- |
| `lace_publaynet.yaml`               | PubLayNet | default       | Plain-PyTorch recipe values wrapped by the package training module. |
| `lace_rico25.yaml`                  | RICO25    | default       | Plain-PyTorch recipe values wrapped by the package training module. |
| `lace_publaynet_deterministic.yaml` | PubLayNet | deterministic | CPU deterministic checks and controlled diagnostics.                |
| `lace_rico25_deterministic.yaml`    | RICO25    | deterministic | CPU deterministic checks and controlled diagnostics.                |

## Scheduler and Recipe Notes

The vendor entry point is `vendor/lace/train.py`, a plain-PyTorch loop. The verified recipe values are dimension 1024, four transformer layers, feature dimension 2048, batch size 256, learning rate `1e-5`, 1,000 diffusion timesteps, gradient clipping at norm 1.0, and Adam with betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and no AMSGrad. There is no scheduler. EMA is active from initialization with `mu=0.9999` and updates after `optimizer.step()`.

The S3 natural layer runs the package production `LightningModule.training_step` and `configure_optimizers` through a Lightning `Trainer`; the vendor side runs the complete adapter for the code in `vendor/lace/train.py` in its source order. Natural runs use each system's own seeded random draws with no per-step RNG restoration or injected shared tensors. The synchronized layer is a separate diagnostic that restores the vendor boundary state and RNG only to compare the two implementations under identical boundary conditions.

The repository lockfile pins a CUDA build that cannot initialize on the verified host driver. GPU evidence therefore uses the package-local audited runtime below. All CPU-only checks and tests use the lockfile environment.

## Audited Runtime

The audited runtime is outside the repository at `<LACE_AUDIT_VENV>`, with the package installed editable from this worktree and all other dependencies pinned to the lockfile versions. The exact setup command was:

```bash
LACE_AUDIT_VENV=<LACE_AUDIT_VENV> python3.11 -m venv "$LACE_AUDIT_VENV"
LACE_AUDIT_VENV=<LACE_AUDIT_VENV> "$LACE_AUDIT_VENV/bin/python" -m pip install --no-deps -e models/lace
```

The runtime has Python 3.11.15, `torch-2.8.0+cu128` (the same torch version as the lock, with the driver-compatible CUDA tag), and `torchvision-0.23.0+cu128`. The torch wheel SHA-256 is `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`; the torchvision wheel SHA-256 is `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`. The complete `pip freeze` SHA-256 is `8c975e4eae4858016484ec20bde31eb723fb3acdf1c7094fa4c2667c4f1f3c5`. The package's audited-runtime test count was 35/35 passed before GPU evidence. A CUDA kernel smoke passed on the assigned GPU; its process-local device is reported as `cuda:0` because of `CUDA_VISIBLE_DEVICES` remapping.

## Seed Policy

S0-S3 use seed 42975 for the real loader order and natural S3 uses seed 10000 for each system's own training random stream. S4 uses sampling seeds `20260000 + test batch index` for both evaluation paths. These are evaluation and parity seeds, not S5 training-seed evidence. No S5 training-seed claim is made.

## Validation Stages

| Stage | Scope                                                                                                 | Purpose                                                                                                                      |
| ----- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config, independent initialization, topology, optimizer, EMA, and processed-data compatibility | Establishes package/vendor static parity and identifies the actual LACE reproduction stream.                                 |
| S1    | Real loader batch before optimizer mutation                                                           | Compares prepared inputs, model output, reconstruction, every loss component, and total loss.                                |
| S2    | One optimizer step on a real loader batch                                                             | Compares gradients, clipping, Adam state, parameters, learning rate, and EMA.                                                |
| S3    | 300 real loader steps at batch size 256                                                               | Compares natural production paths and a separate synchronized layer with per-step traces and repeat envelopes.               |
| S4    | Full deterministic loader and full TEST evaluation path                                               | Compares stream order and preprocessing, then the vendor evaluation entry point and package pipeline over every TEST layout. |
| S5    | Full-run statistical comparison                                                                       | `not-yet-run` for this issue; it requires a separate user decision.                                                          |

## Stage Evidence

The stage evidence rows cite the source commit recorded by each run. Exact means the machine-written comparison returned equality for the stated population; diagnostic measurements are not promoted to exact claims.

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Artifact                                                                                                                                                                                                  | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> bash -c '"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s0 --dataset <publaynet\|rico25> --lace-data-root .cache/lace/data --layoutdm-data-root <LAYOUTDM_DATA_ROOT> --output-root .cache/lace/stage-evidence/head-s0-<dataset>'`                                                                                                                                             | `.cache/lace/stage-evidence/head-s0-publaynet/s0-static/summary.json` and `.cache/lace/stage-evidence/head-s0-rico25/s0-static/summary.json`                                                              | Exact independent initialization and topology parity; 46,105,610 PubLayNet parameter scalars and 46,146,590 RICO25 parameter scalars; Adam defaults and EMA activation match. All compared split hashes differ; PubLayNet and RICO25 val/test differ only in serialization, while RICO25 train is a 10x LACE duplication of the approved-source stream. Source commit `5d20d8ddafcd607cc382940be0b3fe1f37a06981`.                                                                                                                                                                                                                             |
| S1    | `LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> bash -c '"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s1 --dataset <publaynet\|rico25> --device cuda:0 --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/head-fdf69e2-s1-<dataset>'`                                                                                                                                              | `.cache/lace/stage-evidence/head-fdf69e2-s1-publaynet/s1-fixed-batch/summary.json` and `.cache/lace/stage-evidence/head-fdf69e2-s1-rico25/s1-fixed-batch/summary.json`                                    | Exact agreement (`exact=true`) for the real 256-layout train batch, including model outputs, vendor-operation-order reconstruction, alignment loss, all loss components, and total loss, for both datasets. Source commit `fdf69e2eb3a000a9eb37cc0bec4b8cd9961ebb0a`.                                                                                                                                                                                                                                                                                                                                                                         |
| S2    | `LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 bash -c '"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet\|rico25> --device cuda:0 --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/head-fdf69e2-s2-<condition>-<dataset>'`                                                                                                  | `.cache/lace/stage-evidence/head-fdf69e2-s2-{natural,deterministic,warn-recorded,sdpa-math}-{publaynet,rico25}/s2-optimizer-step/summary.json`                                                            | Adam implementation, defaults, single-tensor/foreach/fused options, clipping order, weight decay, epsilon, and float32 gradient dtypes match; same-gradient probes produce zero parameter and optimizer-state differences. Natural and deterministic/CUBLAS conditions remain non-bitwise because memory-efficient SDPA backward is nondeterministic; `warn_only=True` identifies `attention_backward.cu:775`, and forcing math SDPA does not make gradients bitwise. The natural one-step parameter maximum difference is `1.862645149230957e-09` across the parameter population. Source commit `fdf69e2eb3a000a9eb37cc0bec4b8cd9961ebb0a`. |
| S3    | `LACE_AUDIT_FREEZE_SHA256=8c975e4eae4858016484ec20bde31eb723fb3acdf1c7094fa4c2667c4f1f3c5 LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> bash -c '"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s3 --dataset <publaynet\|rico25> --device cuda:0 --steps 300 --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/head-207829a-s3-lockstep-b256-<dataset>'`                           | `.cache/lace/stage-evidence/head-207829a-s3-lockstep-b256-publaynet/s3-lockstep/summary.json` and `.cache/lace/stage-evidence/head-207829a-s3-lockstep-b256-rico25/s3-lockstep/summary.json`              | PubLayNet natural first loss divergence over the 1e-3 report threshold is step 23; synchronized comparison is exact after each boundary synchronization with zero post-sync parameter, EMA, and optimizer-state maximum differences. Both populations are 300 optimizer steps at batch size 256. The RICO25 result is recorded in the artifact after its run completes. Source commit `207829a3752759389ea9f34e29d743a741a2a110` for the PubLayNet run.                                                                                                                                                                                       |
| S4    | `LACE_AUDIT_FREEZE_SHA256=8c975e4eae4858016484ec20bde31eb723fb3acdf1c7094fa4c2667c4f1f3c5 LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> bash -c '"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s4 --device cuda:0 --evaluation-batch-size 256 --lace-data-root .cache/lace/data --checkpoint-root .cache/lace/original/model --fid-root .cache/lace/fidroot --output-root .cache/lace/stage-evidence/head-7df18a5-s4-full'` | `.cache/lace/stage-evidence/head-7df18a5-s4-full/s4-loader-evaluation/summary.json` (SHA-256 `a63bd07749a2e401139344c69426f01cd12fe4ed7598540c46cd004ce9d05d18`) plus per-dataset `evaluation.json` files | Exact full TEST evaluation-path parity for PubLayNet 11,142/11,142 layouts and RICO25 4,218/4,218 layouts. Each system produced the full layout count, zero out-of-bounds elements, bitwise-equal predictions, and metric-equal values computed by the vendor metric functions with its per-batch reduction. Vendor entry point `vendor/lace/test.py::test_layout_cond`; package entry point `models/lace/src/lace/pipeline_lace.py::__call__`; source commit `7df18a55b70e896f2704588d23968eac179bdf8a`.                                                                                                                                     |
| S5    | `not-yet-run`                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `.cache/lace/full-run/<dataset>/manifest.json; evaluation-path-parity: https://github.com/creative-graphic-design/design-generators/issues/423`                                                           | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`; full-run training and statistical evaluation are outside this issue's S0-S4 scope.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

## Reproduction Results

S0-S4 reproduction evidence covers the real LACE processed stream for PubLayNet and RICO25. S1 is bitwise exact on one real batch per dataset, S2 identifies an unavoidable memory-efficient SDPA backward difference while matching the optimizer path, S3 records both natural and synchronized 300-step populations, and S4 is exact over both full TEST splits. S4 uses the vendor evaluation entry point and the package pipeline with the same released checkpoint, inputs, settings, and sampling seeds; both systems' metrics are computed by the vendor metric functions with the vendor's per-batch reduction. The S4 evidence source commit `7df18a55b70e896f2704588d23968eac179bdf8a` is retained as an ancestor of the final PR head; main is merged without rebasing after evidence capture. S5 remains outside this issue's scope.

| Dataset   | System               | Status                                                                                  | Seed scope                                   | Primary metrics                                                                                     | Loss evidence                                               | Artifact summary                                                                                            |
| --------- | -------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| PubLayNet | original and package | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity seeds only                            | S4 exact: alignment 0.1216632277, overlap 4.4502921104, FID 4.8615175926, maximum IoU 0.3281917111  | S1 exact; S2 diagnostic; S3 natural and synchronized traces | `.cache/lace/stage-evidence/head-7df18a5-s4-full/s4-loader-evaluation/publaynet-evaluation/evaluation.json` |
| RICO13    | original and package | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | no seed scope; dataset is outside this issue | no S0-S4 evaluation claim                                                                           | no S0-S4 training evidence                                  | `https://github.com/creative-graphic-design/design-generators/issues/423`                                   |
| RICO25    | original and package | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity seeds only                            | S4 exact: alignment 0.1111927703, overlap 83.5333480835, FID 3.3846049958, maximum IoU 0.3358800030 | S1 exact; S2 diagnostic; S3 natural and synchronized traces | `.cache/lace/stage-evidence/head-7df18a5-s4-full/s4-loader-evaluation/rico25-evaluation/evaluation.json`    |

### Comparison Scope

| Dataset   | System | Evaluator                                                                                        | Test split                         | Checkpoint-selection rule                             | Sample count                                     |
| --------- | ------ | ------------------------------------------------------------------------------------------------ | ---------------------------------- | ----------------------------------------------------- | ------------------------------------------------ |
| PubLayNet | both   | `vendor/lace/test.py::test_layout_cond` plus package `LacePipeline.__call__`; vendor metric code | approved-source LACE TEST stream   | authors' released LACE checkpoint `publaynet_best.pt` | 11,142 layouts per evaluation seed               |
| RICO13    | both   | not applicable for this issue; no approved LACE RICO13 checkpoint is present                     | not applicable; outside this issue | not applicable; outside this issue                    | 0 layouts; outside this issue's S0-S4 population |
| RICO25    | both   | `vendor/lace/test.py::test_layout_cond` plus package `LacePipeline.__call__`; vendor metric code | approved-source LACE TEST stream   | authors' released LACE checkpoint `rico25_best.pt`    | 4,218 layouts per evaluation seed                |

## Regeneration Metadata

The authors' released checkpoints came from [the LACE model archive](https://huggingface.co/datasets/puar-playground/LACE/resolve/main/model.tar.gz). PubLayNet uses `.cache/lace/original/model/publaynet_best.pt` with SHA-256 `d13ea9a64d913d910a35db6204a4be7b115101ffea54874683ef8c9f98607ff4`; RICO25 uses `.cache/lace/original/model/rico25_best.pt` with SHA-256 `9c2f2198dbab7d530363a413d98efa779bc5a1ebcdf55d1c1fc64175361abe93`.

Each machine-written stage record includes its own source commit, vendor commit `3df36879a1e80cce58affa4aadeeb768f676c7f1`, runtime metadata, command inputs, and artifact paths. The S4 evaluation records additionally include the checkpoint URL and SHA-256, same-weight and same-input hashes, per-system prediction-file SHA-256 values, per-system prediction and out-of-bounds counts in the original normalized center-`xywh` frame, evaluator source commits, settings, sampling seeds, metrics, the vendor metric source used for both systems, and measured runtime. The S4 summary SHA-256 is `a63bd07749a2e401139344c69426f01cd12fe4ed7598540c46cd004ce9d05d18`; its PubLayNet and RICO25 evaluation-record SHA-256 values are `2acc33d57d4f40e6a5198d4c524d1d11fc0663892e1fa4236d463ccab0b17aec` and `e597b8c578a09d99b184a8ed22a618816c9732fa451c97bd2a9495daa70951ef`.

```text
.cache/lace/stage-evidence/
.cache/lace/data/
.cache/lace/original/model/
.cache/lace/fidroot/
```

## Training Commands

Run the package tests and repository CPU-only checks in the lockfile environment:

```bash
UV_FROZEN=1 uv run --package lace pytest models/lace/tests -q
UV_FROZEN=1 uv run --package lace scripts/run_member_tests.sh models/lace
```

Run the regular package recipe with the audited runtime after S0-S4 review:

```bash
LACE_AUDIT_VENV=<LACE_AUDIT_VENV> CUDA_VISIBLE_DEVICES=<gpu> bash -c '"$LACE_AUDIT_VENV/bin/python" -m lightning.pytorch.cli models/lace/configs/training/lace_<publaynet\|rico25>.yaml'
```

The full-run command is intentionally not provided as a reproduction claim in this issue. S5 requires a separate user decision.
