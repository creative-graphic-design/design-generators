---
icon: lucide:dumbbell
tags:
  - Training
  - Reproducibility
  - DS-GAN
---

# DS-GAN Training

This document records the package-local reproduction checks for DS-GAN, a plain-PyTorch GAN for content-aware poster layout generation. The covered evidence uses the approved PKU PosterLayout source, the authors' `vendor/posterlayout-cvpr2023/train.sh` → `main.py` entry point, and the package's own training model. Full-run statistical reproduction is not claimed because S5 is out of scope for [issue 425](https://github.com/creative-graphic-design/design-generators/issues/425).

Run commands from the repository root. Generated data, logs, checkpoints, converted pipelines, and staged evidence stay under `.cache/ds-gan/` and are not committed.

## Install

The lockfile environment is the source of truth for all CPU-only checks and tests. GPU evidence uses the audited runtime described below; `uv.lock` and `pyproject.toml` are unchanged.

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
git diff --exit-code -- uv.lock
UV_FROZEN=1 uv sync --package ds-gan --extra vendor --frozen
```

The locked environment pins `torch 2.13.0+cu130`, which cannot initialize on the verified host driver. The audited runtime uses Python 3.11.15, `torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (SHA-256 `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`), and `torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (SHA-256 `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`). Torch 2.8.0 was selected because the 2.11.0 cu128 build has no usable V100 sm70 kernels. All other dependencies remain at lockfile versions. The audited runtime `pip freeze` SHA-256 is `e2bb5a0088f2fc67990b1f0e1c132dfbec18fcfd70403b5045f20f02448454da`.

Create the audited runtime outside the repository and install this worktree into it as an editable package. The lockfile environment remains the environment for CPU-only checks and tests.

```bash
UV_FROZEN=1 uv venv --python 3.11 <DSGAN_AUDIT_VENV>
UV_FROZEN=1 uv export --frozen --package ds-gan --extra vendor --format requirements-txt --output-file .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python <DSGAN_AUDIT_VENV>/bin/python .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python --no-deps <torch-2.8.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python --no-deps <torchvision-0.23.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python --no-deps --editable .
CUDA_VISIBLE_DEVICES="" <DSGAN_AUDIT_VENV>/bin/python -m pytest models/ds-gan/tests -q
```

The audited package tests pass `35/37` with `2` skips. The audited runtime uses Python 3.11.15, `torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (SHA-256 `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`), and `torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (SHA-256 `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`). Its `pip freeze` SHA-256 is `448ed2d8331517208adc497253b15d0daee98059049b784fd70e7360e1e615a5`. The lockfile environment was used for all CPU-only checks and tests. Before GPU evidence, run the CUDA kernel smoke on the selected device and record the wheel and freeze hashes in the run records.

## Data

The approved source is [creative-graphic-design/PKU-PosterLayout](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout), configuration `default`, revision `af3a6fdadeaf604fec8a735fd05043e44b93879d`, already present in the Hugging Face cache. It contains 9,974 train rows and 905 TEST rows.

The read-only bridge maps `inpainted_poster` to the vendor train poster, `canvas` to the vendor TEST canvas, `pfpn_saliency_map` to PFPNet saliency, `basnet_saliency_map` to BASNet saliency, and `annotations` to the vendor CSV annotations. The approved source provides every required field; `missing_source_fields` is empty. Both saliency maps are converted to grayscale, merged by pixelwise maximum, and resized to 350x240. Train annotations filter `INVALID` and convert pixel `ltrb` boxes to normalized center `xywh`. The bridge manifest records these mappings, source and generated-file hashes, split counts, and transformations.

| Dataset          | Source                                                                                                               | Split and coverage                                                                                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PKU PosterLayout | [creative-graphic-design/PKU-PosterLayout](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout) | 9,974 train rows for S1-S3 and 905 TEST rows for S4; no rows are excluded from the source stream, while padded copies in the final batch are excluded from the 905-row prediction count because they are not TEST samples |

## Configs

The recipe config is `models/ds-gan/configs/training/ds_gan_pku_posterlayout.yaml`.

| Config                         | Dataset          | Seed mode | Purpose                                                                                                  |
| ------------------------------ | ---------------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `ds_gan_pku_posterlayout.yaml` | PKU PosterLayout | `seed=0`  | Batch 128, 300 epochs, TEST batch 4, 16 workers, G-then-D Adam updates, and epoch-stepped `MultiStepLR`. |

## Scheduler and Recipe Notes

The vendor recipe uses a batch size of 128, 300 epochs, 32 layout elements, a ResNet-50 generator, a ResNet-18 discriminator, and a TEST batch size of 4. Adam rates are generator head/backbone `1e-4`/`1e-5` and discriminator head/backbone `1e-3`/`1e-4`. `MultiStepLR` uses gamma `0.8`, generator milestones `0,50,100,150,200,250`, and discriminator milestones `0,25,50,75,100,125,150,175,200,225,250,275`; schedulers step after each complete 78-batch epoch. The generator update precedes the discriminator update. The adversarial weight ramps linearly to one over the first 100 epochs. No EMA or AMP is active.

The setup-seed and random-initial-layout paths record Python, NumPy, CPU/CUDA Torch RNG states and use the shared `traingen_parity` capture/restore and tensor-hash helpers. Deterministic controls are `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Seed Policy

The staged evidence uses a fixed seed of 0. The natural repeated trajectory uses two independent 300-step runs with the same seed and real shuffled train stream; per-step loss, gradient, parameter, optimizer-state, scheduler, and RNG fields are report-only diagnostics unless the stage row names an asserted comparison. The synchronized diagnostic uses the same seed and re-synchronizes parameters, optimizer state, and RNG state at each optimizer boundary. S5 training-seed and evaluation-seed claims are not made.

## Validation Stages

| Stage | Scope                                             | Purpose                                                                                                                                                          |
| ----- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static configuration and initialized state parity | Compare independent package/vendor topology, counts, state keys, optimizer defaults, and initial-state hashes.                                                   |
| S1    | Fixed-batch pre-optimizer trace parity            | Compare the first real loader batch, explicit random layout, outputs, losses, and input encoding.                                                                |
| S2    | One optimizer-step parity                         | Compare G-then-D gradients, optimizer state, post-step parameters, and scheduler cadence.                                                                        |
| S3    | 300-step deterministic lockstep                   | Retain natural repeated traces and a separately synchronized per-step diagnostic with scheduler state.                                                           |
| S4    | Deterministic loader and evaluation stream        | Compare real train/TEST loader semantics, the hash-manifested bridge, vendor evaluation, package evaluation, predictions, counts, coordinate frame, and metrics. |
| S5    | Full-run statistical comparison                   | Out of scope; requires a separate user decision.                                                                                                                 |

## Stage Evidence

The run records contain the clean-head source commit, vendor commit, Python version, torch and torchvision wheel URLs and SHA-256 values, audited-venv path, freeze hash, deterministic controls, resolved recipe, and cache-relative artifacts.

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Artifact                                                                                                                                                                                                                                                                                                   | Result                                                                                  |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| S0    | `DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s0-static`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `.cache/ds-gan/stage-evidence/s0-static/run.json`                                                                                                                                                                                                                                                          | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |
| S1    | `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s1-forward-loss`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `.cache/ds-gan/stage-evidence/s1-forward-loss/run.json`                                                                                                                                                                                                                                                    | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |
| S2    | `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s2-optimizer-step`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `.cache/ds-gan/stage-evidence/s2-optimizer-step/run.json`                                                                                                                                                                                                                                                  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |
| S3    | `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep && CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep-synchronized`                                                                                                                                                                                                                                                    | `.cache/ds-gan/stage-evidence/s3-lockstep/run.json`, `.cache/ds-gan/stage-evidence/s3-lockstep/natural-repeat-1.jsonl`, `.cache/ds-gan/stage-evidence/s3-lockstep/natural-repeat-2.jsonl`, `.cache/ds-gan/stage-evidence/s3-lockstep-synchronized/run.json`                                                | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |
| S4    | `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" DSGAN_CHECKPOINT=.cache/ds-gan/original/DS-GAN-Epoch300.pth DSGAN_CONVERTED_DIR=.cache/ds-gan/converted/ds-gan-pku-posterlayout "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge && CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 DSGAN_AUDIT_VENV="$DSGAN_AUDIT_VENV" DSGAN_AUDIT_FREEZE_SHA256="$DSGAN_AUDIT_FREEZE_SHA256" DSGAN_TORCH_WHEEL_PATH="$DSGAN_TORCH_WHEEL_PATH" DSGAN_TORCHVISION_WHEEL_PATH="$DSGAN_TORCHVISION_WHEEL_PATH" DSGAN_CHECKPOINT=.cache/ds-gan/original/DS-GAN-Epoch300.pth DSGAN_CONVERTED_DIR=.cache/ds-gan/converted/ds-gan-pku-posterlayout "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-evaluation` | `.cache/ds-gan/bridge/pku_posterlayout_manifest.json`, `.cache/ds-gan/stage-evidence/s4-bridge/run.json`, `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`, `.cache/ds-gan/stage-evidence/s4-evaluation/vendor-predictions.npz`, `.cache/ds-gan/stage-evidence/s4-evaluation/package-predictions.npz` | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |
| S5    | `:`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `.cache/ds-gan/s5-not-run/manifest.json`                                                                                                                                                                                                                                                                   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` |

## Reproduction Results

S0-S4 evidence covers PKU PosterLayout with fixed-seed staged checks, two natural 300-step real-loader repeats, a synchronized diagnostic, and the 905 TEST samples. Exact claims are limited to fields whose run-time comparisons are bitwise equal; any measured difference is retained in the run record with its first differing field and values. The S5 full-run population is zero because it is outside this task, so no S5 training or evaluation claim is made.

| Dataset                           | System             | Status                                                                                  | Seed scope                    | Primary metrics                                               | Loss evidence                                           | Artifact summary                |
| --------------------------------- | ------------------ | --------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------- |
| PKU PosterLayout DS-GAN Epoch 300 | vendor and package | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` | fixed staged seed 0; S5 unrun | S4 metrics and counts are recorded in the evaluation artifact | S1-S3 comparison fields are recorded in stage artifacts | `.cache/ds-gan/stage-evidence/` |

### Comparison Scope

| Dataset                           | System | Evaluator                                                                                                                                | Test split | Checkpoint-selection rule                                                                   | Sample count |
| --------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------- | ------------ |
| PKU PosterLayout DS-GAN Epoch 300 | both   | `vendor/posterlayout-cvpr2023/infer.py:test` followed by `vendor/posterlayout-cvpr2023/eval.py:main` and `ds_gan.DSGANPipeline.__call__` | TEST       | Authors' released `DS-GAN-Epoch300.pth`, loaded into the vendor and converted package model | 905 layouts  |

## Regeneration Metadata

The bridge manifest records the approved source revision, source fields and mappings, missing-field status, train/TEST counts, transform rules, and source/bridge hashes. Stage records are generated only from a clean committed head and reference their own `source_commit`. The vendor checkout remains read-only at the recorded submodule commit.

## Training Commands

Run CPU-only checks and tests in the lockfile environment:

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package ds-gan --extra vendor --with pytest pytest models/ds-gan/tests -q
UV_FROZEN=1 uv run --package design-generators ty check lib models tools/devharness/src
```

Run the stage harness from the audited runtime after the worktree is committed and clean:

```bash
DSGAN_AUDIT_VENV=<your audit venv> DSGAN_HF_CACHE=<your Hugging Face datasets cache> DSGAN_AUDIT_FREEZE_SHA256=<freeze sha256> DSGAN_TORCH_WHEEL_PATH=<torch wheel> DSGAN_TORCHVISION_WHEEL_PATH=<torchvision wheel> "$DSGAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge
```

S5 is intentionally not included. No full-run training command, trained-checkpoint statistical claim, or S5-scale evaluation is supported by this document.
