---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Layout-Corrector
---

# Layout-Corrector Training

This document records the package-local Layout-Corrector reproduction through S4 for RICO25 and PubLayNet. S5 full-run statistical reproduction is intentionally out of scope for [issue 426](https://github.com/creative-graphic-design/design-generators/issues/426) and is not claimed here.

Run commands from the repository root. Generated data, logs, checkpoints, and evaluation files stay under `.cache/layout-corrector/` or the separate local LayoutDM cache.

## Install

The lockfile environment is the source of truth for CPU-only repository checks and tests. It was synchronized before the evidence rerun and `uv.lock` remained unchanged.

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
git diff --exit-code -- uv.lock
```

GPU evidence uses the audited runtime because the locked `torch 2.13.0+cu130` build cannot initialize on the verified host driver. The audited wheel is `torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (torch `2.8.0+cu128`, CUDA tag `cu128`, wheel SHA-256 `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`), selected because the `2.11.0+cu128` wheel has no usable sm70 kernels on the assigned V100 host. The matching audited wheel is `torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (torchvision `0.23.0+cu128`, CUDA tag `cu128`, wheel SHA-256 `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`). The audited runtime uses Python `3.11.15`; its complete `uv pip freeze` output, including `file://` editable lines, has SHA-256 `915bb3b975274708f2f7c8d9aac6e506080a329337e13c7e8d1c09a24a6b5da8`.

```bash
LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv>
UV_FROZEN=1 uv venv --python 3.11 "$LAYOUT_CORRECTOR_AUDIT_VENV"
UV_FROZEN=1 uv pip install --python "$LAYOUT_CORRECTOR_AUDIT_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
UV_FROZEN=1 uv pip install --python "$LAYOUT_CORRECTOR_AUDIT_VENV/bin/python" -e lib/laygen -e models/layout-dm -e models/layout-corrector
UV_FROZEN=1 uv -q pip freeze --python "$LAYOUT_CORRECTOR_AUDIT_VENV/bin/python" | sha256sum
```

The package tests were also run inside the audited runtime before GPU evidence: `53 passed, 25 deselected, 3 warnings`. The lockfile environment remains the environment used for CPU-only repository checks and ordinary member tests; the audited-runtime package-test result is an additional runtime gate.

```bash
CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$LAYOUT_CORRECTOR_AUDIT_VENV/bin/pytest" models/layout-corrector/tests -m 'not vendor_parity' -q
```

## Data

The approved public sources are [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico), configuration `ui-screenshots-and-hierarchies-with-semantic-annotations`, and [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet). The parity runs reuse the LayoutDM processed representation and seed-0 frozen LayoutDM weights already present in the local LayoutDM training cache.

| Dataset   | Source                                                                                                 | Config or path                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| RICO25    | [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico)           | LayoutDM processed `rico25-max25`, `train`, `val`, and `test` splits; LayoutDM `layoutdm_rico/0/best_model.pt`         |
| PubLayNet | [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | LayoutDM processed `publaynet-max25`, `train`, `val`, and `test` splits; LayoutDM `layoutdm_publaynet/0/best_model.pt` |
| Crello    | [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello)                                 | Out of scope: an approved Layout-Corrector starter-kit format and matching local assets are not present.               |

The processed split hashes are RICO25 `train.pt` `7ae17f4f5ef5061932609e93fb6568c2b0d4bd553d27e72affa550e1686f37b9`, `val.pt` `6bd5a3d0333cfb4cd97cc3e526eb580ace1959fa7b766bdbe33d9f92b0a926f5`, and `test.pt` `2377d2893db7d0356fd174f943f894ea40e44166f4142827137b2c7d1f217ebc`; PubLayNet `train.pt` `efb09fd0571be3a4be07afeccdc07bc04f283f0f8b946ef742be3ef17e662d71`, `val.pt` `5725c2cae89549497c07d4ca8a96fc24a82bb397f146cc66bf3cb0388a648874`, and `test.pt` `392ca5119d1d8c21878f43fb9214cca705a9600e10dfac744a5624a9a48c8e73`. The clustering files are RICO25 `a5639d8dbaee174841820ab13954be87b32814c6ed3470bf11207fe38b219be4` and PubLayNet `ae82be807dd3ab68ed605c2d48fafa96ddc4e7037144ffb75d6dcb0702c56d3b`.

## Configs

Training configs live under `models/layout-corrector/configs/training`.

| Config                                         | Dataset   | Seed mode     | Purpose                                                                                                                            |
| ---------------------------------------------- | --------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `layoutcorrector_rico25.yaml`                  | RICO25    | default       | Package-local LightningCLI recipe with frozen LayoutDM reference, AdamW, epoch-level plateau scheduler, and 16 configured workers. |
| `layoutcorrector_publaynet.yaml`               | PubLayNet | default       | Package-local LightningCLI recipe with frozen LayoutDM reference, AdamW, epoch-level plateau scheduler, and 16 configured workers. |
| `layoutcorrector_rico25_deterministic.yaml`    | RICO25    | deterministic | Short CLI wiring and deterministic smoke configuration.                                                                            |
| `layoutcorrector_publaynet_deterministic.yaml` | PubLayNet | deterministic | Short CLI wiring and deterministic smoke configuration.                                                                            |

## Scheduler and Recipe Notes

The vendor entry point is `vendor/layout-corrector/bin/corrector_train.sh`, a Hydra recipe from the LayoutDM codebase that trains the corrector with a frozen LayoutDM generator. The package adapter uses the merged LayoutDM training components where they fit and exposes the equivalent LightningCLI surface. The effective recipe is AdamW with learning rate `5.0e-4`, betas `(0.9, 0.98)`, weight decay `0.1`, gradient clipping at norm `1.0`, batch size `64`, and `ReduceLROnPlateau` on `val_loss` with factor `0.5`, patience `2`, threshold `0.01`, and epoch cadence. Training uses the vendor's 16-worker loader; S4 evidence overrides both sides to zero workers only to make the comparison record single-process and deterministic, and the override is recorded in each loader artifact.

The frozen LayoutDM sampler uses the importance-sampled corruption branch and records its timestep and probability tensors before package mutation. RNG state is captured and restored before each paired comparison; the GPU lockstep uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms=False`, and `cudnn.deterministic=False` on both systems. AMP is disabled by the 32-bit recipe. EMA is not part of this loop, so the EMA rule is not applicable. The scheduler is applicable and remains at the validation epoch boundary; S2 records that it is not stepped before validation.

Registration order is part of the numerical training surface: `clip_grad_norm_` reduces gradients in parameter iteration order. The original corrector registered the backbone first while the package initially registered the embedding first, producing a reduction-order drift. The package now pins backbone-first registration, and the focused registration-order test compares the complete ordered parameter-name list.

## Seed Policy

S0-S3 use training seed `42975` per dataset, with the synchronized diagnostic restoring model, optimizer, and RNG state at every step. S4 loader evidence uses seed `314159`; its loader stream covers train, validation, and TEST with two batches per split. S5 has no training-seed or evaluation-seed evidence in this issue and remains unrun.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                                                                    |
| ----- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Confirms frozen LayoutDM checkpoint identity, corrector topology, parameter counts, state mapping, and initial tensors.                                    |
| S1    | Fixed-batch pre-optimizer trace parity     | Confirms importance-sampled corruption, reconstructed tokens, per-attribute BCE tensors, and total loss before optimizer mutation.                         |
| S2    | One optimizer-step parity                  | Confirms per-dataset gradient norms, clipped gradients, optimizer step, and post-step parameters.                                                          |
| S3    | Short deterministic multi-batch run        | Confirms natural 300-step lockstep, synchronized diagnostics, clipping, RNG, and scheduler wiring.                                                         |
| S4    | Deterministic loader stream                | Confirms train/validation/TEST loader streams and full TEST evaluation-path agreement.                                                                     |
| S5    | Full-run statistical comparison            | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; full training and S5 evaluation require a separate user decision. |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                         | Artifact                                                                                                                                                          | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `LAYOUT_DM_CACHE=<layout-dm-cache> CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s0_training_static_state_matches_vendor' -q`                                                                                                               | `.cache/layout-corrector/stage-evidence/s0-static/rico25/summary.json`, `.cache/layout-corrector/stage-evidence/s0-static/publaynet/summary.json`                 | PASS; frozen and corrector max absolute differences `0.0` for both datasets. The seed-0 LayoutDM weights are RICO25 SHA-256 `7759bdf9e05cccef7a6a7e4260adc50f8c1ef6e6faa10351b79fb63f6b51c853` and PubLayNet SHA-256 `9f7aee8ca600cc7cc96182affc85f96ebafc2b41a9ae72b05dfacfd64e89791d`; both systems load identical tensors.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| S1    | `LAYOUT_DM_CACHE=<layout-dm-cache> CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s1_fixed_batch_pre_optimizer_trace_matches_vendor' -q`                                                                                                     | `.cache/layout-corrector/stage-evidence/s1-fixed-batch/rico25/summary.json`, `.cache/layout-corrector/stage-evidence/s1-fixed-batch/publaynet/summary.json`       | PASS; every recorded prepared-input, corruption, reconstruction, per-attribute BCE, weighted BCE, and total-loss max absolute difference is `0.0`; first divergence is `null` for both datasets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| S2    | `LAYOUT_DM_CACHE=<layout-dm-cache> CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s2_one_optimizer_step_matches_vendor' -q`                                                                                                                  | `.cache/layout-corrector/stage-evidence/s2-optimizer-step/rico25/summary.json`, `.cache/layout-corrector/stage-evidence/s2-optimizer-step/publaynet/summary.json` | PASS; RICO25 vendor/package gradient norms are both `4.250988960266113`, PubLayNet vendor/package gradient norms are both `8.330658912658691`, and every post-step parameter max absolute difference is `0.0`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| S3    | `LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_S3_STEPS=300 CUDA_VISIBLE_DEVICES=<gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s3_natural_lockstep_matches_vendor or test_s3_synchronized_diagnostic_matches_vendor' -q` | `.cache/layout-corrector/stage-evidence/s3-lockstep/`, `.cache/layout-corrector/stage-evidence/s3-lockstep-synchronized/`                                         | PASS; natural RICO25 and PubLayNet runs each retain 300 steps with first divergence `null`, max relative loss difference `0.0`, and max parameter difference `0.0`. Synchronized RICO25 and PubLayNet runs each retain 300 steps with first trace/gradient/parameter divergence `null` and all three max differences `0.0`. The 16-step diagnostic had `7.450580596923828e-9` residual drift on the original parameter order for RICO25, while aligned `clip_grad_norm_(..., norm_type=2.0, error_if_nonfinite=False, foreach=None)` with vendor-name order produced `0.0` for both datasets; the cause is the original backbone-first versus package embedding-first registration order and its reduction order, not a tolerance selected after the fact. |
| S4    | `LAYOUT_DM_CACHE=<layout-dm-cache> CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s4_loader_stream_matches_vendor or test_s4_test_evaluation_path_matches_vendor' -q`                                                                        | `.cache/layout-corrector/stage-evidence/loader-stream/`, `.cache/layout-corrector/stage-evidence/evaluation-path/`                                                | PASS; loader evidence compares two batches for train, validation, and TEST with configured 16 workers and evidence override 0. The separate full TEST evaluation-path artifact compares every TEST layout with identical weights, inputs, evaluator settings, prediction files, counts, metrics, and original-frame bounds.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| S5    | Not run by this task.                                                                                                                                                                                                                                                                                                                                                                                                           | [issue 426](https://github.com/creative-graphic-design/design-generators/issues/426)                                                                              | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; no S5-scale GPU training or full-run claim is made.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

The stage records were generated from package source commit `18d78714cafe1f60c34ffe7d3498af08256fc104`, LayoutDM source commit `873b5eebe4c61862e5c08a10859accf65a168dfd`, and Layout-Corrector vendor/evaluator commit `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8`; each summary JSON records these commits independently.

### Full TEST Evaluation-Path Artifact

The loader-stream artifact is intentionally only a two-batch stream check. The evaluation-path artifact runs the full TEST split through `vendor/layout-corrector/bin/corrector_test_eval.py` and `trainer.corrector_test.run`, then sends the same preprocessed input IDs through the package evaluation path. Both systems use condition `gt`, timestep `10`, the vendor evaluator's default batch size `512`, no random ordering, and no Gumbel noise, and report normalized center `xywh` in `[0, 1]`.

| Dataset   | TEST layouts | Vendor/package predictions | Vendor/package out-of-bounds in original frame |                                               Vendor/package metric | Prediction max abs diff | Evaluator commit                           |
| --------- | -----------: | -------------------------: | ---------------------------------------------: | ------------------------------------------------------------------: | ----------------------: | ------------------------------------------ |
| RICO25    |        4,218 |            47,129 / 47,129 |                                  4,571 / 4,571 |   11.1733048838312 / 11.1733048838312 valid elements per prediction |                   `0.0` | `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8` |
| PubLayNet |       11,142 |          119,402 / 119,402 |                                          1 / 1 | 10.71638844013642 / 10.71638844013642 valid elements per prediction |                   `0.0` | `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8` |

RICO25 records the full-input SHA-256 `047a11008553a748d839ffd28d9f4037902a5048978d87993c61ec4d808db868`, identical vendor/package weight SHA-256 `d2936af3875bc9b7b7483cfa0ad86dfceaa2dfcdc1070fcdc8b283124dc4eb33`, and identical prediction-file SHA-256 `8b25c2738656f898aee0425683a119abb2d6f165619727996ecb1142632ca851`. PubLayNet records input SHA-256 `56965504467f7db77baa464c04e6ed94d153858597cbb27c67a892d0152c301c`, identical vendor/package weight SHA-256 `c642d00ff8db4443a88abf748234fd75733c8a392ca1fae8fdba845b8b65f2a8`, and identical prediction-file SHA-256 `eb3f204c2b60fae9f0825882f75922e4c37f80f8dca405c4ec3990ef2543ec9b`. The diagnostic confidence-score max differences are approximately `9.834766387939453e-7` for RICO25 and `1.0728836059570312e-6` for PubLayNet; prediction and metric equality are the claimed exact results.

## Reproduction Results

S0-S4 reproduction is exact for RICO25 and PubLayNet at the recorded static, fixed-batch, one-step, 300-step, loader, and full TEST evaluation comparison points. This is not an S5 trained-checkpoint reproduction: no full training run, seed queue, or S5 statistical claim was executed. Crello is not claimed because its approved starter-kit format and local assets are unavailable.

| Dataset   | System   | Status                                                                                  | Seed scope         | Primary metrics                    | Loss evidence                         | Artifact summary                                                                     |
| --------- | -------- | --------------------------------------------------------------------------------------- | ------------------ | ---------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------ |
| RICO25    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | S5 not run         | S4 TEST metric `11.1733048838312`  | S1 exact; S2 norm `4.250988960266113` | `.cache/layout-corrector/stage-evidence/`                                            |
| RICO25    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | S5 not run         | S4 TEST metric `11.1733048838312`  | S1 exact; S2 norm `4.250988960266113` | `.cache/layout-corrector/stage-evidence/`                                            |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | S5 not run         | S4 TEST metric `10.71638844013642` | S1 exact; S2 norm `8.330658912658691` | `.cache/layout-corrector/stage-evidence/`                                            |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | S5 not run         | S4 TEST metric `10.71638844013642` | S1 exact; S2 norm `8.330658912658691` | `.cache/layout-corrector/stage-evidence/`                                            |
| Crello    | original | `blocked (approved starter-kit format and local assets unavailable)`                    | Not in S0-S5 scope | Not measured                       | Not measured                          | [issue 426](https://github.com/creative-graphic-design/design-generators/issues/426) |
| Crello    | package  | `blocked (approved starter-kit format and local assets unavailable)`                    | Not in S0-S5 scope | Not measured                       | Not measured                          | [issue 426](https://github.com/creative-graphic-design/design-generators/issues/426) |

The exact claim covers the named inputs, frozen weights, evaluator settings, and comparison criteria in the stage artifacts. Confidence scores are reported as approximate diagnostics because the package and original reduction paths differ by floating-point operation order; prediction files, counts, original-frame bounds, and reported TEST metrics are byte- or value-identical.

### Comparison Scope

| Dataset   | System | Evaluator                                                                         | Test split     | Checkpoint-selection rule                                                               |             Sample count |
| --------- | ------ | --------------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------- | -----------------------: |
| RICO25    | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` plus package evaluation path | TEST           | Issue-approved frozen seed-0 LayoutDM corrector state, with identical corrector tensors |            4,218 layouts |
| PubLayNet | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` plus package evaluation path | TEST           | Issue-approved frozen seed-0 LayoutDM corrector state, with identical corrector tensors |           11,142 layouts |
| Crello    | both   | Not run; approved starter-kit format unavailable                                  | Not applicable | No checkpoint selected                                                                  | 0 layouts (out of scope) |

## Regeneration Metadata

The stage summaries are the run records: each records the package, LayoutDM, and vendor source commits, first-divergence fields, comparison maxima, and artifact paths. The S0 records additionally pin the frozen LayoutDM checkpoint SHA-256 and report zero tensor difference after both systems load it.

```text
.cache/layout-corrector/stage-evidence/
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/test-input-ids.bin
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/{vendor,package}-weights.bin
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/{vendor,package}-predictions.json
```

## Training Commands

Initialize the vendor checkout before gated parity checks.

```bash
git submodule update --init vendor/layout-corrector vendor/layout-dm
```

Run the package-local LightningCLI recipe after S0-S4 evidence and a separate S5 decision.

```bash
CUDA_VISIBLE_DEVICES=<gpu> UV_FROZEN=1 uv run --package layout-corrector --extra training traingen fit --config models/layout-corrector/configs/training/layoutcorrector_<rico25|publaynet>.yaml
```

Run the exact S0-S2 evidence command.

```bash
LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s0_training_static_state_matches_vendor or test_s1_fixed_batch_pre_optimizer_trace_matches_vendor or test_s2_one_optimizer_step_matches_vendor' -q
```

Run the exact 300-step natural and synchronized GPU evidence command.

```bash
LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_CORRECTOR_S3_STEPS=300 CUDA_VISIBLE_DEVICES=<gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s3_natural_lockstep_matches_vendor or test_s3_synchronized_diagnostic_matches_vendor' -q
```

Run the exact loader and full TEST evaluation-path command.

```bash
LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_s4_loader_stream_matches_vendor or test_s4_test_evaluation_path_matches_vendor' -q
```

S5 full training and evaluation commands are deliberately omitted because this task does not authorize an S5 run.
