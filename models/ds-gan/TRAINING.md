---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# DS-GAN Training

This document records the current S0–S4 reproduction boundary for DS-GAN (PosterLayout) on the approved [creative-graphic-design/PKU-PosterLayout source](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout). S0, S1, S2, the production-wiring layer, and S4 pass their stated checks. Natural S3 shows the expected drift from the shared CUDA nondeterministic reduction. Its synchronized diagnostic passes the declared trace and numeric-state envelope gates while RNG and scheduler state remain bitwise equal; the trace itself is not bitwise equal. No S5 trained-checkpoint reproduction claim is made; it is `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)`.

Run commands from the repository root. Generated data, logs, checkpoints, converted pipelines, and staged evidence stay under `.cache/ds-gan/` and are not committed.

## Install

The package declares its training dependencies in the `training` extra. The vendor parity dependencies are in the `vendor` extra. The repository lockfile is used frozen after the deliberate lock update that added the training extra.

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
UV_FROZEN=1 uv sync --package ds-gan --extra training --extra vendor --frozen
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package ds-gan --extra training --extra vendor pytest models/ds-gan/tests -q
```

The audited CUDA runtime was created with Python 3.11.15, Torch `2.8.0+cu128`, and torchvision `0.23.0+cu128`. Its measured freeze is `.cache/ds-gan/stage-evidence/runtime/pip-freeze.txt` with SHA-256 `28888e119167cef79dae5e06b124c9419cb5077a7be21ee1c446aa4ef3358437`. The evidence records the actual command as `UV_FROZEN=1 uv venv --python 3.11 .cache/ds-gan/runtime/audit-cu128`; the wheel direct URLs and distributions are recorded in every stage `run.json`.

Create and populate that audited runtime with the documented commands below. Replace only the angle-bracket paths with local paths; do not set an evidence freeze hash from an environment variable.

```bash
UV_FROZEN=1 uv venv --python 3.11 <DSGAN_AUDIT_VENV>
UV_FROZEN=1 uv export --frozen --package ds-gan --extra training --extra vendor --format requirements-txt --output-file .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python <DSGAN_AUDIT_VENV>/bin/python .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python <torch-2.8.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python <torchvision-0.23.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python <DSGAN_AUDIT_VENV>/bin/python --no-deps --editable models/ds-gan
UV_FROZEN=1 uv pip freeze --python <DSGAN_AUDIT_VENV>/bin/python > .cache/ds-gan/stage-evidence/runtime/pip-freeze.txt
CUDA_VISIBLE_DEVICES="" <DSGAN_AUDIT_VENV>/bin/python -m pytest models/ds-gan/tests -q
```

The runtime records measure the package set at execution time with `uv pip freeze --python <DSGAN_AUDIT_VENV>/bin/python` and record Torch and torchvision through package metadata and direct URLs. The released checkpoint is downloaded from the authors' [Google Drive weights folder](https://drive.google.com/drive/folders/1UYJ34BhqgYztfh5n5A4GU4nqgboPtoWS); its measured SHA-256 is `d1afdf0a4965229122f111a842a8afabaca94ef2501ad156a930193994b6bf15`. The two pinned backbone downloads are:

```bash
mkdir -p .cache/ds-gan/backbones
curl -L --fail --output .cache/ds-gan/backbones/resnet50_a1_0-14fe96d1.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
curl -L --fail --output .cache/ds-gan/backbones/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
echo 14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f .cache/ds-gan/backbones/resnet50_a1_0-14fe96d1.pth | sha256sum -c
echo 5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8 .cache/ds-gan/backbones/resnet18-5c106cde.pth | sha256sum -c
UV_FROZEN=1 uv run --package ds-gan --extra download models/ds-gan/scripts/download_original.py --output-dir .cache/ds-gan/original
echo d1afdf0a4965229122f111a842a8afabaca94ef2501ad156a930193994b6bf15 .cache/ds-gan/original/DS-GAN-Epoch300.pth | sha256sum -c
```

The ordered inference and conversion workflow is in [`REPRODUCING.md`](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/REPRODUCING.md).

## Data

The approved source is [creative-graphic-design/PKU-PosterLayout](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout), configuration `default`, revision `af3a6fdadeaf604fec8a735fd05043e44b93879d`, with 9,974 train rows and 905 TEST rows. The source fields are mapped by the bridge manifest `.cache/ds-gan/bridge/pku_posterlayout_manifest.json` to the vendor poster, canvas, saliency maps, and annotations. The source has no missing vendor-required fields.

The bridge transforms annotations from pixel `ltrb` to normalized center `xywh`, filters the `INVALID` class, merges the two saliency maps by pixelwise maximum, resizes training inputs to 350x240, and retains the TEST canvas at 513x750. S4 re-hashes the source Arrow/metadata inputs and all bridge PNG/annotation outputs; both hash checks pass in `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`.

## Configs

Training configuration is [`ds_gan_pku_posterlayout.yaml`](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/configs/training/ds_gan_pku_posterlayout.yaml). It is consumed by `traingen fit`, configures the CSV logger and ModelCheckpoint callback, and supplies the non-null ResNet-50 and ResNet-18 paths pinned above. The recipe uses batch size 128, 300 epochs, 32 layout elements, generator-before-discriminator Adam updates, and epoch-stepped `MultiStepLR` schedulers.

## Scheduler and Recipe Notes

The generator and discriminator learning-rate groups are measured from the package and vendor optimizer objects in S0: generator head/backbone `8e-5`/`8e-6`, discriminator head/backbone `8e-4`/`8e-5`, Adam betas `(0.9, 0.999)`, epsilon `1e-8`, and no weight decay or AMSGrad. The schedulers have gamma `0.8`, generator milestones `0,50,100,150,200,250`, and discriminator milestones `0,25,50,75,100,125,150,175,200,225,250,275`; the measured scheduler objects and order are equal. The production config uses the same scheduler objects through `traingen fit`, and its checkpoint records two scheduler states.

The production adversarial ramp is `min(1, (epoch - 1) / 100)`. `training_step` passes `self.current_epoch + 1` into the package step, and the epoch-two regression test proves the ramp is nonzero. `loss_reconstruction` contributes to the backpropagated generator objective. The real vendor loader consumes the global Torch RNG, while the production package data module uses a seeded `torch.Generator`; the seed-paired evidence records the pre-model and pre-loader RNG digests and the first loader sample IDs. The parity overlay uses the global-RNG stream so that the first 128 sample IDs match the vendor record.

The S3 natural/synchronized protocol amendment is [issue 425 comment 5970950954](https://github.com/creative-graphic-design/design-generators/issues/425#issuecomment-5970950954): the natural layer uses independent seed-zero streams without per-step RNG restoration or an injected shared layout, and the synchronized diagnostic copies the vendor parameters, optimizer state, scheduler state, and RNG at every optimizer boundary. Scheduler cadence, the natural loader deviation, and the evaluated data-path mapping are recorded here because the amendment changes how those checks are interpreted.

## Seed Policy

S0–S4 use fixed seed 0. The initial layout seed is derived from the global Torch RNG immediately after model construction, as recorded by the S4 RNG digests; it is not an independently asserted seed. Natural S3 has two independent 300-step vendor/package runs and separate vendor/package self-repeat runs. The synchronized layer has 300 optimizer-boundary steps. S5 has no training-seed or evaluation-seed claim.

## Validation Stages

| Stage | Scope                                               | Purpose                                                                                                                                              |
| ----- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static configuration and initialized state parity   | Compare topology, state keys, measured optimizer/scheduler objects, data encoding, loader RNG pairing, and pinned backbone initialization.           |
| S1    | Fixed-batch pre-optimizer trace parity              | Compare the real batch, derived layout, outputs, loss components, and total trace before optimizer mutation.                                         |
| S2    | One optimizer-step parity                           | Compare gradients, optimizer state, post-step parameters, and scheduler objects, with the CUDA reduction cause proven by like-for-like self repeats. |
| S3    | Natural, synchronized, and production-wiring layers | Measure natural drift, re-synchronize each optimizer boundary, and run the real `traingen fit` logger/checkpoint/scheduler path.                     |
| S4    | Deterministic loader and evaluation stream          | Re-hash the bridge inputs, preserve untouched vendor outputs for `eval.py:main`, and compare predictions and metrics.                                |
| S5    | Full-run statistical comparison                     | Not yet run for issue 425.                                                                                                                           |

## Stage Evidence

The audited commands use one selected CUDA device represented by `<gpu-index>` below. Evidence records carry the measured source commit, runtime distribution metadata, direct URLs, and freeze hash; the commands use cache-relative paths so they can be rerun on another host.

| Stage | Command                                                                                                                                                                                     | Artifact                                                                                                                                                                                | Result                                                                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s0-static`                     | `.cache/ds-gan/stage-evidence/s0-static/run.json`                                                                                                                                       | `PASS`                                                                                                                                                                              |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s1-forward-loss-re-evaluate`   | `.cache/ds-gan/stage-evidence/s1-forward-loss/run.json`                                                                                                                                 | `PASS` by re-evaluating the recorded trace under the shared distributional envelope rule; no S1 run was repeated                                                                    |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s2-optimizer-step-re-evaluate` | `.cache/ds-gan/stage-evidence/s2-optimizer-step/run.json`; 12 vendor and 12 package state files under `.cache/ds-gan/stage-evidence/s2-optimizer-step-attempts/`                        | `PASS` by re-evaluating the recorded envelope cause; no S2 run was repeated                                                                                                         |
| S3    | `<DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep`, then `s3-lockstep-synchronized`, then `s3-production-wiring`                      | `.cache/ds-gan/stage-evidence/s3-lockstep/run.json`, `.cache/ds-gan/stage-evidence/s3-lockstep-synchronized/run.json`, and `.cache/ds-gan/stage-evidence/s3-production-wiring/run.json` | Natural `FAIL` because cross final-state drift is outside both self envelopes; synchronized `PASS` with trace/numeric envelopes and bitwise RNG/scheduler; production wiring `PASS` |
| S4    | `<DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge`, then `s4-evaluation`                                                                | `.cache/ds-gan/stage-evidence/s4-bridge/run.json` and `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`                                                                             | `PASS`                                                                                                                                                                              |
| S5    | Not launched                                                                                                                                                                                | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)`; evaluation-path parity is `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`                | `not-yet-run`                                                                                                                                                                       |

## Reproduction Results

The covered dataset is PKU PosterLayout, with fixed staged seed 0 and 300-step S3 evidence. S0, S1, S2, the bounded production-wiring layer, and S4 pass. S3 natural training is not numerical parity because independent nondeterministic CUDA reductions produce cross-system final-state drift outside both measured self envelopes; the synchronized record isolates the same reduction and passes its aggregate trace/numeric envelope gates while preserving bitwise RNG and scheduler agreement. Neither system has an S5 status because full-run retraining was not launched.

| Dataset                           | System   | Status                                                                                  | Seed scope                                        | Primary metrics                                                                                                                                                                                                                                                                                                   | Loss evidence                                                                                                                                                       | Artifact summary                |
| --------------------------------- | -------- | --------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| PKU PosterLayout DS-GAN Epoch 300 | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` | fixed staged seed 0; two natural 300-step repeats | S4 `eval.py:main` output on 905 TEST rows: `metrics_val=0.8808247422680412`, `metrics_ove=0.021536158`, `metrics_ali=0.004672066506470047`, `metrics_und_l=0.8307596`, `metrics_und_s=0.434466984884646`, `metrics_uti=0.25395164914166374`, `metrics_occ=0.20991928398105858`, `metrics_rea=0.18753128338574718` | S2 vendor self/cross envelopes and the S3 natural envelope are in the cited JSON records                                                                            | `.cache/ds-gan/stage-evidence/` |
| PKU PosterLayout DS-GAN Epoch 300 | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` | fixed staged seed 0; two natural 300-step repeats | The same `eval.py:main` metrics and 905-row evaluation path; 4,832 valid predictions and 868 out-of-bounds predictions                                                                                                                                                                                            | S1 reconstruction loss is inside its declared nondeterministic-loss envelope; S2 has the measured CUDA reduction envelope; S3 numeric state equality is not claimed | `.cache/ds-gan/stage-evidence/` |

For S1 `loss_ce` and every S2 distributional comparison, the shared nondeterministic-op rule is: cross max absolute difference must be no larger than the larger of the vendor-self and package-self maxima, and cross median absolute difference must be no larger than the larger of the vendor-self and package-self medians. For S2's elementwise gradient and parameter gates, each cross element is compared with the larger corresponding vendor-self/package-self element envelope; any parameter element outside that larger envelope must be explained by the sign-straddling Adam first-step probe. This compares cross-system differences with the same distribution generated by the shared operation; it does not require cross values to be below both self summaries. S1 `loss_bbox` and `loss_giou` remain bitwise gates.

The recorded S1 artifact was re-evaluated without rerunning its systems. Its `loss_ce` cross max/median absolute differences are `4.76837158203125e-07`/`1.1920928955078125e-07`; vendor-self values are `8.344650268554688e-07`/`2.384185791015625e-07`; package-self values are `3.5762786865234375e-07`/`1.1920928955078125e-07`. The cross distribution is inside the larger self summaries, and `loss_bbox` and `loss_giou` are bitwise equal. The recorded S2 artifact was re-evaluated under the same rule without rerunning its optimizer step.

S2 uses path B because strict deterministic mode raised at the vendor operation `loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)` in `vendor/posterlayout-cvpr2023/RecLoss.py:119`, which lowers to CUDA `nll_loss2d`. The package uses `torch.nn.functional.cross_entropy` with the same implicit mean reduction. The warning is captured in `s2-optimizer-step/run.json`; production does not enable deterministic mode.

The S2 cross-system distributions have 12 paired comparisons: gradient max/median absolute differences `9.080395102500916e-09`/`8.498318493366241e-09`, optimizer-state `9.313225746154785e-10`/`9.022187441587448e-10`, post-step parameter `4.76837158203125e-07`/`4.76837158203125e-07`, and reconstruction loss `4.76837158203125e-07`/`0.0`. The first differing tensor is `generator.resnet_fpn.resnet_tilconv4.0.weight` in the gradient comparison, with max absolute difference `8.498318493366241e-09` and relative difference `0.006547757890075445`. The like-for-like elementwise gate reports zero parameter elements outside the combined self envelope. Its one sign-straddling Adam probe is `generator.resnet_fpn.resnet_tilconv4.1.bias[17]`; all 24 measured gradients contain both signs, the cross gradient maximum is `2.255546860396862e-10` versus the same self maximum, and all 12 Adam first-step predictions have zero residual. The configured Adam values for this element are `lr=8.000000000000001e-06`, `betas=(0.9, 0.999)`, and `eps=1e-08`; the positive/negative sign choices produce the measured cross-parameter deltas `±2.384185791015625e-07`. This is the recorded mechanism, not a post-hoc tolerance.

The natural S3 record contains two independent 300-step repeats in separate processes. Cross-system final-state differences have parameter max/median `0.0874451994895935`/`0.07912372425198555` and optimizer-state max/median `0.00891980528831482`/`0.008294710190966725`; vendor self maxima are `0.021291889250278473` and `0.007202007807791233`, while package self maxima are `0.09652704000473022` and `0.008627012372016907`. Both cross distributions are outside both self envelopes, so natural S3 remains `FAIL`. The synchronized 300-step record copies the vendor batch at all 300 steps because the independent loader streams differ. Its aggregate cross trace max/median is `9.5367431640625e-07`/`0.0`, gradient `1.9976869225502014e-07`/`2.0416337065398693e-08`, optimizer-state `2.0023435354232788e-08`/`2.321030478924513e-09`, and parameter `1.1082738637924194e-06`/`2.384185791015625e-07`; every aggregate cross distribution is inside both its vendor and package self envelopes. The trace is not bitwise, while RNG and scheduler state are bitwise equal. The CUDA warning owner remains the `nll_loss2d` operation.

S4 preserves each raw vendor prediction file before canonical comparison: `eval.py:main` is run on untouched vendor output and on package output, and only the in-memory comparison squeezes the vendor class singleton axis and maps normalized boxes to pixel `xyxy`. The two `.npz` prediction files are bitwise equal, both have 905 rows and SHA-256 `d127e029cbdc7c1f7583ddcefeeb63f5744bad158a2963dd810f9807454df197`, and both captured evaluator stdout files have SHA-256 `b2223f32c9fbf9f26b5d8811630367bf42c81d2c9d2c99a96a6b3b39b10958d1`. The vendor writes are isolated in `.cache/ds-gan/stage-evidence/s4-evaluation/vendor-overlay`; the bridge remains read-only and the stale `output` and `test_order.pt` paths are recorded by the bridge stage.

### Comparison Scope

| Dataset                           | System | Evaluator                                                                    | Test split | Checkpoint-selection rule                                                                                                                                               | Sample count |
| --------------------------------- | ------ | ---------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| PKU PosterLayout DS-GAN Epoch 300 | both   | `vendor/posterlayout-cvpr2023/eval.py:main` on each system's raw predictions | TEST       | Authors' released `DS-GAN-Epoch300.pth`, SHA-256 `d1afdf0a4965229122f111a842a8afabaca94ef2501ad156a930193994b6bf15`, loaded into the vendor and converted package model | 905 layouts  |

## Regeneration Metadata

The current evidence source commit is `3475eb2c0e6e9aa321a8e265a9187d0f6c7b4a89`; the vendor checkout is read-only at `44ee576471f3c06d06fc632f13e3f96d0c381847`. The approved source revision is `af3a6fdadeaf604fec8a735fd05043e44b93879d`. The measured audited runtime is Python 3.11.15, Torch `2.8.0+cu128`, torchvision `0.23.0+cu128`, and freeze SHA-256 `28888e119167cef79dae5e06b124c9419cb5077a7be21ee1c446aa4ef3358437`. Every stage `run.json` records these values, the direct wheel URLs, and the backbone URLs and SHA-256 values.

The current `sha256sum` of `.cache/ds-gan/bridge/pku_posterlayout_manifest.json` is `0c4b673b302e7d37c7ac6fe305fe1a8c4397e48926cfdcb931e5f639a2cb7b69`. The bridge stage records that file hash, source hashes, bridge hashes, the read-only bridge, and the vendor write overlay; a local audit must recompute the file hash rather than copy an embedded self-hash field.

## Training Commands

Run member tests and repository-local checks in the frozen lockfile environment:

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package ds-gan --extra training --extra vendor pytest models/ds-gan/tests -q
UV_FROZEN=1 uv run --package design-generators ty check lib models tools/devharness/src
UV_FROZEN=1 uv run --package devharness devharness check jaxtyping-annotations
```

Run the production configuration through its consumer:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package ds-gan --extra training traingen fit --config models/ds-gan/configs/training/ds_gan_pku_posterlayout.yaml --trainer.devices=1
```

Run the ordered evidence stages after the worktree is committed and clean:

```bash
export DSGAN_RESNET18_WEIGHTS=.cache/ds-gan/backbones/resnet18-5c106cde.pth
export DSGAN_RESNET50_WEIGHTS=.cache/ds-gan/backbones/resnet50_a1_0-14fe96d1.pth
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s1-forward-loss-re-evaluate
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s2-optimizer-step-re-evaluate
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep-synchronized
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-production-wiring
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 <DSGAN_AUDIT_VENV>/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-evaluation
```

S5 is intentionally not included. No full-run training command, trained-checkpoint statistical claim, or S5-scale evaluation is supported by this document.
