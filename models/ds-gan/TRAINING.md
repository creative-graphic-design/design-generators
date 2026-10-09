---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
---

# DS-GAN Training

This document records the current S0–S4 reproduction boundary for DS-GAN (PosterLayout) on the approved [creative-graphic-design/PKU-PosterLayout source](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout). S0, S1, S2, the production-wiring layer, and S4 pass their stated checks. Natural S3 has identical batch streams at all 300 steps but is bitwise `FAIL`: parameter drift is inside the larger self envelope, while optimizer-state drift is outside both self envelopes. Its synchronized diagnostic has zero pre-synchronization loader mismatches and zero batch copies after harness-only epoch-boundary generator alignment; its trace is not bitwise, but the declared trace and numeric-state envelopes and bitwise RNG and scheduler checks pass. No S5 trained-checkpoint reproduction claim is made; it is `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)`.

Run commands from the repository root. Generated data, logs, checkpoints, converted pipelines, and staged evidence stay under `.cache/ds-gan/` and are not committed.

## Install

The package declares its training dependencies in the `training` extra. The vendor parity dependencies are in the `vendor` extra. The repository lockfile is used frozen after the deliberate lock update that added the training extra.

```bash
UV_FROZEN=1 uv sync --all-packages --frozen
UV_FROZEN=1 uv sync --package ds-gan --extra training --extra vendor --frozen
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package ds-gan --extra training --extra vendor pytest models/ds-gan/tests -q
```

The audited CUDA runtime was created with Python 3.11.15, Torch `2.8.0+cu128`, and torchvision `0.23.0+cu128`. Its measured freeze is `.cache/ds-gan/stage-evidence/runtime/pip-freeze.txt` with SHA-256 `02a1f7cfdb058ce8bfd5134c4860ea98d6385c0abf9c6e1a38804f7ee6c3346b`. The evidence records synthesized venv-command provenance; the creating shell command was not captured. The wheel direct URLs and distributions are recorded in the stage `run.json` files.

Create and populate that audited runtime with the documented commands below. Do not install `pip` into the audited venv, and do not set an evidence freeze hash from an environment variable.

```bash
export DS_GAN_AUDIT_VENV=.cache/ds-gan/runtime/audit-cu128
UV_FROZEN=1 uv venv --python 3.11 "$DS_GAN_AUDIT_VENV"
UV_FROZEN=1 uv export --frozen --package ds-gan --extra training --extra vendor --format requirements-txt --output-file .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python "$DS_GAN_AUDIT_VENV/bin/python" .cache/ds-gan/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python "$DS_GAN_AUDIT_VENV/bin/python" <torch-2.8.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python "$DS_GAN_AUDIT_VENV/bin/python" <torchvision-0.23.0+cu128-wheel>
UV_FROZEN=1 uv pip install --python "$DS_GAN_AUDIT_VENV/bin/python" --no-deps --editable models/ds-gan
UV_FROZEN=1 uv pip freeze --python "$DS_GAN_AUDIT_VENV/bin/python" > .cache/ds-gan/stage-evidence/runtime/pip-freeze.txt
CUDA_VISIBLE_DEVICES="" "$DS_GAN_AUDIT_VENV/bin/python" -m pytest models/ds-gan/tests -q
```

The runtime records measure the package set at execution time with `uv pip freeze --python "$DS_GAN_AUDIT_VENV/bin/python"` and record Torch and torchvision through package metadata and direct URLs. The released checkpoint is downloaded from the authors' [Google Drive weights folder](https://drive.google.com/drive/folders/1UYJ34BhqgYztfh5n5A4GU4nqgboPtoWS); its measured SHA-256 is `d1afdf0a4965229122f111a842a8afabaca94ef2501ad156a930193994b6bf15`. The two pinned backbone downloads are:

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

The production adversarial ramp is `min(1, (epoch - 1) / 100)`. `training_step` passes `self.current_epoch + 1` into the package step, and the epoch-two regression test proves the ramp is nonzero. `loss_reconstruction` contributes to the backpropagated generator objective. The real vendor loader consumes the global Torch RNG, while the production package data module uses a seeded `torch.Generator`; the seed-paired evidence records the pre-model and pre-loader RNG digests and the first loader sample IDs. Natural S3 records identical batch digests at all 300 steps. The synchronized harness aligns the package loader generator with the vendor global-RNG state at each epoch boundary; its 300 batch digests match before synchronization, so the batch-copy policy triggers zero times. This alignment is harness-only and is not a production DataModule change.

The S3 natural/synchronized protocol amendment is [issue 425 comment 5970950954](https://github.com/creative-graphic-design/design-generators/issues/425#issuecomment-5970950954): the natural layer uses independent seed-zero streams without per-step RNG restoration or an injected shared layout, and the synchronized diagnostic copies the vendor parameters, optimizer state, scheduler state, and RNG at every optimizer boundary. Scheduler cadence, the natural loader deviation, and the evaluated data-path mapping are recorded here because the amendment changes how those checks are interpreted.

## Seed Policy

S0–S4 use fixed seed 0. The initial layout seed is derived from the global Torch RNG immediately after model construction, as recorded by the S4 RNG digests; it is not an independently asserted seed. Natural S3 has three independent 300-step vendor/package runs in separate processes. The synchronized layer has 300 optimizer-boundary steps. S5 has no training-seed or evaluation-seed claim.

## Validation Stages

| Stage | Scope                                               | Purpose                                                                                                                                                                                                                                    |
| ----- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | Static configuration and initialized state parity   | Run the named fail-closed `test_s0_ds_gan_topology_guard` for parameter counts, state-dict maps with extra-key rejection, parameters and buffers, same-seed copied-weight forwards, optimizer/EMA/sampler, schedulers, and dataset checks. |
| S1    | Fixed-batch pre-optimizer trace parity              | Compare the real batch, derived layout, outputs, loss components, and total trace before optimizer mutation.                                                                                                                               |
| S2    | One optimizer-step parity                           | Compare gradients, optimizer state, post-step parameters, and scheduler objects, with the CUDA reduction cause proven by like-for-like self repeats.                                                                                       |
| S3    | Natural, synchronized, and production-wiring layers | Measure natural drift, re-synchronize each optimizer boundary, and run the real `traingen fit` logger/checkpoint/scheduler path.                                                                                                           |
| S4    | Deterministic loader and evaluation stream          | Re-hash the bridge inputs, compare the full TRAIN stream including order, transforms, labels, boxes, masks, and padding, compare TEST, then truncate padded inference rows before `eval.py:main` and compare predictions and metrics.      |
| S5    | Full-run statistical comparison                     | Not yet run for issue 425.                                                                                                                                                                                                                 |

## Stage Evidence

The audited commands use one selected CUDA device represented by `<gpu-index>` below. Evidence records carry the measured source commit, runtime distribution metadata, direct URLs, and freeze hash; the commands use cache-relative paths so they can be rerun on another host.

| Stage | Command                                                                                                                                                                                     | Artifact                                                                                                                                                                                | Result                                                                                                                                                                                                                                                                                               |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 $DS_GAN_AUDIT_VENV/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s0-static`                     | `.cache/ds-gan/stage-evidence/s0-static/run.json`                                                                                                                                       | `PASS` at source commit `ca44c8ce65f690d5d22f422fe3511d2b20bf5866`, artifact SHA-256 `1a3cfc21d64e51d2b1144337b4e528b466be6c33be7509245b9aecdc7ab51aa4`; the named test and all S0 gate fields pass.                                                                                                 |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 $DS_GAN_AUDIT_VENV/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s1-forward-loss-re-evaluate`   | `.cache/ds-gan/stage-evidence/s1-forward-loss/run.json`                                                                                                                                 | `STAND`: recorded `PASS` at `d0b7fcd72d4deb151419c4ba9f49d598851da509`; `models/ds-gan/src` and `lib` diff is empty through the evidence head                                                                                                                                                        |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 $DS_GAN_AUDIT_VENV/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s2-optimizer-step-re-evaluate` | `.cache/ds-gan/stage-evidence/s2-optimizer-step/run.json`; 12 vendor and 12 package state files under `.cache/ds-gan/stage-evidence/s2-optimizer-step-attempts/`                        | `PASS` by re-evaluating the recorded repeats at the harness evidence head; no optimizer-step rerun                                                                                                                                                                                                   |
| S3    | `$DS_GAN_AUDIT_VENV/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep`, then `s3-lockstep-synchronized`, then `s3-production-wiring`                      | `.cache/ds-gan/stage-evidence/s3-lockstep/run.json`, `.cache/ds-gan/stage-evidence/s3-lockstep-synchronized/run.json`, and `.cache/ds-gan/stage-evidence/s3-production-wiring/run.json` | Natural `FAIL`: parameter cross max `0.07886166870594025` is inside the larger self envelope (`0.08465087413787842`), optimizer-state cross max `0.009529690258204937` is outside both self envelopes (`0.007647970225661993` and `0.007998878136277199`); synchronized and production wiring `PASS` |
| S4    | `$DS_GAN_AUDIT_VENV/bin/python models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge`, then `s4-evaluation`                                                                | `.cache/ds-gan/stage-evidence/s4-bridge/run.json` and `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`                                                                             | `PASS`: bridge artifact SHA-256 `126e25007d413f168cbfe7b25fef0f7dd87c61387e7464768d262dcc656af865`; evaluation artifact SHA-256 `6cd8fc7d94a52109c1b47b7742f5912743ab47034142f3f4c5135f3b46df6b65`; full TRAIN and TEST stream gates pass.                                                           |
| S5    | Not launched                                                                                                                                                                                | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)`; evaluation-path parity is `.cache/ds-gan/stage-evidence/s4-evaluation/run.json`                | `not-yet-run`                                                                                                                                                                                                                                                                                        |

## Reproduction Results

The covered dataset is PKU PosterLayout, with fixed staged seed 0 and 300-step S3 evidence. S0, S1, S2, the bounded production-wiring layer, and S4 pass. S3 natural training is bitwise `FAIL` despite identical batch streams at all 300 steps: parameter drift is inside the larger self envelope, while optimizer-state drift is outside both self envelopes under the shared nondeterministic CUDA reduction. The synchronized record uses harness-only epoch-boundary generator alignment, observes zero pre-synchronization loader mismatches and zero vendor-batch copies, and passes its aggregate trace/numeric envelope gates while preserving bitwise RNG and scheduler agreement. Neither system has an S5 status because full-run retraining was not launched.

| Dataset                           | System   | Status                                                                                  | Seed scope                                          | Primary metrics                                                                                                                                                                                                                                                                                                     | Loss evidence                                                                                                                                                       | Artifact summary                |
| --------------------------------- | -------- | --------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| PKU PosterLayout DS-GAN Epoch 300 | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` | fixed staged seed 0; three natural 300-step repeats | S4 `eval.py:main` output on 905 TEST rows: `metrics_val=0.8803807947019867`, `metrics_ove=0.02160755`, `metrics_ali=0.0046809901056436036`, `metrics_und_l=0.8307172`, `metrics_und_s=0.43483033932135734`, `metrics_uti=0.25395164914166374`, `metrics_occ=0.20991928398105858`, `metrics_rea=0.18753128338574718` | S2 vendor self/cross envelopes and the S3 natural envelope are in the cited JSON records                                                                            | `.cache/ds-gan/stage-evidence/` |
| PKU PosterLayout DS-GAN Epoch 300 | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/425)` | fixed staged seed 0; three natural 300-step repeats | The same `eval.py:main` metrics and 905-row evaluation path; 4,832 valid predictions and 868 out-of-bounds predictions                                                                                                                                                                                              | S1 reconstruction loss is inside its declared nondeterministic-loss envelope; S2 has the measured CUDA reduction envelope; S3 numeric state equality is not claimed | `.cache/ds-gan/stage-evidence/` |

For S1 `loss_ce` and every S2 distributional comparison, the shared nondeterministic-op rule is: cross max absolute difference must be no larger than the larger of the vendor-self and package-self maxima, and cross median absolute difference must be no larger than the larger of the vendor-self and package-self medians. For S2's elementwise gradient and parameter gates, each cross element is compared with the larger corresponding vendor-self/package-self element envelope; any parameter element outside that larger envelope must be explained by the sign-straddling Adam first-step probe. This compares cross-system differences with the same distribution generated by the shared operation; it does not require cross values to be below both self summaries. S1 `loss_bbox` and `loss_giou` remain bitwise gates.

The recorded S1 artifact was re-evaluated without rerunning its systems. Its `loss_ce` cross max/median absolute differences are `4.76837158203125e-07`/`1.1920928955078125e-07`; vendor-self values are `7.152557373046875e-07`/`2.384185791015625e-07`; package-self values are `3.5762786865234375e-07`/`1.1920928955078125e-07`. The cross distribution is inside the larger self summaries, and `loss_bbox` and `loss_giou` are bitwise equal. The recorded S2 artifact was re-evaluated under the same rule without rerunning its optimizer step.

S2 uses path B because strict deterministic mode raised at the vendor operation `loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)` in `vendor/posterlayout-cvpr2023/RecLoss.py:119`, which lowers to CUDA `nll_loss2d`. The package uses `torch.nn.functional.cross_entropy` with the same implicit mean reduction. The warning is captured in `s2-optimizer-step/run.json`; production does not enable deterministic mode.

The S2 cross-system distributions have 12 paired comparisons: gradient max/median absolute differences `1.4901161193847656e-08`/`8.789356797933578e-09`, optimizer-state `1.3969838619232178e-09`/`9.022187441587448e-10`, post-step parameter `4.76837158203125e-07`/`4.76837158203125e-07`, and reconstruction loss `4.76837158203125e-07`/`4.76837158203125e-07`. One cross pair first differs at `generator.resnet_fpn.resnet_tilconv4.0.weight`, with max absolute difference `6.111804395914078e-09` and relative difference `0.007666847202926874`. The like-for-like elementwise gate reports zero gradient and parameter elements outside the combined self envelopes. Its maximum parameter probe is `generator.resnet_fpn.resnet_tilconv4.1.bias[57]`; all 24 measured gradients are positive, so no sign straddle occurred, and all 12 Adam first-step predictions have zero residual. The probe's cross gradient and self maxima are both `4.6202330850064754e-10`; the configured Adam values are `lr=8.000000000000001e-06`, `betas=(0.9, 0.999)`, and `eps=1e-08`. Because no parameter element was outside the combined self envelope, no sign-straddling exception was needed. This is the recorded mechanism, not a post-hoc tolerance.

The natural S3 record contains three independent 300-step repeats per system in separate processes. Cross-system final-state max absolute differences are parameter `0.07886166870594025` and optimizer state `0.009529690258204937`; vendor self maxima are `0.06539378315210342` and `0.007647970225661993`, while package self maxima are `0.08465087413787842` and `0.007998878136277199`. Parameter drift is inside the larger self envelope; optimizer-state drift is outside both self envelopes, so natural S3 remains bitwise `FAIL`. The synchronized 300-step record has identical vendor/package batch digests at all 300 steps, zero pre-synchronization loader mismatches, and zero vendor-batch copies after harness-only epoch-boundary generator alignment. Its aggregate cross trace max/median is `9.5367431640625e-07`/`0.0`, gradient `2.1187588572502136e-07`/`1.802982296794653e-08`, optimizer-state `2.1187588572502136e-08`/`1.9208528101444244e-09`, and parameter `9.5367431640625e-07`/`1.1920928955078125e-07`; every aggregate cross distribution is inside both its vendor and package self envelopes. The trace is not bitwise, while RNG and scheduler state are bitwise equal. The CUDA warning owner remains the `nll_loss2d` operation.

S4 compares the full deterministic TRAIN population of 9,974 rows in 78 batches. Every batch has equal order and equal `pixel_values`, `layout`, `labels`, `boxes`, and `mask` fields, with zero failed field comparisons. TEST has 905 rows in 227 batches, equal order, and three expected padded rows. The source has no validation split. For evaluation, S4 preserves each raw vendor prediction file before canonical comparison and truncates the final padded inference batch from 908 raw rows to 905 TEST rows before `eval.py:main`; class-axis squeezing and pixel-scale conversion are only in-memory canonical comparisons. The two `.npz` prediction files are bitwise equal, both have 905 rows and SHA-256 `d127e029cbdc7c1f7583ddcefeeb63f5744bad158a2963dd810f9807454df197`, and both captured evaluator stdout files have SHA-256 `449538f81d20bfd177e8f79811f05607b46d188e8752b2fe2765351b150c4ca1`. The vendor writes are isolated in `.cache/ds-gan/stage-evidence/s4-evaluation/vendor-overlay`; the bridge remains read-only and the stale `output` and `test_order.pt` paths are recorded by the bridge stage.

### Comparison Scope

| Dataset                           | System | Evaluator                                                                    | Test split | Checkpoint-selection rule                                                                                                                                               | Sample count |
| --------------------------------- | ------ | ---------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| PKU PosterLayout DS-GAN Epoch 300 | both   | `vendor/posterlayout-cvpr2023/eval.py:main` on each system's raw predictions | TEST       | Authors' released `DS-GAN-Epoch300.pth`, SHA-256 `d1afdf0a4965229122f111a842a8afabaca94ef2501ad156a930193994b6bf15`, loaded into the vendor and converted package model | 905 layouts  |

## Regeneration Metadata

The S0 record uses source commit `ca44c8ce65f690d5d22f422fe3511d2b20bf5866`; S1 and S2 standing records use source commit `d0b7fcd72d4deb151419c4ba9f49d598851da509`; the natural S3 record uses `ba7a0a17237e959afff319dd777c32f089398574`; the synchronized S3 and production-wiring S3 records use `083e78b92b7170f7e3b15b456b42468ad77eef47`; the bridge and evaluation S4 records use `5eba9c006faa259a1b7c8ca0f878a7ac119d5e7a`. The harness-only S0 and S4 changes have an empty `models/ds-gan/src` and `lib` diff from the standing S1/S2 source. The vendor checkout is read-only at `44ee576471f3c06d06fc632f13e3f96d0c381847`. The approved source revision is `af3a6fdadeaf604fec8a735fd05043e44b93879d`. The final audited runtime measured in the fresh S3–S4 records is Python 3.11.15, Torch `2.8.0+cu128`, torchvision `0.23.0+cu128`, and freeze SHA-256 `02a1f7cfdb058ce8bfd5134c4860ea98d6385c0abf9c6e1a38804f7ee6c3346b`. The stage `run.json` files record the direct wheel URLs and backbone URLs and SHA-256 values.

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
export DS_GAN_AUDIT_VENV=.cache/ds-gan/runtime/audit-cu128
export DSGAN_RESNET18_WEIGHTS=.cache/ds-gan/backbones/resnet18-5c106cde.pth
export DSGAN_RESNET50_WEIGHTS=.cache/ds-gan/backbones/resnet50_a1_0-14fe96d1.pth
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s1-forward-loss-re-evaluate
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s2-optimizer-step-re-evaluate
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-lockstep-synchronized
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s3-production-wiring
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-bridge
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 "$DS_GAN_AUDIT_VENV/bin/python" models/ds-gan/tests/vendor_parity/training_stage_evidence.py s4-evaluation
```

S5 is intentionally not included. No full-run training command, trained-checkpoint statistical claim, or S5-scale evaluation is supported by this document.
