---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - RADM
---

# RADM Training

This document records the phase-1 package training surfaces and staged
reproduction status for RADM. The supported Python 3.11/V100 rerun accepts S0,
S1, and S2 on the source-generated fixed batch, with S3 blocked at its first
step-2 gradient divergence. No released checkpoint or approved training data
assets are present in this worktree. The current candidate includes the
narrowly proven GRAM operation correction and the source-order ROI view
correction; the ladder below is fresh post-correction evidence. S4 and S5
evidence is not claimed.

Run commands from the repository root. Keep generated data, logs, checkpoints,
converted local pipelines, and evaluation artifacts under `.cache/radm/`.

## Install

```bash
uv sync --package radm --extra training
```

The checked optional `vendor` extra uses the repository lock's Detectron2 v0.6,
fvcore, and iopath pins. It is required only for the original-code reference
adapter and has been validated on CPU in this worktree. Supported-runtime
diagnostic evidence and disposable environments remain outside the repository
under `.cache/radm/`.

```bash
uv sync --package radm --extra training --extra vendor
```

The vendor extra is not required for package-local static checks. This phase
does not download data or checkpoints and does not launch training or
evaluation jobs.

## Data

The original recipe consumes CGL and CGL-v2 image annotations plus precomputed
768-dimensional text features. Neither dataset nor feature assets are included
here. The package adapter accepts explicit local paths and never downloads
data.

The approved S4 gate requires, at minimum, the authorized CGL train/test
annotations, the cleaned CGL training images produced through the original
LaMa preprocessing requirement, the matching train/test 768-D text-feature
trees, and a recorded license/data-provenance decision. These assets must be
available at the selected local paths before S4; the source-generated S1-S3
fixtures are not substitutes for them.

| Dataset | Source | Config or path |
| --- | --- | --- |
| CGL | `creative-graphic-design/CGL-Dataset` when approved by the project data policy | `.cache/radm/data/cgl/` |
| CGL-v2 | source and license still under review | `.cache/radm/data/cgl-v2/` |

The original source's data provenance and the CGL-v2 license remain unresolved.

## Configs

Training configs live under `models/radm/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `effective_radm_config.yaml` | CGL and CGL-v2 | captured static state | Mechanical record of effective model, optimizer, scheduler, sampler, and input values. |
| `radm_cgl.yaml` | CGL | default | Future one-GPU package training launch; not run in phase 1. |
| `radm_cgl_v2.yaml` | CGL-v2 | default | Future one-GPU package training launch; not run in phase 1. |
| `radm_s0_deterministic.yaml` | CGL | deterministic | S0/initialization wiring; data and original-code state are still unavailable. |
| `radm_smoke.yaml` | CGL fixture | deterministic | Package-local CPU wiring only; not reproduction evidence. |

## Scheduler and Recipe Notes

The effective original recipe uses one GPU, batch size 16, AdamW with learning
rate `2.5e-5`, weight decay `1e-4`, default AdamW betas and epsilon, full-model
gradient clipping at `1.0`, warmup factor `0.01` for 1,000 optimizer steps,
step-cadence milestones at 150,000 and 220,000, and `MAX_ITER=250000`. The
diffusion schedule has 1,000 training steps, `SNR_SCALE=2.0`, and
`SAMPLE_STEP=1`; the model uses a ResNet-50 FPN, four ROI levels, ROIAlignV2
resolution 7 with sampling ratio 2, 100 proposals, six repeated heads, VTRAM,
GRAM, and dynamic convolution with two projections of width 64.

The effective class state is intentionally asymmetric: the model predicts four
classes while the CGL vocabulary contains five labels. The fifth vocabulary
entry is preserved in metadata and is not silently normalized into the model
output space. The post-correction supported rerun's package training loop and
loss passed the recorded S1 fixed-batch pre-optimizer comparison and S2
one-step backward, full-model clipping, AdamW state, post-step parameter, and
step-cadence scheduler comparison. A three-batch S3 source-generated probe
reached the first update but diverged on its second pre-clip gradient surface,
so no multi-batch, loader, trained-checkpoint, or full-run parity is claimed.

## Seed Policy

The S1-S3 source-generated checks use fixed parity seed `261`; they are not
training-seed or evaluation-seed reproduction results. The static configuration
records seed `1`, matching the checked original recipe. The eventual matched
S5 matrix is vendor and package runs for the same approved data/config under
training seeds `1`, `2`, and `3`, followed by the same evaluation seed scope;
only seed `1` is currently documented and no matrix run has started. S4
requires the approved assets and S3 must pass first. S5 and the real-scale
300-step lockstep probe are prohibited until S0-S4 pass.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state | Accepted on the supported V100 rerun: 23 passed in 26.50s, 0 skipped, vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`; text encoding fields are derived from the selected mapper, and class-mapping and derived-output guards passed. |
| S1 | Fixed-batch pre-optimizer trace parity | Accepted post-correction: 1 passed in 33.76s, 0 skipped, executed=1, first divergence `none`, max abs `6.103515625e-05`, max rel `7.620204911518158e-08`. |
| S2 | One optimizer-step parity | Accepted post-correction: 1 passed in 59.98s, 0 skipped, executed=1, first divergence `none`, max abs `1.9073486328125e-06`, max rel `11.25` under the existing S2 tolerance. |
| S3 | Short deterministic multi-batch run | Blocked post-correction on supported V100 step 2: 1 failed, 0 skipped in 62.92s; first divergence `pre_clip_gradients.backbone.bottom_up.res3.0.conv1.weight`, max abs `0.0038013458251953125`, max rel `102.74234771728516`. This does not establish unconstrained production trajectory parity. |
| S4 | Deterministic loader stream | Not run: strict stage order stops at the S3 numerical divergence; approved CGL images/annotations, matching 768-D text features, and license/provenance approval also remain absent. |
| S5 | Full-run statistical comparison | Not claimed; prohibited until S0-S4 evidence exists. |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 .cache/radm/reference-env/bin/python -m pytest models/radm/tests/test_training_s0_contracts.py models/radm/tests/vendor_parity/test_s0_reference_adapter.py models/radm/tests/vendor_parity/test_s0_radm_topology.py -m 'not integration' -q -rs` | `models/radm/configs/training/effective_radm_config.yaml` | Accepted on physical GPU 0/logical `cuda:0`: 23 passed in 26.50s, 0 skipped, vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`; config SHA-256 `308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`. |
| S1 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s1_fixed_batch_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s1_radm_fixed_batch_pre_optimizer_parity -s -q -rs` | `.cache/radm/supported-v100/source-order-20260814/s1_fixed_batch_trace.json` | Accepted: 1 passed in 33.76s, executed=1, skipped=0, first divergence `none`, max abs `6.103515625e-05`, max rel `7.620204911518158e-08`, evidence SHA-256 `8f27ccd99b90fac3f1ed3e922308df7c8013197c7b05799e32f2055a7c66b63a`. |
| S2 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s1_fixed_batch_trace.json RADM_S2_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s2_one_step_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s2_radm_one_optimizer_step_parity -s -q -rs` | `.cache/radm/supported-v100/source-order-20260814/s2_one_step_trace.json` | Accepted: 1 passed in 59.98s, executed=1, skipped=0, first divergence `none`, max abs `1.9073486328125e-06`, max rel `11.25`, evidence SHA-256 `0af1cbd069e43f4e0005f8d728e569dd1b77e6800a2ea23bc1e750d974518181`. |
| S3 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s1_fixed_batch_trace.json RADM_S2_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s2_one_step_trace.json RADM_S3_EVIDENCE_PATH=.cache/radm/supported-v100/source-order-20260814/s3_multi_batch_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s3_radm_deterministic_multi_batch_parity -s -q -rs` | `models/radm/tests/vendor_parity/test_s1_radm_training.py` | Blocked: deterministic-algorithm harness restored its caller setting; 1 failed, 0 skipped in 62.92s at S3 step 2, first divergence `pre_clip_gradients.backbone.bottom_up.res3.0.conv1.weight`, max abs `0.0038013458251953125`, max rel `102.74234771728516`; no S3 artifact was written. |
| S4 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity -m 'vendor_parity and training' -rs` | `models/radm/src/radm/training/dataset.py` | Not run: strict order stops after the S3 numerical divergence, and approved CGL image/annotation, 768-D text-feature, and license/provenance prerequisites remain absent. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package radm --extra training traingen fit --config models/radm/configs/training/radm_cgl.yaml` | `.cache/radm/full-run/` | Not run and prohibited in phase 1; S5 is not claimed. |

## Reproduction Results

RADM S2 one-optimizer-step parity is accepted for one source-generated
in-memory fixture after accepted S1 fixed-batch pre-optimizer parity on the
supported V100 runtime. This is
not a dataset, training-seed, evaluation-seed, multi-batch, trained-checkpoint,
or full training reproduction result. The package trace uses the runtime
`RADMDenoiser`; it does not inject the original model into the package module.
Text inputs are generated as two deterministic 768-D rows and then padded by
the original mapper. The supported S3 multi-batch attempt is blocked by the
exact step-2 gradient divergence recorded above even under the deterministic
diagnostic algorithm setting. That setting is not a claim about unconstrained
production trajectory parity. Approved dataset assets and S4-S5 evidence
remain unavailable or pending.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| CGL | original | `blocked (no released checkpoint or approved data assets)` | not run | not measured | not measured | `models/radm/tests/vendor_parity/reference_adapter.py` |
| CGL | package | `not-yet-run (#261 phase 1)` | not run | not measured | not measured | `models/radm/src/radm/training/` |
| CGL-v2 | original | `blocked (license and data provenance unresolved)` | not run | not measured | not measured | `models/radm/configs/training/effective_radm_config.yaml` |
| CGL-v2 | package | `not-yet-run (#261 phase 1)` | not run | not measured | not measured | `models/radm/src/radm/training/` |

There is no released checkpoint. The eventual reference checkpoint must first
be created by the pinned original training procedure, then the package must be
trained under the same approved dataset/config/seed conditions. Only after both
checkpoints exist may the matched evaluation protocol compare them. That work
is outside this phase and is not claimed here.

## Regeneration Metadata

The committed static record is
`models/radm/configs/training/effective_radm_config.yaml` (SHA-256
`308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`). The
reference source revision used by the accepted S0 topology run is
`413f87a45760ceac5635b6a08c8047f86478acf5` in `vendor/radm`. Initialized-state
metadata from a permitted reference run belongs outside git under
`.cache/radm/reference-state/`. The S1 evidence artifact is
`.cache/radm/s1/fixed_batch_trace.json` (SHA-256
`22e65d7ad160e694bc56a8ffe7ac0d74a26d5fc6092b4a4c3d0447a19be25c58`). Its
fixture tensor hashes are: prepared image
`094511648c94ea5018cf06fce963885b6399e79c9fdad6c98ed1315f8e8e116a`, boxes
`8d84c32fcf0b82c02586289eae7748f2bce3fbc2683044b8e7da3289ccd4b36f`, labels
`96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a`, image
scales `e40c361d7eccd1db58edf4d1b54626358b168d27896224a935d154a9dabacf4b`,
text features `8cafba2145cfb78881368f9f503765f246e9dfe700cf298c005b5774e33247b`,
and text mask
`da774fa5934ac062f6c8dcf0b4747ec0e2cb6783768d13b234df2cf611f24ca5`. Do not
commit checkpoints, tensors, images, datasets, or generated reference outputs.

The supported-runtime S1 and S2 evidence artifacts are
`.cache/radm/supported-v100/s1_fixed_batch_trace.json` (SHA-256
`5c935d69c30e8d8ad14bd304cf73c7fed43c3c4f07ab60b94a510692f00a42b1`) and
`.cache/radm/supported-v100/s2_one_step_trace.json` (SHA-256
`0b0427b4fbc142688eac8f19633f9a7f8c3ae74d3bcfbec12d83ef10ee0e5ca4`). They use
the same source-generated fixture hashes above, source revision
`413f87a45760ceac5635b6a08c8047f86478acf5`, and config SHA-256
`308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`. The S3
failure occurred before its evidence writer, so no S3 artifact hash exists.

## Training Commands

Run the accepted S0 contracts without training data or checkpoints.

```bash
PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest \
  models/radm/tests/test_training_s0_contracts.py \
  models/radm/tests/vendor_parity/test_s0_reference_adapter.py \
  models/radm/tests/vendor_parity/test_s0_radm_topology.py \
  -m "not integration" -q -rs
```

Regenerate the accepted S1 trace from the pinned original mapper and graph with
the ignored in-memory fixture; this command does not download data or mutate
an optimizer.

```bash
CUDA_VISIBLE_DEVICES='' PARITY_REQUIRE=1 \
RADM_S1_EVIDENCE_PATH=.cache/radm/s1/fixed_batch_trace.json \
uv run --package radm --extra training --extra vendor pytest \
  models/radm/tests/vendor_parity/test_s1_radm_training.py -s -q
```

When the optional reference runtime and approved local assets are available,
capture initialized original-code state only; this command does not train.

```bash
uv run --package radm --extra vendor \
  python models/radm/tests/vendor_parity/capture_reference_state.py \
  --vendor-root ./vendor/radm \
  --output .cache/radm/reference-state/initialized.json
```

## Reference Checkpoint Gate

The pinned source documents this exact single-GPU training command from the
`vendor/radm` directory:

```bash
cd vendor/radm
python3 train_net.py --num-gpus 1 \
  --config-file configs/radm.yaml
```

Before that command is authorized, `configs/radm.yaml` must point
`DATASETS.DATASET_PATH`, `DATASETS.TEXT_FEATURE_PATH`, and `OUTPUT_DIR` at
approved runtime paths. The source README requires the CGL annotations, clean
training images, test images, and train/test text features; the pinned source
also requires its Detectron2/fvcore/iopath stack and documents Python 3.7,
PyTorch 1.8.0, and CUDA 11.1. The data license and provenance must be recorded
before downloading or using any of those assets.

The original run is expected to produce Detectron2 checkpoint state under its
selected `OUTPUT_DIR` (including `model_*.pth` and `last_checkpoint`), training
metrics/events, and evaluation output under `OUTPUT_DIR/inference/`. The
source evaluation command is:

```bash
cd vendor/radm
python3 train_net.py --num-gpus 1 \
  --config-file configs/radm.yaml \
  --eval-only --resume
```

The matching package run must use the same approved dataset/config and seed,
then its checkpoint must be evaluated on the same test stream with the same
COCO/evaluator settings. The source `metrics.py` additionally requires explicit
test-image, annotation, and label paths and does not provide all internal
`R_shm`/`R_sub` functions; those limitations must be recorded rather than
silently replaced by another metric. The seed matrix is therefore
`{vendor, package} x {1, 2, 3}` for future training-seed-n=3 evidence, with
the same evaluation seed scope for each pair. No vendor or package training,
checkpoint generation, or evaluation has started in this phase.

The following package training command is a future recipe and must not be run
in phase 1. It is shown only to preserve the member-scoped LightningCLI entry
surface.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
uv run --package radm --extra training traingen fit \
  --config models/radm/configs/training/radm_cgl.yaml
```
