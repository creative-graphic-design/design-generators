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
S1, and S2 on the source-generated fixed batch, and S4 accepts the bounded CGL
loader stream. S3 is recorded in three separate layers: synchronized numerical
diagnostics, natural-trajectory drift recording, and a bounded production
wiring representative. The natural trajectory is not accepted as unconstrained
trajectory parity. The approved RADM data archive is available through the
local cache. No released checkpoint is present in this worktree. S5 evidence
is not claimed.

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
768-dimensional text features. The package adapter accepts explicit local paths
and never downloads data.

The approved S4 gate requires, at minimum, the authorized CGL train/test
annotations, the cleaned CGL training images produced through the original
LaMa preprocessing requirement, the matching train/test 768-D text-feature
trees, and a recorded license/data-provenance decision. These assets must be
available at the selected local paths before S4; the source-generated S1-S3
fixtures are not substitutes for them.

| Dataset | Source | Config or path |
| --- | --- | --- |
| CGL | `creative-graphic-design/CGL-Dataset` when approved by the project data policy | `.cache/radm/data/cgl/` |
| CGL-v2 | approved vendor README distribution; full S4 stream remains blocked | `.cache/radm/data/cgl/` |

### Approved data provenance

On 2026-08-14 the user approved the source distribution named by the pinned
RADM README: CGL training data from the Tianchi CGL dataset, including the
LaMa-clean training images, and testing data plus precomputed text features
from the README distribution at `3.cn/10-dQKDKG` (`RADM_dataset.tar.gz`). No
text encoder is used. The existing local archive was reused without download;
its SHA-256 is
`1348a2ad70513c90287c5d2ae6d4aa87b70c49676cac5f3531cbff62610fb75b`.
The materialized path is `.cache/radm/data/cgl/` and contains the source
`annotations/`, `images/`, `texts/`, and `text_features/` layout with five
source categories and train/test splits. Missing text features use the source
all-zero/all-padding fallback only in the S4 diagnostic; the package default
remains strict.

### S4 transform boundaries

The source mapper's effective train path reads RGB `uint8` pixels, applies the
existing flip and shortest-edge resize, transforms integer pixel boxes, and
normalizes only after those operations. The package loader now preserves that
ordering and resampling behavior while keeping its public normalized box
output. On the selected V100, a same-state five-seed control matched source
image and normalized-box tensors exactly for both flip and no-flip cases; the
single-sample loader evidence below also records zero error for both surfaces.

## Configs

Training configs live under `models/radm/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `effective_radm_config.yaml` | CGL and CGL-v2 | captured static state | Mechanical record of effective model, optimizer, scheduler, sampler, and input values. |
| `radm_cgl.yaml` | CGL | deterministic | Bounded production wiring representative; not a full training run. |
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
step-cadence scheduler comparison. The bounded production wiring representative
also exercised the same loss path through validation, step-cadence scheduler,
CSV logging, and monitored checkpoint creation. The natural three-batch S3
trajectory still records its accepted-contract step-2 drift; this is separate
from the wiring PASS and is not a multi-batch trajectory parity claim.

## Seed Policy

The S1-S3 source-generated checks use fixed parity seed `261`; they are not
training-seed or evaluation-seed reproduction results. The static configuration
records seed `1`, matching the checked original recipe. The eventual matched
S5 matrix is vendor and package runs for the same approved data/config under
training seeds `1`, `2`, and `3`, followed by the same evaluation seed scope;
only seed `1` is currently documented and no matrix run has started. The S3
synchronized/natural trajectory acceptance rule is pending the protocol
generalization in meta issue [#268](https://github.com/creative-graphic-design/design-generators/issues/268)
and the RALF protocol update in PR [#270](https://github.com/creative-graphic-design/design-generators/pull/270).
S5 and the real-scale 300-step lockstep probe remain stopped by scope.

## S3 Evidence Layers

S3 evidence intentionally separates diagnostic state synchronization from the
natural trajectory and from the production wiring boundary. Existing tolerances
are unchanged.

| Layer | Result | Evidence |
| --- | --- | --- |
| Synchronized numerical diagnostic | `PASS` as diagnostic only; `first_divergence: None`, `executed=1`, `skipped=0`, with optimizer-state storage independence and zero-diff synchronization assertions | `.cache/radm/s3/runs/run-005/s3-trace.json` (SHA-256 `9172d0f21940e56e1482f048cff3fde5d6d1f54adcfa2b4b2e24a79aa5523e64`); aggregate `.cache/radm/s3/two-layer-20260814-rerun.json` (SHA-256 `b4498680c80c04eb193e7166e79553fa8d9ab2e93120b039e4ba291f9b60677d`) |
| Natural trajectory recording | `RECORDED`, not trajectory-parity acceptance; both runs first diverge at S3 step 2 on `pre_clip_gradients.backbone.bottom_up.res3.0.conv1.weight` | `.cache/radm/s3/runs/run-003/s3-trace.json` (SHA-256 `6248349c9f34a627739add18b9aec4a2f4d172446c82b1e31d15f4d599cb44fc`), `.cache/radm/s3/runs/run-004/s3-trace.json` (SHA-256 `a6252d4e1d4ea3cf7459b7d4fc0166c9c9fd69a73b58c37e7393b963a70b5cb2`) |
| Production wiring representative | `PASS`; two optimizer steps, scheduler/checkpoint cadence, validation loss logging, and checkpoint payload were observed through `traingen fit` | `.cache/radm/s3/wiring/run-002.json` (SHA-256 `3709a048fbfd2044464c55d5aa10207af1cfbbad450d72c463e341f509a38c01`); checkpoint SHA-256 `bc49decd95cd81f1177f730aced255f3b339f08e169b58721f32b2170df0cdf6`; metrics SHA-256 `7bd7ed5d71044db8f618941ae9952849d60d48190e9c7967574ce9db6aa6c682` |

The synchronized run is a graph/operation and state-synchronization diagnostic;
the natural runs record the unconstrained drift envelope. Neither layer alone
establishes a trained-checkpoint or full-run reproduction claim.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state | Accepted on the supported V100 rerun: 23 passed in 26.50s, 0 skipped, vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`; text encoding fields are derived from the selected mapper, and class-mapping and derived-output guards passed. |
| S1 | Fixed-batch pre-optimizer trace parity | Accepted post-correction: 1 passed in 33.76s, 0 skipped, executed=1, first divergence `none`, max abs `6.103515625e-05`, max rel `7.620204911518158e-08`. |
| S2 | One optimizer-step parity | Accepted post-correction: 1 passed in 59.98s, 0 skipped, executed=1, first divergence `none`, max abs `1.9073486328125e-06`, max rel `11.25` under the existing S2 tolerance. |
| S3 | Two-layer numerical diagnostic plus production wiring representative | Synchronized diagnostic: `first_divergence=None`, `1 executed`, `0 skipped`; natural runs: `RECORDED` with first divergence at S3 step 2 on `pre_clip_gradients.backbone.bottom_up.res3.0.conv1.weight`; wiring representative: `1 passed` with exit code 0, two optimizer steps, scheduler/checkpoint cadence, CSV `train_loss`/`val_loss`, and monitored checkpoint. Natural trajectory acceptance awaits meta issue #268 and RALF PR #270. |
| S4 | Deterministic loader stream | The supported V100 loader boundary is exact for train/test order, labels, image preprocessing, boxes, and missing-feature fallback in the recorded CGL fixture; the S3 natural trajectory remains a separate diagnostic boundary and CGL-v2 coverage is not claimed. |
| S5 | Full-run statistical comparison | Not claimed; stopped by current scope. |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 .cache/radm/reference-env/bin/python -m pytest models/radm/tests/test_training_s0_contracts.py models/radm/tests/vendor_parity/test_s0_reference_adapter.py models/radm/tests/vendor_parity/test_s0_radm_topology.py -m 'not integration' -q -rs` | `models/radm/configs/training/effective_radm_config.yaml` | Accepted on physical GPU 0/logical `cuda:0`: 23 passed in 46.32s, 0 skipped, vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`; config SHA-256 `308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`. |
| S1 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s1_radm_fixed_batch_pre_optimizer_parity -s -q -rs` | `.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json` | Accepted: 1 passed, 0 skipped, first divergence `none`, max abs `6.103515625e-05`, max rel `7.620204911518158e-08`, evidence SHA-256 `8f27ccd99b90fac3f1ed3e922308df7c8013197c7b05799e32f2055a7c66b63a`. |
| S2 | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="models/radm/src:models/radm/tests/vendor_parity:lib/laygen/src:lib/posgen/src:lib/traingen-parity/src" PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json RADM_S2_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s2_radm_one_optimizer_step_parity -s -q -rs` | `.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json` | Accepted: 1 passed, 0 skipped, first divergence `none`, max abs `1.9073486328125e-06`, max rel `11.25`, evidence SHA-256 `0af1cbd069e43f4e0005f8d728e569dd1b77e6800a2ea23bc1e750d974518181`. |
| S3 | `CUDA_VISIBLE_DEVICES=0 RADM_PHYSICAL_GPU=0 PARITY_REQUIRE=1 RADM_S3_WIRING_EVIDENCE_PATH=.cache/radm/s3/wiring/run-002.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s3_radm_wiring.py -s -q -rs` plus the recorded synchronized/natural runs | `.cache/radm/s3/two-layer-20260814-rerun.json` | Synchronized diagnostic `first_divergence=None`; natural drift recorded; wiring representative `1 passed` in 193.97s with exit code 0, `global_step=2`, `optimizer_steps=2`, `scheduler_last_epoch=2`, final LR `2.9950000000000005e-07`, CSV `train_loss`/`val_loss`, and monitored checkpoint. |
| S4 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 RADM_S4_ALLOW_MISSING=1 RADM_S4_DATA_ROOT=.cache/radm/data/cgl RADM_S4_ARCHIVE_SHA256=1348a2ad70513c90287c5d2ae6d4aa87b70c49676cac5f3531cbff62610fb75b RADM_S4_EVIDENCE_PATH=.cache/radm/s4/run-012-image-box-resampling.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s4_radm_loader_stream.py -s -q -rs` | `.cache/radm/s4/run-012-image-box-resampling.json` | Supported V100 (physical GPU 0, logical `cuda:0`; Tesla V100-SXM2-32GB, torch `2.6.0+cu124`, CUDA `12.4`): `1 passed, 0 skipped`, `first_divergence: none`; evidence SHA-256 `304892b296c24f7bec86e92ffcc4b6ab380ab62bfc9a7ce70dc414b3a9c59415`. Train order `fbf1603af946c5b606d154bf0acf02f7ab651bd1e65aab21b41e4003314a4e0c`, test order `2c6e86368deb082930d4be32fd4403634157f172880f7e7f4b43f4ef19fe964d`; aligned labels, image, boxes, text features, masks, and fallback are exact (`max_abs=0.0`). |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package radm --extra training traingen fit --config models/radm/configs/training/radm_cgl.yaml` | `.cache/radm/full-run/` | Not run and stopped by current scope; S5 is not claimed. |

## Reproduction Results

RADM S2 one-optimizer-step parity is accepted for one source-generated
in-memory fixture after accepted S1 fixed-batch pre-optimizer parity on the
supported V100 runtime. S3 now has a bounded production wiring representative
PASS and two diagnostic layers: the synchronized lockstep has
`first_divergence=None`, while the natural trajectory runs record the same
step-2 pre-clip gradient drift. This is not a natural trajectory, dataset,
training-seed, evaluation-seed, trained-checkpoint, or full training
reproduction result. The package trace uses the runtime `RADMDenoiser`; it does
not inject the original model into the package module. Text inputs are
generated as two deterministic 768-D rows and then padded by the original
mapper. The synchronized/natural trajectory acceptance rule awaits the
protocol generalization in meta issue #268 and RALF PR #270; no tolerance was
changed. The approved archive is materialized locally, and the bounded CGL S4
loader boundary is exact for order, labels, fallback, image preprocessing, and
boxes. This does not make the untested CGL-v2 stream a reproduction claim. S5
and the real-scale 300-step lockstep probe remain prohibited.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| CGL | original | `blocked (natural S3 acceptance pending; no trained checkpoint)` | not run | not measured | not measured | `models/radm/tests/vendor_parity/reference_adapter.py` |
| CGL | package | `not-yet-run (#261 phase 1)` | not run | not measured | not measured | `models/radm/src/radm/training/` |
| CGL-v2 | original | `blocked (natural S3 acceptance pending; S4 coverage not claimed)` | not run | not measured | not measured | `models/radm/configs/training/effective_radm_config.yaml` |
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

The current supported-runtime S1 and S2 regression artifacts are
`.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json`
(SHA-256 `8f27ccd99b90fac3f1ed3e922308df7c8013197c7b05799e32f2055a7c66b63a`)
and `.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json`
(SHA-256 `0af1cbd069e43f4e0005f8d728e569dd1b77e6800a2ea23bc1e750d974518181`).
They use the same source-generated fixture hashes above, source revision
`413f87a45760ceac5635b6a08c8047f86478acf5`, and effective static config
SHA-256 `308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`.
The two-layer S3 aggregate is
`.cache/radm/s3/two-layer-20260814-rerun.json` (SHA-256
`b4498680c80c04eb193e7166e79553fa8d9ab2e93120b039e4ba291f9b60677d`), with
natural run roots `.cache/radm/s3/runs/run-003` and `run-004`, and synchronized
run root `.cache/radm/s3/runs/run-005`. The production wiring evidence is
`.cache/radm/s3/wiring/run-002.json` (SHA-256
`3709a048fbfd2044464c55d5aa10207af1cfbbad450d72c463e341f509a38c01`); its
disposable checkpoint and CSV metric hashes are recorded in the S3 evidence
layer above. These artifacts remain outside git.

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
