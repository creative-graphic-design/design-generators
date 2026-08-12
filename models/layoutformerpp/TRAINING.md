---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

This package currently provides static training-parity infrastructure only. The
remediated S0 candidate checks pass for all twelve RICO25 and PubLayNet recipe
families: the ordinary gate recorded 55 passed with 68 deselected, the real-source
S0 gate recorded 39 passed, and the focused task-ID gate recorded 12 passed. No
optimizer update, DataLoader, trained checkpoint, or full-run reproduction is
claimed. Issue #9 is the closed inference implementation issue; issue #265
tracks training reproduction. Its first durable S0 comment records the rejected
25-test candidate. Independent read-only review accepted the remediated technical
S0 gate; local manifest provenance is complete, and separately authorized durable
posting remains before any S1 work; S1-S5 stay stopped.

Run commands from the repository root. Keep generated evidence, logs, data, and
checkpoints under `.cache/layoutformerpp/`.

## Install

```bash
uv sync --package layoutformerpp --extra training
```

Install the original-code adapter dependencies only for static parity checks.

```bash
uv sync --package layoutformerpp --extra training --extra vendor
```

The package runtime and training namespace do not import DeepSpeed or Detectron2.
The S0 vendor-parity harness reads the original requirements pin and constructs
the real DeepSpeed 0.5.10 `WarmupLR` class in an isolated `uv --with` overlay;
the original package import is incompatible with the currently verified Torch
because it imports the removed `torch._six` module. No dependency-design change
is claimed. Original data/evaluation parity may add further dependencies later
behind the `vendor` extra after the required research gate succeeds.

## Data

S0 constructs no dataset or DataLoader. These sources are reserved for the
deferred data slice.

| Dataset | Source | Config or path |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | `ui-screenshots-and-hierarchies-with-semantic-annotations`; ingestion pending |
| PubLayNet | `creative-graphic-design/PubLayNet` | ingestion pending |

RICO25 persists separate public zero-based and sequence one-based maps. The maps
are joined by normalized semantic name and hashed; integer arithmetic is never
used for translation. The public/config/Hub slug is `rico25`; `rico` is accepted
only at named original CLI/cache boundaries.

## Configs

The twelve faithful LightningCLI YAMLs live under
`models/layoutformerpp/configs/training`. They intentionally have no `data`
section until the later DataModule slice.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `rico25_{label,label_size,relation,refinement,completion,unconditional}.yaml` | RICO25 | original CLI late-seed behavior recorded | six faithful static recipes |
| `publaynet_{label,label_size,relation,refinement,completion,unconditional}.yaml` | PubLayNet | original CLI late-seed behavior recorded | six faithful static recipes |

PubLayNet relation is one family with task order `refinement,gen_ts,gen_t,completion,ugen,gen_r`
and package task-ID tuple `0,4,3,1,2,5`; its partition buckets are
`-1,-1,-2,0,0,-3`. Each single-task family also persists its package task ID:
refinement `0`, completion `1`, unconditional `2`, label `3`, label-size `4`,
and relation `5`. S0 compares those package-owned values with task IDs extracted
independently from the original implementation. Diagnostic deterministic and
smoke configs are deferred.

## Scheduler and Recipe Notes

The reference mode is plain PyTorch `basic`; the distributed DeepSpeed trainer
remains secondary, provenance-gated evidence. The basic path nevertheless
imports DeepSpeed's pinned `WarmupLR`, so S0 constructs that real scheduler class
through the vendor-only compatibility probe and compares it with the package
scheduler. Every faithful recipe uses Adam at `1e-4` with default betas and
epsilon, zero weight decay, accumulation one, and no clipping, EMA, or AMP.
`vendor_effective_cross_entropy` includes pad targets; the runtime model's
ordinary forward loss remains pad-masked.

The logarithmic WarmupLR starts at index `-1` without an eager scheduler step.
The first optimizer update uses `1e-4`; its post-update scheduler call writes
`0`; subsequent post-update values are `1e-4 * log(step + 1) / log(W)` until the
warmup maximum. Lightning receives the scheduler with `interval: step`, and the
module advances it exactly once after each optimizer update.

Validation occurs every 20 RICO25 epochs or 50 PubLayNet epochs. The selected
checkpoint is the minimum aggregate evaluation-loss epoch, matching the
original `best_epoch` decision. Original basic checkpoints contain model state
only, so exact optimizer/scheduler/RNG resume is not a parity claim.

## Seed Policy

The original CLI constructs the model before its trainer applies the loader and
experiment seed. The paired-run harness will therefore seed ambient construction
separately for each replicate, capture that distinct initialized state, copy it
to both systems, and record the construction control, `initial_state_sha256`,
and late loader-transform seed. This is a documented harness deviation from the
unmodified late-seed CLI. Reusing one initial state for all three replicates does
not constitute three distinct training-seed replicates.

No training or evaluation seed scope is claimed in this S0-only slice. A complete
twelve-family claim would require 12 families by 2 systems by 3 training seeds,
or 72 full runs; PubLayNet relation is counted once.

## Validation Stages

The stages below follow `docs/training-reproduction.md`.

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | all twelve recipe, topology, state, vocab, label-map, loss, optimizer, scheduler, seed-order, and checkpoint facts |
| S1 | Fixed-batch pre-optimizer trace parity | pending; outside this slice |
| S2 | One optimizer-step parity | pending; outside this slice |
| S3 | Short deterministic multi-batch run | pending; outside this slice |
| S4 | Deterministic loader stream | pending; outside this slice |
| S5 | Full-run statistical comparison | pending and hard-gated by family-local S0-S4 plus the 300-step real-scale lockstep probe |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s0 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json`; [issue #265 rejected predecessor evidence](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-5264137469) | TECHNICAL S0 ACCEPTED by independent read-only review: 39 tests cover all twelve families, including independent original-versus-package task-ID checks, using local original source and the pinned scheduler distribution; focused task-ID slice recorded 12 passed; no skips; accepted evidence is not yet durably posted; S1-S5 remain stopped |
| S1 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 python -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s1 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s1` | PASS: 12 families, 12 passed, 0 skipped on physical GPU 0 as logical `cuda:0` (`torch 2.12.0+cu126`, CUDA 12.6, SM 7.0, float32); `rtol=1e-4`, `atol=1e-5`; first divergence `null`; max absolute error `3.814697265625e-06`; max relative error `8.155166142387316e-08`; no model-state mutation, paired RNG mismatch, or caller-RNG restoration failure; S2-S5 remain gated |
| S2 | `PENDING` | `PENDING` | `PENDING` |
| S3 | `PENDING` | `PENDING` | `PENDING` |
| S4 | `PENDING` | `PENDING` | `PENDING` |
| S5 | `PENDING` | `PENDING` | `PENDING` |

## Reproduction Results

No trained-family result exists. All twelve checkpoint families remain
not-yet-run with no seed scope or metrics; #9 tracks prior inference work, while
#265 tracks S0; its durable 25-test candidate was independently rejected, and
the remediated 39-test technical S0 gate was independently accepted. No S1-S5
evidence or trained-checkpoint claim exists.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| RICO25 label | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 label | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet label | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet label | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet label-size | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet label-size | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | original | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | package | `not-yet-run (#265; S0 candidate only)` | not run | not run | S0 static only | `.cache/layoutformerpp/s0/` |

Released-checkpoint trainer provenance and source/license approval remain
unresolved. They block comparison against released weights and publication, but
do not block future paired fresh original/package runs once S1-S4 and the
lockstep gate pass.

## Regeneration Metadata

The authoritative remediated local manifest contains the worktree commit,
original-source revision, reviewed boundary hashes, RICO25 map hash,
recipe/test/document hashes, command results, pinned DeepSpeed source hash, and
late-seed/captured-initialization contract at:

```text
.cache/layoutformerpp/s0/static-parity.json
```

The manifest's `manifest_generation` object is the single machine-readable
recipe for reproducing its aggregate Python/YAML digest; do not create a second
manifest or helper.

The [issue #265 evidence comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-5264137469)
is the durable record for the rejected predecessor candidate, not the current
manifest. The current technical S0 review is independently accepted, but the
accepted evidence is not yet durably posted and neither local record authorizes
S1.

Native official-document and GitHub-restricted research both failed with
`403 Selected provider is forbidden`; the S0 adapter therefore uses the local
primary source checkout and its pinned requirements. Dependency metadata was not
changed because the required external research stages were inaccessible.

## Training Commands

Rerun the ordinary static checks.

```bash
uv run --package layoutformerpp --extra training --no-sync pytest \
  models/layoutformerpp/tests/test_training_imports.py \
  models/layoutformerpp/tests/test_training_recipes.py \
  models/layoutformerpp/tests/test_training_lightning.py \
  models/layoutformerpp/tests/test_layoutformerpp_processor.py \
  models/layoutformerpp/tests/test_model_card.py -q
```

Rerun S0 original-code parity with missing assets treated as failures.

```bash
PARITY_REQUIRE=1 \
  uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest \
  models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py \
  -m "vendor_parity and training" -k s0 -rs -q
```

`traingen fit`, checkpoint conversion, and local loading commands are deferred
until their respective package-local DataModule and later evidence slices exist.
