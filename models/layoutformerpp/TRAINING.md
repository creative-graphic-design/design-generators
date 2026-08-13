---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

The retained LayoutFormer++ candidate has accepted S0-S4 evidence for all
twelve RICO25 and PubLayNet recipe families. S2 uses a stepped authoritative
reference scheduler at the optimizer cadence; S3 keeps the 12-family manual
numerical lockstep separate from bounded production `traingen fit` wiring
evidence over the ten runtime-distinct representative configs; and S4 compares
the pinned original loaders with the production package DataModule. The
required 300-step lockstep run is a diagnostic pre-S5 probe through those
loader outputs. No training-seed parity, trained-checkpoint parity, or S5
full-run reproduction is claimed.

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
is claimed. S3 and S4 use bounded slices of the authoritative original processed
RICO25 and PubLayNet splits supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT`;
no full dataset download is part of this candidate.

## Data

S0-S1 use the accepted source-shaped fixed fixtures. S3-S4 use the pinned
original processed split files through
`LAYOUTFORMERPP_PARITY_DATA_ROOT` and compare the original loader with the
package-local DataModule/DataLoader.

| Dataset | Source | Config or path |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | `ui-screenshots-and-hierarchies-with-semantic-annotations`; processed `pre_processed_20_25` slice used for S3-S4 |
| PubLayNet | `creative-graphic-design/PubLayNet` | processed `pre_processed_20_5` slice used for S3-S4 |

RICO25 persists separate public zero-based and sequence one-based maps. The maps
are joined by normalized semantic name and hashed; integer arithmetic is never
used for translation. The public/config/Hub slug is `rico25`; `rico` is accepted
only at named original CLI/cache boundaries.

## Configs

The twelve faithful LightningCLI YAMLs live under
`models/layoutformerpp/configs/training`. The package-local DataModule is the
existing production training path used by `traingen fit` and the bounded
loader-parity gate.

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

No training or evaluation seed scope is claimed. A complete twelve-family
training-seed claim would require 12 families by 2 systems by 3 training seeds,
or 72 full runs; PubLayNet relation is counted once. S5 remains stopped.

## Validation Stages

The stages below follow `docs/training-reproduction.md`.

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | all twelve recipe, topology, state, vocab, label-map, loss, optimizer, scheduler, seed-order, and checkpoint facts |
| S1 | Fixed-batch pre-optimizer trace parity | all twelve families pass on the selected CUDA device |
| S2 | One optimizer-step parity | accepted: authoritative scheduler, backward/optimizer state, post-step parameters, LR/cadence, RNG, and first divergence matched for all twelve recipes |
| S3 | Short deterministic multi-batch run | accepted: 12-family manual numerical lockstep passed; ten representative real-data `traingen fit` runs exercised production scheduler, logging, validation, and `ModelCheckpoint` wiring |
| S4 | Deterministic loader stream | accepted: production package DataModule/loaders matched pinned original RICO25/PubLayNet train and validation streams |
| S5 | Full-run statistical comparison | stopped; hard-gated by accepted S0-S4 and the later lockstep probe |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json` | PASS: 39 tests, all twelve families, zero skips; pinned original revision `1498ff300710b4fc204aece537582d37ca447fc7`; independent task-ID, topology, scheduler, loss, and state checks. |
| S1 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 python -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s1 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s1` | PASS: 12 tests, zero skips on physical GPU 0 as logical `cuda:0` (`torch 2.12.0+cu126`, CUDA 12.6, SM 7.0, float32); `rtol=1e-4`, `atol=1e-5`; first divergence `null`; max absolute error `3.814697265625e-06`; max relative error `8.155166142387316e-08`. |
| S2 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 python -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s2 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s2` | PASS: 12 tests, zero skips; one real backward and optimizer step per recipe matched gradients, clipping behavior, optimizer state, post-step parameters, authoritative scheduler cadence/LR, RNG, and first divergence within `rtol=1e-4`, `atol=1e-5`. |
| S3 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT="${LAYOUTFORMERPP_PARITY_DATA_ROOT:?set authoritative original processed data root}" python -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s3 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s3` | PASS: 22 tests, zero skips: 12 manual numerical lockstep cases plus 10 production `traingen fit` representative cases (`rico25_label`, `rico25_label_size`, `rico25_relation`, `rico25_refinement`, `rico25_completion`, `rico25_unconditional`, `publaynet_label`, `publaynet_label_size`, `publaynet_relation`, `publaynet_refinement`). Every console run exited 0, reached `global_step=2` and two optimizer steps, recorded scheduler `last_epoch=1` and the expected post-step LR, delivered `train_loss`/`val_loss` to the CSV logger, and selected a real `ModelCheckpoint` file. The ordinary 12-YAML matrix guard proves shared Trainer/DataModule/model/checkpoint wiring for all recipes; manual numerical parity remains the separate all-family claim, while PubLayNet completion/unconditional are not separately claimed as console-fit runs. |
| S4 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT="${LAYOUTFORMERPP_PARITY_DATA_ROOT:?set authoritative original processed data root}" python -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m 'vendor_parity and training' -k s4 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s4` | PASS: 12 tests, zero skips; production package DataModule/loaders matched pinned original RICO25 and PubLayNet train/validation streams, including order, bytes, tokenization, masks/padding, task IDs, and first divergence. |
| S5 | `PARITY_REQUIRE=1 python -c "print('S5 intentionally stopped before full training/evaluation')"` | `.cache/layoutformerpp/s0/static-parity.json#s5-stop` | STOPPED/NOT CLAIMED: deliberately stopped before full training/evaluation, trained-checkpoint comparison, and training-seed claims. |

The loader-based 300-step real-scale lockstep diagnostic passed for the
`rico25_label` recipe after S0-S4. It recorded per-step loss, gradient norm,
parameter difference, scheduler/LR, loader order, and RNG/state hashes for 300
steps; first divergence was `null`. This remains diagnostic preflight only and
is not a trained-checkpoint, training-seed, or S5 claim.

The production `traingen fit` runs are wiring evidence only, not an additional
12-family numerical parity claim. They used the package
`LayoutFormerPPDataModule` and `LayoutFormerPPTrainingModule` with the
authoritative real-data root. The ten representatives cover both datasets,
all six conditioning paths, the distinct train/evaluation batch branches,
PubLayNet label-size flags, and the PubLayNet relation multitask partition
path. Native 20/50 validation cadence values are statically validated from
YAML; runtime validation/logging/checkpoint wiring was exercised at bounded
cadence 1. The ordinary 12-YAML recipe guard proves that all
recipe configs select the same Trainer/DataModule/model/checkpoint wiring
branch; the two PubLayNet families not directly run as console representatives
remain covered by the separate 12-family manual numerical gate only.

## Reproduction Results

No trained-family result exists. All twelve recipe families have accepted S0-S4
stage evidence; the 300-step result is a diagnostic pre-S5 probe only.
There is no training-seed, trained-checkpoint, or full-run metric claim. S5
remains intentionally stopped.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| RICO25 label | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label-size | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label-size | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |

Released-checkpoint trainer provenance and source/license approval remain
unresolved. They block comparison against released weights and publication, but
do not invalidate the accepted bounded S0-S4 candidate. No S5 run is
authorized by this evidence.

## Regeneration Metadata

The authoritative local manifest contains the worktree commit, exact
original-source revision, reviewed boundary hashes, RICO25 map hash,
recipe/test/document hashes, S0-S4 command results, the production Trainer
wiring smoke, the loader-based 300-step diagnostic, pinned DeepSpeed source
hash, and late-seed/captured-initialization contract at:

```text
.cache/layoutformerpp/s0/static-parity.json
```

The manifest's `manifest_generation` object is the single machine-readable
recipe for reproducing its aggregate Python/YAML digest; do not create a second
manifest or helper.

The [issue #265 evidence comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-5264137469)
is the durable record for the rejected predecessor candidate, not the current
manifest. The current S0-S4 evidence uses the production package DataModule
and the pinned original loaders. The 300-step artifact is diagnostic only; S5
remains explicitly stopped.

Official Lightning/PyTorch documentation and the pinned GitHub source were
consulted before the loader correction. The implementation follows the
documented DataModule/DataLoader ownership and the original split/sampler path;
vendor code remains gated under `tests/vendor_parity`.

## Training Commands

Rerun the ordinary static checks.

```bash
uv run --package layoutformerpp --extra training --no-sync pytest \
  models/layoutformerpp/tests -m "not vendor_parity and not integration" -q
```

Rerun S0 original-code parity with missing assets treated as failures.

```bash
PARITY_REQUIRE=1 \
  uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest \
  models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py \
  -m "vendor_parity and training" -k s0 -rs -q
```

Run S1-S4 with the compatible activated environment and set
`LAYOUTFORMERPP_PARITY_DATA_ROOT` to the authoritative processed source tree
for S3-S4. Keep `PARITY_REQUIRE=1`; missing source assets must fail the run.

`traingen fit` uses the package-local DataModule; checkpoint conversion and S5
full-run commands remain deferred.
