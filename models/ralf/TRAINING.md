---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - RALF
---

# RALF Training

This document records package-local training commands and staged reproduction
evidence for RALF. CGL has accepted S0-S4 evidence under the general S3
Evidence Layers rule at one seed; PKU loader and full-run reproduction are not
claimed.
CGL S3 is formally accepted under the general S3 Evidence Layers rule: natural
runs 009/010 exited the existing S0-S2 contract, synchronized run-012 supplied a
bounded contract-internal PASS, and the production-wiring layer is recorded
independently. Only the S5 scope remains pending. No S5 or real-scale 300-step
probe has been run.

Run commands from the repository root. Generated data, logs, checkpoints, and
downloaded assets remain under `.cache/ralf/` or the authoritative cache
selected below and are not committed.

## Install

```bash
uv sync --package ralf --extra training
```

Install the `vendor` extra only when rerunning the original-code parity checks.

```bash
uv sync --package ralf --extra training --extra vendor
```

## Data

The package-local CGL `RalfDataModule` and the original RALF loader consume the
same authoritative cache for the accepted CGL S0-S4 evidence. The cache is
provided read-only through the `RALF_CACHE_DIR` environment variable; its
machine-specific mount path is intentionally not recorded in repository docs.

| Dataset | Source | Config or path |
| --- | --- | --- |
| CGL | RALF cache supplied through `RALF_CACHE_DIR` | `$RALF_CACHE_DIR/dataset/cgl`, CGL train/validation splits |
| PKU | RALF cache supplied through `RALF_CACHE_DIR` | `$RALF_CACHE_DIR/dataset/pku`, not run in the accepted S4 evidence |

Its relevant contents include the CGL/PKU dataset trees, retrieval-index
tables, the ResNet and FIDNet precomputed weights, DreamSim retrieval data,
relationship tables, saliency/FAISS data, FID evaluation features, and the
original training-log directories. It is an authoritative read-only reference
for these runs. The cache is not localized into the repository worktree; the
currently selected user-provided local copy is supplied through
`RALF_CACHE_DIR`. Run provenance records the cache content-role inventory and a
per-file SHA256 manifest; the external cache is identified only by
`RALF_CACHE_DIR`, never by its machine-specific mount path.

The accepted CGL stream uses the vendor-effective transforms `image`,
`sort_label`, and `sort_lexicographic`, fixed retrieval indexes with
`top_k=16`, `random_retrieval=false`, and disjoint train/validation membership.

## Configs

Training configs live under `models/ralf/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `cgl.yaml` | CGL | fixed seed, deterministic warning mode | Package-local Lightning training and CGL parity |
| `pku.yaml` | PKU | fixed seed, deterministic warning mode | Package-local training configuration; parity not yet run |

## Scheduler and Recipe Notes

The package training path uses `RalfTrainingModule` and `RalfDataModule`
through the member-scoped `traingen fit` construction. The recipe uses AdamW,
weight decay `1e-4`, gradient clipping at `0.1`, batch size `32`, and
`accumulate_grad_batches=1`. CGL uses 70 epochs, three train batches and two
validation batches per epoch in the accepted S3 production control.

The CGL scheduler is `MultiStepLR` with the vendor milestone at epoch 49
(`0.7 * 70`), stepped at the epoch boundary. The accepted S3 run crossed that
boundary and compared package/vendor learning-rate groups and
`last_epoch=70`. Both configs use Lightning's `deterministic: warn`; the
effective PyTorch state has deterministic algorithms enabled with warn-only
enabled, matching the supported V100 diagnostic protocol.

## Seed Policy

CGL S0-S4 evidence is `training-seed n=1`, seed `1`, on one selected Tesla
V100-SXM2-32GB. This is diagnostic/staged evidence, not S5 training-seed
evidence. PKU has no accepted S0-S4 or S5 seed evidence. S5 remains pending and
must not be inferred from the CGL stage ladder.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | Confirms topology, parameter/state keys, optimizer groups, scheduler milestone, and encoded dataset state. |
| S1 | Fixed-batch pre-optimizer trace parity | Confirms prepared ids/labels, logits, and loss before optimizer mutation. |
| S2 | One optimizer-step parity | Confirms named raw/clipped gradients, optimizer state, post-step parameters, and learning rates. |
| S3 | Production multi-batch diagnostic | Confirms production Trainer construction, scheduler/logging/accumulation/checkpoint wiring, with state-synchronized and natural trajectory layers kept separate. |
| S4 | Deterministic loader stream | Confirms train/validation order, transforms, masks, padding, retrieval tensors, and split membership. |
| S5 | Full-run statistical comparison | Not started; no trained-checkpoint reproduction claim is made. |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 $RALF_AUDIT_PYTHON models/ralf/tests/vendor_parity/run_training_stages.py --stage S0 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/warn-mode-001/s0.json --seed 1` | `.cache/ralf/training-reproduction/cgl/warn-mode-001/s0.json` | PASS; artifact SHA256 `8c9982060b791a8fb34e1b122dec281a59d9ada18e260063212935e3b62b0dc0`. |
| S1 | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 $RALF_AUDIT_PYTHON models/ralf/tests/vendor_parity/run_training_stages.py --stage S1 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s1-after-loader-order-002/s1.json --steps 1 --batch-size 32 --seed 1` | `.cache/ralf/training-reproduction/cgl/s1-after-loader-order-002/s1.json` | PASS; artifact SHA256 `88b08d6b80b6c95af3c48501a89c280514a3086cb6f6550272a43419f22291d1`. |
| S2 | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 $RALF_AUDIT_PYTHON models/ralf/tests/vendor_parity/run_training_stages.py --stage S2 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s2-after-loader-order-002/s2.json --steps 1 --batch-size 32 --seed 1` | `.cache/ralf/training-reproduction/cgl/s2-after-loader-order-002/s2.json` | PASS; artifact SHA256 `22825974c7c8aeca7f29724253c24d667f3823293919a0c26819845bf0193f27`. |
| S3 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 uv run --active --no-sync --package ralf --extra training --extra vendor traingen fit --config models/ralf/configs/training/cgl.yaml ...` | `.cache/ralf/training-reproduction/cgl/s3/runs/run-012/trace/s3-trace.json` | PASS under the general S3 Evidence Layers rule: natural runs 009/010 left the S0-S2 contract; synchronized run-012 bounded PASS with first divergence `null`; production wiring is recorded independently. |
| S4 | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 $RALF_AUDIT_PYTHON models/ralf/tests/vendor_parity/run_training_stages.py --stage S4 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json --steps 8 --batch-size 32 --seed 1` | `.cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json` | PASS; artifact SHA256 `4c989f7e87444d2275d775c06fadcdcfbcff36acf18caca013f6ec156a7106c8`. |
| S5 | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 uv run --package ralf --extra training traingen fit --config models/ralf/configs/training/cgl.yaml --trainer.max_epochs=70` | `.cache/ralf/training-reproduction/cgl/s5/` | Not started; S5 scope remains pending coordinator authorization for issue #44, so this command has not been run. |

`RALF_AUDIT_PYTHON` and `RALF_CACHE_DIR` in the table are the explicit
environment variables shown in [Regeneration Metadata](#regeneration-metadata).

## Reproduction Results

CGL package-local training currently has accepted S0-S4 staged evidence at
`training-seed n=1`, including exact authoritative train/validation stream
parity and formally accepted general-rule S3 evidence. S5 full-run training and
evaluation have not started, and PKU has no accepted training-stage evidence;
therefore no trained-checkpoint reproduction claim is made for any listed
checkpoint.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| CGL unconditional | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | S0-S4 PASS; S4 16 batches / 512 samples; first divergence `null` | S1 loss max_abs `1.430511474609375e-06`; S2 raw gradient max_abs `2.1886080503463745e-08` | `.cache/ralf/training-reproduction/cgl/` |
| CGL label | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | CGL staged evidence is unconditional only | Not run for this checkpoint | `.cache/ralf/training-reproduction/cgl/` |
| CGL label-size | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | CGL staged evidence is unconditional only | Not run for this checkpoint | `.cache/ralf/training-reproduction/cgl/` |
| CGL completion | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | CGL staged evidence is unconditional only | Not run for this checkpoint | `.cache/ralf/training-reproduction/cgl/` |
| CGL refinement | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | CGL staged evidence is unconditional only | Not run for this checkpoint | `.cache/ralf/training-reproduction/cgl/` |
| CGL relation | package/vendor staged path | `not-yet-run (S5 pending; tracked in issue #44)` | `training-seed n=1` | CGL staged evidence is unconditional only | Not run for this checkpoint | `.cache/ralf/training-reproduction/cgl/` |
| PKU unconditional | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |
| PKU label | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |
| PKU label-size | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |
| PKU completion | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |
| PKU refinement | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |
| PKU relation | package | `not-yet-run (staged evidence pending; tracked in issue #44)` | Not run | No staged evidence | Not run | `.cache/ralf/training-reproduction/cgl/` |

The S3 evidence follows the general S3 Evidence Layers rule. Natural runs
009/010 recorded run-to-run warn-only drift and exited the existing S0-S2
contract, so synchronized run-012 restored state at optimizer boundaries and
compared forward values, raw/clipped gradients, optimizer state, parameters,
and learning rates. The synchronized layer is a bounded PASS with no tolerance
widening. The independent production-wiring representative recorded 210
optimizer steps, crossed the epoch-49 `MultiStepLR` milestone, delivered train
and validation logging for 70/70 epochs, and exited the documented `traingen fit`
command with code 0. Its synchronized maxima were raw gradient
`5.587935447692871e-08`, clipped gradient `4.6566128730773926e-09`, parameters
`4.76837158203125e-07`, and optimizer state `4.656612873077393e-10`, with no
tolerance widening.

S4 run-010 compared the actual package `RalfDataModule.train_dataloader()` and
`val_dataloader()` with the vendor `DataLoader` and `collate_fn`. Train and
validation split membership matched exactly (48,544 and 6,002 ids,
respectively), with zero overlap. The package and vendor canonical stream
digest was the same: `664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876`.
The loader digests were package
`2ef173685465a5810cfcaed76f2788cba85c2f48c85b5a83229cf004be1b4f3e` and vendor
`5970e735922f26706018496a90f4bf9cda40c6dea1dbde0f4e980b4446f92ab7`.

## Regeneration Metadata

The currently verified diagnostic environment is a pre-existing coherent
cu128 runtime supplied through an externally managed interpreter, with torch
`2.8.0+cu128`, on one selected Tesla V100-SXM2-32GB. The authoritative cache is
supplied read-only through `RALF_CACHE_DIR`; all accepted CGL evidence references
that environment variable. The candidate source digest for
S1-S4 is `a5fa52900f63553e644d5a02cb5cf6b2fc898559d926db909f4f31d195a35c8e`,
the effective config digest is
`102aea26067c5247d9995ad520e8b9bf22db47f5923d2c6099cffa00ab70389f`, and the
pinned vendor revision is `c51db6032acbd0bd0ce72433becce08317e7874d`.

The currently selected authoritative cache is the user-provided local copy,
selected only through `RALF_CACHE_DIR`; no shared-mount dependency is part of
the rerun contract. Its content-role inventory is the CGL/PKU dataset trees,
retrieval-index tables, precomputed ResNet/FIDNet/DreamSim resources,
relationship and saliency data, evaluation features, and training logs. The
per-file SHA256 manifest is
`.cache/ralf/training-reproduction/cgl/provenance/local-cache-manifest-001.sha256`
(404 files; manifest SHA256
`17ce2b93df63ca8444b4be465f8b72d8a2ea9d2e480d98bb60f98de5a8150a96`). The
manifest is generated from the authoritative local copy and is not committed.

S4 run-010 evidence:

```text
artifact: .cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json
artifact_sha256: 4c989f7e87444d2275d775c06fadcdcfbcff36acf18caca013f6ec156a7106c8
stdout: .cache/ralf/training-reproduction/cgl/s4/runs/run-010/stdout.log
stdout_sha256: c762713e38bdfff432a360550b7f96f2f9694af41bc9584cc2c4b117df36a349
package_canonical_stream_sha256: 664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876
vendor_canonical_stream_sha256: 664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876
package_loader_sha256: 2ef173685465a5810cfcaed76f2788cba85c2f48c85b5a83229cf004be1b4f3e
vendor_loader_sha256: 5970e735922f26706018496a90f4bf9cda40c6dea1dbde0f4e980b4446f92ab7
train_ids_sha256: 1bff3728181120eb8a1daae365a99a26788103501cb3835fc1360cf116307aed
validation_ids_sha256: 56aee7f12c369c9336e04665dc434cdf1b05455d6525e68d2a74584fefbdcdf2
```

S3 run-012 evidence remains under
`.cache/ralf/training-reproduction/cgl/s3/runs/run-012/`:

```text
trace_sha256: d29f0c259eaacad7affa8cffd9b2e55942fdf54a25be19afcc2d28af0ca44094
stdout_sha256: 45aa1a4122261d3608f19800a0c051ae7b277650a387af4c5c90231860504419
checkpoint_sha256: 8b5c76c0f105c25b3b061d77c1136ddc0a0ce90884cdfb3257126d72756064af
runtime_seconds: 3807.7891788799316
```

## Training Commands

Set the diagnostic paths explicitly before rerunning a staged check. The
commands use one GPU and fail closed when local parity assets are unavailable.

```bash
: "${RALF_AUDIT_PYTHON:?set RALF_AUDIT_PYTHON to the verified diagnostic interpreter}"
: "${RALF_CACHE_DIR:?set RALF_CACHE_DIR to the authoritative read-only cache}"
export RALF_PYTHONPATH="$PWD/models/ralf/src:$PWD/models/ralf/tests/vendor_parity:$PWD/lib/laygen/src:$PWD/lib/posgen/src:$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src"
```

Run the focused package training checks.

```bash
PARITY_REQUIRE=1 PYTHONPATH="$RALF_PYTHONPATH" \
  "$RALF_AUDIT_PYTHON" -m pytest \
  models/ralf/tests/test_training.py \
  models/ralf/tests/vendor_parity/test_training_harness.py -q
```

Regenerate CGL S0-S2 into new `.cache` directories rather than overwriting
accepted artifacts.

```bash
CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 PYTHONPATH="$RALF_PYTHONPATH" \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$RALF_AUDIT_PYTHON" \
  models/ralf/tests/vendor_parity/run_training_stages.py \
  --stage S1 --dataset cgl --cache-dir "$RALF_CACHE_DIR" \
  --output .cache/ralf/training-reproduction/cgl/<new-run>/s1.json \
  --steps 1 --batch-size 32 --seed 1
```

Use the same command with `--stage S2` for the one-step optimizer check.

The accepted S3 production command is the member-scoped LightningCLI command
invoked by the parity runner. Its effective boundary is:

```bash
CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 \
  uv run --active --no-sync --package ralf --extra training --extra vendor \
  traingen fit \
  --config models/ralf/configs/training/cgl.yaml \
  --seed_everything=1 --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.max_epochs=70 --trainer.deterministic=warn \
  --trainer.accumulate_grad_batches=1 --data.init_args.batch_size=32
```

The full run-specific overrides, callback/logger wiring, and cache paths are
recorded in the run-012 trace metadata. Do not substitute a package model with
the original model in the package Trainer.

Run the accepted CGL S4 loader-stream check into a fresh artifact directory.

```bash
CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 PYTHONPATH="$RALF_PYTHONPATH" \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$RALF_AUDIT_PYTHON" \
  models/ralf/tests/vendor_parity/run_training_stages.py \
  --stage S4 --dataset cgl --cache-dir "$RALF_CACHE_DIR" \
  --output .cache/ralf/training-reproduction/cgl/s4/runs/<new-run>/s4.json \
  --steps 8 --batch-size 32 --seed 1
```

S5 full-run training, evaluation, and the 300-step real-scale lockstep probe
are not started in this checkpoint.
