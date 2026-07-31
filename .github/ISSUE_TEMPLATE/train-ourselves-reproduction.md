---
name: Train-ourselves reproduction
about: Track package-local S0-S5 training reproduction work.
title: "[<package>] Reproduce package-local training (S0-S5)"
labels: train-ourselves
assignees: ""
---

## Goal

Reproduce package-local training for `<package>` so trained checkpoints can be
claimed only after the staged S0-S5 protocol in `docs/training-reproduction.md`
has durable evidence.

## Method

- Vendor stack mode: `<Lightning | accelerate | plain PyTorch>`.
- Reference adapter location: `<models/<package>/tests/vendor_parity/...>`.
- Package training entrypoint: `<traingen fit ...>`.
- Original implementation training entrypoint: `<vendor command or script>`.

## Scope (S0-S5)

- S0: Static config, initialized state, topology guard, optimizer defaults, and
  dataset encoding.
- S1: Fixed-batch pre-optimizer trace.
- S2: One optimizer step.
- S3: Short deterministic multi-batch run.
- S4: Deterministic loader stream.
- S5: Full-run statistical comparison under the original evaluation protocol.

## Datasets

- `<dataset>`: `<source, config, split, and any processed-stream requirement>`.

## Seed Policy

- Target S5 evidence: training-seed n=3 unless this issue explicitly narrows the
  claim.
- Interim evidence: label evaluation-seed evidence as weaker than training-seed
  evidence.

## Deliverables

- Vendor reference adapter and S0-S2 trace-agreement evidence posted as an issue
  comment before any S5-scale GPU run starts.
- Package-local `LightningModule`, datamodule, configs, deterministic seed
  controls, and parity helpers.
- S3 short-run and S4 loader-stream evidence posted before S5 is claimed.
- S5 full-run original/package metrics, seed scope, and artifact locations.
- `TRAINING.md` with a `Stage Evidence` table covering S0-S5.
- README/model-card updates that describe only the datasets and seed scope with
  completed evidence.

Do not launch S5-scale GPU training or claim S5 reproduction before the issue
comment containing the vendor reference adapter plus S0-S2 evidence exists.

## Tracking

- Refs #2, #60.
- Implementation PR stays draft until S5 is confirmed for every claimed dataset.
