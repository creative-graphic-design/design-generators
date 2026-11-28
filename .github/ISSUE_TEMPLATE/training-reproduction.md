---
name: Training reproduction
about: Reproduce a model package's training and record each stage's result.
title: "[<package>] Reproduce package-local training (S0-S5)"
labels: train-ourselves
assignees: ""
---

## Goal

Reproduce package-local training for `<package>` using the staged [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#stage-overview).

## Method

- Vendor stack mode: `<Lightning | accelerate | plain PyTorch>`.
- Reference adapter location: `<models/<package>/tests/vendor_parity/...>`.
- Package training entrypoint: `<traingen fit ...>`.
- Original implementation training entrypoint: `<vendor command or script>`.

## Scope

List the stages and dataset coverage in this issue. Use the stage definitions and order in the canonical [protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#stage-overview).

- Stages and coverage boundaries: `<...>`.

## Datasets

- `<dataset>`: `<source, config, split, and any processed-stream requirement>`.

## Seed Policy

- Training-seed scope per dataset: `<...>`.
- Evaluation-seed scope per dataset: `<...>`.
- Follow the canonical [Seed Policy](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#seed-policy) when labeling evidence.

## Deliverables

- Reference adapter and S0-S2 evidence: `<path and issue-comment link>`.
- Package-local `LightningModule`, datamodule, configs, deterministic seed controls, and parity helpers.
- S3/S4 evidence: `<artifact paths and issue-comment link>`.
- S5 original/package metrics, seed scope, and artifact locations: `<...>`.
- `TRAINING.md` with a `Stage Evidence` table covering S0-S5.
- README/model-card updates that describe only the datasets and seed scope with completed evidence.

Follow the [training reproduction skill](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-training-reproduction/SKILL.md) for the source-pinned seed-queue guard and fail-closed parity procedure.

## Tracking

- Read the [project roadmap](https://github.com/creative-graphic-design/design-generators/blob/main/docs/roadmap.md) and [shared data sources](https://github.com/creative-graphic-design/design-generators/blob/main/docs/data-sources.md); follow the [implementation checklist](https://github.com/creative-graphic-design/design-generators/blob/main/docs/implementation-checklist.md).
- Follow the protocol's [Pull Request Gates](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#pull-request-gates) section for implementation PR requirements.
