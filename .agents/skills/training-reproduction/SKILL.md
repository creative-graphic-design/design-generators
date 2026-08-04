---
name: training-reproduction
description: Use this skill whenever implementing, reviewing, documenting, or planning package-local training reproduction in design-generators. It enforces the S0-S5 order from docs/training-reproduction.md, requires evidence comments per stage, and blocks S5 claims or S5-scale GPU runs when S0-S4 evidence is missing, even if the user only asks for training, train-ourselves work, TRAINING.md updates, or reproduction evidence.
---

# Training Reproduction

## Source Of Truth

Read `docs/training-reproduction.md` before starting training-reproduction work.
That protocol defines stage scope, dataset coverage, seed policy, GPU placement,
evidence recording, and PR gates. This skill only turns the protocol into an
execution checklist for coding agents.

## Required Order

Work in stage order: S0, S1, S2, S3, S4, then S5. Do not skip ahead because S5 is
the visible deliverable. S0-S2 localize model, loss, optimizer, and reference
adapter differences; S3-S4 localize repeated training and data-stream
differences. Without those records, an S5 mismatch is not diagnosable.

Use this order for every training-first package:

1. Build or update the vendor reference adapter.
2. Produce S0 static config/topology evidence.
3. Produce S1 fixed-batch pre-optimizer trace evidence.
4. Produce S2 one-step optimizer evidence.
5. Post an issue comment summarizing the reference adapter plus S0-S2 evidence.
6. Produce S3 deterministic multi-batch evidence.
7. Produce S4 deterministic loader-stream evidence.
8. Post or update issue evidence for S3-S4.
9. Only then start S5-scale GPU training and full-run evaluation.
10. Record final S0-S5 evidence in `models/<package>/TRAINING.md`.

## S5 Gate

Do not launch S5-scale GPU jobs, mark an issue as parity-verified, or write a
README/model-card/PR claim that S5 reproduction is complete unless S0-S4 evidence
already exists and is cited. If earlier evidence is missing, stop at the current
stage and document the blocker instead of using S5 as a substitute.

The durable package document must include a machine-readable `Stage Evidence`
table in `models/<package>/TRAINING.md`:

```markdown
## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S1 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S2 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S3 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S4 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S5 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
```

Run `uv run --package design-generators python scripts/check_training_stage_evidence.py`
before opening or updating a PR that touches training reproduction docs.

## Evidence Rules

- Commit commands, seeds, config names, metric summaries, issue-comment URLs, and
  repository/cache-relative artifact paths. The checker also accepts project
  issue and PR URLs in the creative-graphic-design/design-generators GitHub
  repository when evidence already lives in repository discussion.
- Do not commit generated tensors, checkpoints, images, downloaded datasets, or
  full-run artifacts.
- Use one explicitly selected GPU for CUDA parity or training runs.
- Label seed scope exactly, such as `training-seed n=3` or
  `evaluation-seed n=3`.
- State dataset coverage per dataset; do not imply full reproduction when only a
  subset has S5 evidence.

## PR Rules

Keep train-ourselves PRs draft until S5 is complete for every claimed dataset.
If a PR intentionally lands S0-S4 infrastructure before full runs complete, the
PR body, README, and `TRAINING.md` must state that S5 trained-checkpoint
reproduction is not yet claimed.
