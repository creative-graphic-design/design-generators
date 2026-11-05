---
name: design-generators-training-reproduction
description: Use this skill whenever implementing, reviewing, documenting, or planning package-local training reproduction in design-generators. It enforces the six ordered stages and S5 gate defined in docs/training-reproduction.md, requires evidence comments for each stage, and blocks unsupported S5 claims, S5-scale training or evaluation launches, parity-verified label changes, and S5 claims in README files or model cards, even if the user only asks for training, models whose weights this repository trains itself (called "train-ourselves" work), TRAINING.md updates, or reproduction evidence.
---

# Training Reproduction

## Source Of Truth

Read `docs/training-reproduction.md` before starting training-reproduction work. That protocol defines stage scope, dataset coverage, seed policy, GPU placement, evidence recording, and PR gates. This skill only turns the protocol into an execution checklist for coding agents.

For exact, p-value, and rounding wording in `TRAINING.md`, follow the claim-strength rules in the documentation skill.

## Required Order

Work in stage order: S0, S1, S2, S3, S4, then S5. S0-S2 localize model, loss, optimizer, and reference adapter differences; S3-S4 localize repeated training and data-stream differences.

Use this order for every training-first package:

1. Build or update the reference adapter for the original implementation, called the vendor reference adapter here.
2. Produce S0 static config/topology evidence.
3. Produce S1 fixed-batch pre-optimizer trace evidence.
4. Produce S2 one-step optimizer evidence.
5. Post an issue comment summarizing the reference adapter plus S0-S2 evidence.
6. Produce S3 deterministic multi-batch evidence.
7. Produce S4 deterministic loader-stream evidence.
8. Post or update issue evidence for S3-S4.
9. Start S5-scale GPU training and full-run evaluation after the GPU-bound step-loop check in the protocol's GPU placement section.
10. Record final S0-S5 evidence in `models/<package>/TRAINING.md`, including the inapplicable-rule notes and amendment citations required by the protocol's [Stage Rules](docs/training-reproduction.md#stage-rules); issues and pull requests may quote them.

## S5 Gate

Before an S5-scale training or evaluation launch, a `parity-verified` label change, or an S5 claim in a README or model card, follow the [checkpoint smoke rule](docs/training-reproduction.md#checkpoint-smoke-before-full-runs).

Before S5, complete the production-path loss-envelope comparison in [S3 Evidence Recording](docs/training-reproduction.md#s3-evidence-recording): use at least 1,500 updates and three matched training seeds, independent production initialization and optimizer stepping, and the S4 stream; gate the median of package per-seed medians against the widened envelope from original production runs and cite the artifact in the S3 Stage Evidence row. A failure calls for diagnosis or more original production seeds; it does not by itself prove a defect.

Before launching S5, write one launch manifest with `source_commit`, `config`, `evaluator_command`, `checkpoint_rule`, `artifacts` (each artifact path mapped to its SHA-256), and `launched_at`; cite its repository- or cache-relative path in the S5 Stage Evidence row. The [protocol](docs/training-reproduction.md#checkpoint-smoke-before-full-runs) defines the required checked-in configuration. In the S5 Comparison Scope table in `TRAINING.md`, the `Evaluator` cell must equal the manifest's `evaluator_command`, and the `Checkpoint-selection rule` cell must equal `checkpoint_rule`.

Before S5, satisfy the protocol's [evaluation-path parity prerequisite](docs/training-reproduction.md#evaluation-path-parity-prerequisite) and cite its artifact beside the launch manifest in the S5 Stage Evidence row.

Finish every commit the campaign intends to make, including documentation commits, before launching the seed queue (the supervisor that trains and evaluates the per-seed runs in sequence) because the queue pins the source commit once at startup. Before launch, run a deliberate mismatch dry run for the source gate (each run's check that the worktree commit equals the commit pinned at queue start) and verify that the queue stops on the gate failure; a nonzero gate exit swallowed by an `&&` chain or a function body does not stop the queue.

The package's `TRAINING.md` lists each reproduction check's command, output artifact, and result so another person can rerun it; use the current table format in [the training template](docs/templates/TRAINING.template.md).

Run `uv run --package design-generators scripts/check_training_stage_evidence.py` before opening or updating a PR that touches training reproduction docs.

## Evidence Rules

- Commit commands, seeds, config names, metric summaries, issue-comment URLs, and repository/cache-relative artifact paths. The checker also accepts project issue and PR URLs in the creative-graphic-design/design-generators GitHub repository when evidence already lives in repository discussion.
- Do not commit generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts.
- Use one explicitly selected GPU for CUDA parity or training runs.
- Label seed scope exactly, such as `training-seed n=3` or `evaluation-seed n=3`.
- State dataset coverage per dataset; do not imply full reproduction when only a subset has S5 evidence.

## PR Rules

Keep train-ourselves PRs draft until S5 is complete for every claimed dataset. If a PR intentionally lands S0-S4 infrastructure before full runs complete, the PR body, README, and `TRAINING.md` must state that S5 trained-checkpoint reproduction is not yet claimed.

## Repository Training Rules

- Train-ourselves models use PyTorch Lightning through each model package's `training` extra and LightningCLI with YAML configs plus CLI overrides.
- Keep `LightningModule`, `LightningDataModule`, and `configs/*.yaml` inside the model package.
- Launch training through the `traingen` console script with the model member and training extra selected: `uv run --package <model> --extra training traingen fit --config models/<model>/configs/training/<config>.yaml`.
- A root-level `traingen` launch is not the supported model-training workflow; select the member environment above.
- Training-first packages follow the canonical [training reproduction protocol](docs/training-reproduction.md) for S0-S5 evidence, topology guards, dataset coverage, seed policy, and evidence recording.
- Package `TRAINING.md` files must follow [docs/templates/TRAINING.template.md](docs/templates/TRAINING.template.md) and pass `scripts/check_training_doc_template.py`.
- PRs for models whose only weight path is self-training stay draft until S5 is confirmed for every claimed dataset; partial coverage must be stated in the package `TRAINING.md`, README, and PR body.

## Comparison Scope Rules

Add the following subsection after the `Reproduction Results` table. Comparison Scope records the evaluation setup for each dataset comparison. Use one row with `System` set to `both` when the evaluator, test split, checkpoint-selection rule, and sample count apply to both systems. Use separate `package` and `original` rows when any of these values differs. The four scope fields state the evaluator used, the evaluated test split, the rule that selects the checkpoint entering evaluation, and the sample count. Sample count is the metric denominator for the reported metrics.

### Comparison Scope

| Dataset     | System | Evaluator     | Test split | Checkpoint-selection rule | Sample count                      |
| ----------- | ------ | ------------- | ---------- | ------------------------- | --------------------------------- |
| `<dataset>` | both   | `<evaluator>` | `<split>`  | `<rule>`                  | `<N> layouts per evaluation seed` |
