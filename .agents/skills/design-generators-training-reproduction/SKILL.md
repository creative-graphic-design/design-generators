---
name: design-generators-training-reproduction
description: Use this skill whenever implementing, reviewing, documenting, or planning package-local training reproduction in design-generators. It enforces the six ordered stages and S5 gate defined in docs/training-reproduction.md, requires evidence comments for each stage, and blocks unsupported S5 claims or S5-scale GPU runs, even if the user only asks for training, models whose weights this repository trains itself (called "train-ourselves" work), TRAINING.md updates, or reproduction evidence.
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

Apply the [protocol's Stage Overview](docs/training-reproduction.md#stage-overview) before launching S5-scale GPU jobs, marking an issue with the `parity-verified` status label, or writing a README/model-card/PR claim that S5 reproduction is complete.

Before launching S5, write one launch manifest: a JSON file with `source_commit`, `config` (a resolved path or inline resolved configuration), `evaluator_command` (the evaluator command and flags), `checkpoint_rule`, `artifacts` (each artifact path mapped to its SHA-256), and `launched_at`; cite its repository- or cache-relative path in the S5 Stage Evidence row. The Comparison Scope `Evaluator` cell names the evaluator that the manifest's `evaluator_command` runs, and the `Checkpoint-selection rule` cell equals the manifest's `checkpoint_rule`.

Finish every commit the campaign intends to make, including documentation commits, before launching the seed queue (the supervisor that trains and evaluates the per-seed runs in sequence) because the queue pins the source commit once at startup. Before launch, run a deliberate mismatch dry run for the source gate (each run's check that the worktree commit equals the commit pinned at queue start) and verify that the queue stops on the gate failure; a nonzero gate exit swallowed by an `&&` chain or a function body does not stop the queue.

The durable package document must include a machine-readable `Stage Evidence` table in `models/<package>/TRAINING.md`:

```markdown
## Stage Evidence

| Stage | Command     | Artifact                                             | Result     |
| ----- | ----------- | ---------------------------------------------------- | ---------- |
| S0    | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S1    | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S2    | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S3    | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S4    | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S5    | `<command>` | `.cache/<package>/full-run/<dataset>/manifest.json`  | `<result>` |
```

Run `uv run --package design-generators scripts/check_training_stage_evidence.py` before opening or updating a PR that touches training reproduction docs.

## Evidence Rules

- Commit commands, seeds, config names, metric summaries, issue-comment URLs, and repository/cache-relative artifact paths. The checker also accepts project issue and PR URLs in the creative-graphic-design/design-generators GitHub repository when evidence already lives in repository discussion.
- Do not commit generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts.
- Use one explicitly selected GPU for CUDA parity or training runs.
- Label seed scope exactly, such as `training-seed n=3` or `evaluation-seed n=3`.
- State dataset coverage per dataset; do not imply full reproduction when only a subset has S5 evidence.

## PR Rules

Keep train-ourselves PRs draft until S5 is complete for every claimed dataset. If a PR intentionally lands S0-S4 infrastructure before full runs complete, the PR body, README, and `TRAINING.md` must state that S5 trained-checkpoint reproduction is not yet claimed.

## Repository Training Contract

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
