---
name: design-generators-training-reproduction
description: Use this skill whenever implementing, reviewing, documenting, or planning package-local training reproduction in design-generators. It enforces the six ordered stages and S5 gate defined in docs/training-reproduction.md, requires evidence comments for each stage, and blocks unsupported S5 claims, S5-scale training or evaluation launches, parity-verified label changes, and S5 claims in README files or model cards, even if the user only asks for training, models whose weights this repository trains itself (called "train-ourselves" work), TRAINING.md updates, or reproduction evidence.
---

# Training Reproduction

## Source Of Truth

Read `docs/training-reproduction.md` before starting training-reproduction work. That protocol defines stage scope, dataset coverage, seed policy, GPU placement, evidence recording, and PR gates. This skill only turns the protocol into an execution checklist for coding agents.

For exact, p-value, and rounding wording in `TRAINING.md`, follow the claim-strength rules in the documentation skill.

## Required Order

Complete stages in the order defined by the protocol. Use the following steps as the execution checklist.

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

Before S5, complete the production-path loss-envelope check in [S3 Evidence Recording](docs/training-reproduction.md#s3-evidence-recording).

### Gate Failure Triage

When any staged check or pre-launch gate fails, pause progression and classify the failure before proposing an action. Use captured inputs and measurements from the original and shipped package's real entrypoints, then independently apply the current protocol and any recorded amendment. Classify it as one of these:

- **Check-script defect:** the check measures incorrectly or applies the protocol criterion incorrectly to the real-entrypoint evidence. Correct it against those measurements, then rerun the failed check.
- **Protocol-rule defect:** the gate rejects an outcome that the rule's stated purpose accepts. Stop and report the defect with the evidence; do not work around the rule or change the run to satisfy it.
- **Package defect:** the measurements show that package behavior differs from the original's effective behavior in a way the protocol and recorded amendments do not permit.
- **Recorded permitted difference:** the measurements show a difference explicitly covered by an existing amendment on the model issue, and the observed case fits that amendment's scope. Cite the amendment and apply its criterion; proceed only if the remaining gate requirements pass.

Do not classify from intent, source-level expectations, or a plausible explanation. If the available measurements do not establish one of these cases, stop and collect evidence before proposing a fix or advancing.

Before implementing a fix, state the cause as a hypothesis, name a measurement that would refute it, and state the condition that will stop the change. Run the measurement through the real entrypoints first; if it refutes the hypothesis or does not isolate the cause, stop and revise the diagnosis before editing.

For a package defect, follow only the protocol path. Add a regression check and confirm it fails before the package fix. Fix the package behavior and confirm that check passes. Apply the protocol's [Regression Rule](docs/training-reproduction.md#regression-rule) by rerunning the evidence ladder from the lowest affected stage. After that ladder passes, rerun the failed gate.

If loss-envelope measurements show no check-script or package defect, collect more original production seeds per [S3 Evidence Recording](docs/training-reproduction.md#s3-evidence-recording).

Never offer continuing package training, waiving a gate, or reinterpreting a gate as alternatives. A deviation exists only through an amendment recorded as required by the protocol. Ask the user only about decisions the protocol assigns to them, such as additional GPU compute or an amendment. Never ask whether to follow the protocol.

Before S5, record the launch manifest required by the [protocol](docs/training-reproduction.md) and complete its [Comparison Scope](docs/training-reproduction.md#comparison-scope) table.

Before S5, satisfy the protocol's [evaluation-path parity prerequisite](docs/training-reproduction.md#evaluation-path-parity-prerequisite) and cite its artifact beside the launch manifest in the S5 Stage Evidence row.

Finish every commit the campaign intends to make, including documentation commits, before launching the seed queue (the supervisor that trains and evaluates the per-seed runs in sequence) because the queue pins the source commit once at startup. Before launch, run a deliberate mismatch dry run for the source gate (each run's check that the worktree commit equals the commit pinned at queue start) and verify that the queue stops on the gate failure; a nonzero gate exit swallowed by an `&&` chain or a function body does not stop the queue.

The package's `TRAINING.md` lists each reproduction check's command, output artifact, and result so another person can rerun it; use the current table format in [the training template](docs/templates/TRAINING.template.md).

Run `uv run --package design-generators scripts/check_training_stage_evidence.py` before opening or updating a PR that touches training reproduction docs.

## Evidence Rules

- Commit commands, seeds, config names, metric summaries, issue-comment URLs, and repository/cache-relative artifact paths. The checker also accepts project issue and PR URLs in the creative-graphic-design/design-generators GitHub repository when evidence already lives in repository discussion.
- Do not commit generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts.
- Follow the [protocol](docs/training-reproduction.md) for GPU placement, seed scope, and dataset coverage.
- For each recorded training behavior, state what the run did, how current code behaves, and the test that verifies the current behavior.

## PR Rules

Follow the protocol's [Pull Request Gates](docs/training-reproduction.md#pull-request-gates) for implementation PR status and claims.

## Repository Training Rules

- Train-ourselves models use PyTorch Lightning through each model package's `training` extra and LightningCLI with YAML configs plus CLI overrides.
- Keep `LightningModule`, `LightningDataModule`, and `configs/training/*.yaml` inside the model package.
- Launch training through the `traingen` console script with the model member and training extra selected: `uv run --package <model> --extra training traingen fit --config models/<model>/configs/training/<config>.yaml`.
- A root-level `traingen` launch is not the supported model-training workflow; select the member environment above.
- Training-first packages follow the canonical [training reproduction protocol](docs/training-reproduction.md) for staged evidence and claims.
- Package `TRAINING.md` files must follow [docs/templates/TRAINING.template.md](docs/templates/TRAINING.template.md) and pass `scripts/check_training_doc_template.py`.
