---
name: parallel-plan-discussion
description: Orchestrate a two-round council of Herdr-managed agents that investigate distinct targets, debate shared interface decisions directly, and have one chair consolidate the result into a unified specification. Use when several plans must agree on a shared design.
---

# Parallel Plan Discussion

Use this protocol when independent investigations must converge on shared decisions. The protocol owns council policy only. The installed Herdr binary and the generic Herdr skill own all CLI syntax and pane, tab, worktree, agent, and notification mechanics.

Before any orchestration command, verify `HERDR_ENV=1` and inspect the current installed guidance with `herdr --skill`. If this skill and the installed guidance differ, follow the installed guidance. Do not copy a Herdr command manual into this skill.

## Council contract

- Choose 2–6 targets, a concrete agenda, a shared drafts directory outside the repository, and exactly two rounds.
- Assign one Herdr-recognized agent per target. Every participant must use a separate Herdr-managed worktree and Herdr tab.
- Record a participant ledger before sending work. It must include the actual coordinator agent name and coordinator pane ID, plus role, agent names, pane IDs, tab IDs, worktree paths, and chair designation.
- The coordinator is the agent recorded in that ledger; it designs the council, relays rounds, reconciles progress, reviews artifacts, and publishes only when separately authorized. It does not write participant plans.
- Participants are read-only in repositories. They may write only their plan to the shared drafts directory. The chair additionally writes `unified-interface.md` there in Round 2.

## Round 1: independent proposals

1. Read the umbrella and target issue or document first, then investigate the target-specific sources named in the task.
2. Write `<shared-drafts-dir>/<target-slug>.md` with evidence, implementation boundaries, open questions, and an `## Interface proposal` section covering every agenda item.
3. Deliver a compact proposal (15 lines or fewer) directly to each peer with the generic Herdr agent surface: `herdr agent prompt <peer-agent-name> "<proposal>"`.
4. Report completion and the draft path directly to the recorded coordinator agent name or pane ID with `herdr agent prompt`, then end the turn so the coordinator can start Round 2.

## Round 2: direct debate and consolidation

1. Each participant reviews the peer proposals delivered through the agent prompt surface.
2. Participants debate disagreements directly with the relevant peer using `herdr agent prompt`; the coordinator is not a message relay.
3. Update each plan to the agreed proposal. Record target-specific exceptions as `Deviation` entries with reasons.
4. The chair updates the agreed `## Interface proposal` and writes `<shared-drafts-dir>/unified-interface.md`, including decisions for every agenda item, deviations, and unresolved questions.
5. Every participant reports agreements and remaining disagreements directly to the recorded coordinator agent name or pane ID with `herdr agent prompt`, then ends the turn.

## Bounded reconciliation and fallback

For each participant and round, the coordinator uses a finite reconciliation budget (default: three passes). A pass may call immediate `herdr agent get` and `herdr agent read`; get and read have no timeout option, and only `herdr agent wait` uses a finite timeout. Do not add an open-ended loop, time-based retry, or alternate transport. After the budget is exhausted, record the participant as unresolved and use `herdr notification show` as a UI fallback for the coordinator/user; it does not send a participant message. Do not infer completion from silence.

The coordinator/user may use the same UI fallback when a prompt or wait reports a blocked participant. Preserve the ledger, the last readable output, and the unresolved reason so the user can decide what happens next.

## Review, retention, and authority

Review the unified specification and every target plan. Confirm every agenda item is a decision or an explicitly listed open question, and spot-check load-bearing claims against the source documents. Publishing or implementation is outside this protocol and remains with the coordinator and user.

Retain the participant worktrees, tabs, branches, ledger, drafts, and readable outputs by default. Cleanup is a separate action and requires explicit user authorization naming the resources and scope. Never close, remove, or delete council resources as an automatic completion step.
