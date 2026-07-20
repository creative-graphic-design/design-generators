---
name: parallel-plan-discussion
description: Orchestrate multiple disposable Codex agents (via delegate-codex + agmsg) that investigate or plan different targets in parallel and then converge on shared decisions through structured discussion rounds — Round 1 independent proposals broadcast to peers, Round 2 debate and convergence, with a designated chair producing a unified spec. Use whenever the user wants several things investigated/planned at once AND the results need to be consistent with each other — e.g. 「モデルごとにエージェントを立てて話し合って」「それぞれ調べてインターフェースを統一して」「並列で計画を立てて議論して」, aligning APIs/schemas/conventions across models, services, modules, or migration targets. Requires HERDR_ENV=1, herdr, agmsg, and the delegate-codex skill.
---

# Parallel Plan Discussion

## Overview

Spawn one disposable Codex agent per target, have each produce an investigation-grounded plan, then make the agents discuss with each other over agmsg until their proposals converge. A designated chair consolidates the agreement into a unified spec. The coordinator (`main`) designs the council, relays rounds, reviews the outcome, and publishes it.

Use this instead of a plain delegate-codex fan-out when the outputs must agree with each other (shared output schema, common API signature, naming conventions, shared utility extraction). Independent agents working in parallel produce individually-good but mutually-inconsistent plans; the discussion rounds are what buy the consistency. If the targets don't need to agree on anything, plain delegation is cheaper — skip this skill.

This skill defines only the council protocol. All spawn/trust/nudge/monitor/cleanup mechanics come from the delegate-codex skill: re-read `~/.claude/skills/delegate-codex/SKILL.md` (and `~/.agents/AGENTS-private.md` for environment specifics) at the start of every council, and follow it for those steps. Do not reproduce those procedures from memory here.

## Roles

- Coordinator: `main` (you). Designs the council, sends tasks, relays rounds, reviews, publishes, cleans up. Never writes the plans itself.
- Participants: `impl-plan-<target-slug>`, one per target, each in its own worktree and Herdr tab. Read-only with respect to all repositories; they write only into the shared drafts directory.
- Chair: exactly one participant (pick the target with the most prior context, or the first-mover target). In the final round the chair writes the unified spec consolidating everyone's agreement. Tell the chair it is the chair in its task message; the others don't need to know.

## Step 1: Design the council

Decide these before spawning anything, and state them in a short user-facing update:

- Targets: 2–6. Pick for diversity of constraints, not volume — the discussion is only informative if the participants have genuinely different needs (e.g. a GAN, a diffusion model, and a seq2seq model arguing about one output schema). More than ~6 makes the peer-to-peer message volume quadratic and the discussion shallow.
- Agenda: the explicit list of things that must converge (output schema, API signature, naming, shared-code candidates, ...). Vague agendas produce vague consensus; every agenda item should be answerable with a concrete decision.
- Deliverables: per-target plan file, the unified spec, and any cross-target comparison tables. Fix the shared drafts directory now — use the session scratchpad (e.g. `<scratchpad>/impl-plans/`), never a repository.
- Rounds: default two. Round 1 = investigate + draft + broadcast proposal. Round 2 = read peers, debate, converge, chair consolidates. Add a Round 3 only if the Round 2 reports show unresolved disagreements on agenda items (not mere model-specific deviations — those are expected and get recorded, not argued away).

## Step 2: Set up and spawn

Follow delegate-codex for the mechanics. Council-specific notes:

- One worktree per participant (`gwq add -b plan-<slug>` from the main checkout). Run `gwq get` from inside the repository — it resolves against the current repo, so calling it after `cd`-ing elsewhere fails; capture explicit paths early and reuse them.
- Worktrees do not contain submodule contents. If targets live in submodules, instruct participants to read sources from the main checkout's absolute path, read-only.
- Add the Codex trust entry for every worktree path before spawning (see AGENTS-private), spawn all participants in one pass, and record the pane ids the spawn wrapper prints — you need them for nudges and waits.

## Step 3: Send task messages

Send each participant one agmsg task message built from `references/task-message-template.md` (read it now). The template encodes what made this work:

- Per-target context: issue/doc links to read first, source paths, known facts, and the target's expected implementation direction.
- Round 1 protocol: investigate → write plan draft to the shared directory → write an `## Interface proposal` section → send a compact summary (≤15 lines) of the proposal to every peer by name → report completion to `main` → end the turn. Ending the turn is what lets you relay rounds; say it explicitly.
- Round 2 protocol: read the inbox → argue disagreements directly with the relevant peer (participants message each other, not through you) → update the plan to the agreed version, recording target-specific deviations as explicit `Deviation` entries with reasons → report agreements/remaining disagreements to `main` → end the turn.
- The chair's extra duty: consolidate everything into the unified spec file in Round 2.
- Boundaries: repository writes, commits, pushes, and PRs are forbidden; only the shared drafts directory is writable. No merge/publish authority — that stays with `main` and the user.

After sending, nudge every participant to start Round 1 (nudge rules per delegate-codex).

## Step 4: Relay the rounds

- Wait for all panes with a background loop over `wait-pane.sh` so you're re-invoked when the round finishes; don't poll in the foreground.
- Leave a grace period between nudging and waiting: a just-nudged pane stays `idle` for a few seconds before the agent picks the message up, so an immediate `wait-pane.sh` returns instantly and you'll conclude the round finished when it never started. Sleep ~20s after the nudges (or confirm the panes show `working`) before starting the wait, and treat "wait returned but no new reports/files" as this race, not as agent failure — re-wait instead of re-nudging.
- Between rounds: check `main`'s inbox for the round reports, confirm the draft files actually exist, run the occupancy check on each pane, then nudge everyone into the next round with a message that names what to process (peer proposals, any addenda).
- Mid-flight requirement changes (the user adds "also compare training datasets"): send an addendum via agmsg to all participants immediately. Messages queue in inboxes and get processed at the next round — folding new requirements into the running council is far cheaper than restarting it. Tell the next-round nudge to include the addendum.
- Never read a participant's inbox from `main` — inbox reads consume messages meant for them. Judge progress from their reports and the draft files.

## Step 5: Review and publish

Read the unified spec and each plan before publishing anything. The review bar:

- Every agenda item is either a concrete decision or an explicitly-listed open question — nothing silently dropped.
- Target-specific deviations are listed with reasons, not blended into the consensus.
- Spot-check one or two load-bearing claims against the actual sources (the same way you'd verify a subagent's report).

Publish per the user's conventions (e.g. unified spec as a comment on the umbrella issue, each plan as a comment on its target's issue). Publishing to GitHub is outward-facing — follow whatever approval pattern the user has established.

## Step 6: Clean up

Per delegate-codex: close each participant's Herdr tab and reset its agmsg registration. Additionally remove the disposable worktrees and branches (they were read-only anchors; nothing in them is worth keeping). Verify with `git worktree list` that only the main checkout remains.

## Pitfalls (all observed in real runs)

- `gwq get <branch>` outside the repository fails even when the worktree exists — use explicit `<repo-path>=<branch>` paths after creation.
- A nudge to a working pane exits with code 3; wait for idle/done and retry. This bites in two places: right after spawn (the pane is still booting) and mid-round. The task message is already in the inbox either way, so a delayed nudge loses nothing — but a skipped first nudge means the participant never starts, so check nudge exit codes instead of assuming delivery. Duplicate nudge output lines are normal.
- Participants that don't end their turn after a round can't be relayed — the round protocol in the task message must tell them to report and stop.
- If a participant's plan must cover data specifics (label vocabularies, coordinate systems), say so explicitly in the agenda; "investigate the data handling" alone yields prose, not comparable tables.
- Keep proposals compact (≤15 lines) in peer messages; full detail lives in the draft files. Peers read N-1 proposals each — long messages blow up their context.
