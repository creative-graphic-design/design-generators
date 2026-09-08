# Participant task-message template

Send one direct message to each participant with the generic Herdr agent surface:

```text
herdr agent prompt <agent-name> "<message>"
```

Fill every `<...>` placeholder and remove sections that do not apply. Keep both round protocols intact.

```markdown
## Background

<One or two sentences describing the project>. Your target is **<target-slug>** (<issue or document pointer>). The council must agree on <concrete agenda summary> across <N> targets.

- Participants: <comma-separated target slugs and role names `impl-plan-<target-slug>`>
- Coordinator: `<coordinator-agent-name>` on pane `<coordinator-pane-id>`; use these recorded values for reports.
- Worktree: your assigned Herdr-managed worktree and tab; the coordinator recorded your agent name and pane ID in the participant ledger.
- Sources: read <source checkout absolute path> and the named issue or documents read-only.
- Writable location: only the shared drafts directory `<shared-drafts-dir>/`. Do not write to any repository.

## Round 1: independent proposal

1. Read <umbrella issue or document> and <target issue or document> first.
2. Investigate <target-specific sources and agenda questions>.
3. Write `<shared-drafts-dir>/<target-slug>.md` with evidence, file-level plan, open questions, and an `## Interface proposal` section covering:
   - <agenda item 1>
   - <agenda item 2>
   - <agenda item 3>
4. Send a proposal summary of 15 lines or fewer directly to every peer by their recorded agent name using `herdr agent prompt`.
5. Send `Round 1 complete` and the draft path directly to `<coordinator-agent-name-or-pane-id>` using `herdr agent prompt`, then end your turn.

## Round 2: debate and convergence

1. Review all peer proposals delivered through the agent prompt surface.
2. Debate each disagreement directly with the relevant peer using `herdr agent prompt`; do not route peer discussion through the coordinator.
3. Update your plan to the agreed version. Record target-specific exceptions as `Deviation` entries with reasons.
4. <If you are the chair: write the unified spec at `<shared-drafts-dir>/unified-interface.md` with every agenda decision, each `Deviation`, and unresolved questions.>
5. Send `Round 2 complete` to `<coordinator-agent-name-or-pane-id>` with the agreements and remaining disagreements, then end your turn.

## Boundaries

- This council has exactly two rounds. Leave unresolved questions explicit; do not add another round implicitly.
- Stay in the assigned Herdr-managed worktree and tab. Do not create alternate worktrees or transports.
- Write only the plan or chair specification in `<shared-drafts-dir>/`; do not modify repository files, commit, push, publish, or implement.
- The coordinator reconciles progress with bounded `herdr agent get` and `herdr agent read` checks, followed by `herdr agent wait` only when a finite wait is needed. Get/read are immediate; only wait uses a timeout. Preserve evidence when a participant is unresolved.
- `herdr notification show` is a UI fallback for the coordinator/user when a participant is blocked or timed out; it does not send a participant message. Do not treat silence as completion.
- Retain worktrees, tabs, branches, drafts, and outputs unless the user gives explicit authorization naming cleanup targets and scope.

## Completion

- Round 1: draft written, proposal sent directly to every peer, and completion reported to `<coordinator-agent-name-or-pane-id>`.
- Round 2: agreed plan written, chair specification written when applicable, and agreements or remaining disagreements reported to `<coordinator-agent-name-or-pane-id>`.
```
