# Participant assignment template

Use this template with the round protocol in `design-generators-parallel-plan-discussion`. Fill the task facts and remove non-applicable fields. Give each participant a readable path to the skill; do not duplicate the protocol in every assignment.

Send the completed assignment with `herdr agent prompt <agent-name> "<message>"`.

```markdown
## Task

Read `<checkout>/.agents/skills/design-generators-parallel-plan-discussion/SKILL.md` and follow its two-round protocol.

Investigate `<target>` and agree on `<shared interface decisions>` with the other participants. The intended reader of your plan is `<implementer or reviewer>`.

## Context and ownership

- Target issue/document and amendments: `<sources>`.
- Agenda: `<decisions every proposal must cover>`.
- Participants and direct peer recipients: `<target, agent name, pane id>`.
- Coordinator: `<coordinator-agent-name>` on pane `<coordinator-pane-id>`.
- Chair: `<agent name>`.
- Assigned worktree and tab: `<paths/ids from the coordinator's ledger>`.
- Writable draft: `<shared-drafts-dir>/<target-slug>.md`.
- Chair's unified specification: `<shared-drafts-dir>/unified-interface.md`.

You are not alone in the repository. Read sources without modifying them or another participant's draft. Write in only the shared drafts directory, to your assigned draft and, if you are chair, the unified specification.

## Acceptance

Cover each agenda item with evidence, an interface decision, or an explicit unresolved question. Record target-specific exceptions with reasons. Preserve the user's literal requirements and distinguish them from your proposals.

After Round 1, report the draft path directly to `<coordinator-agent-name-or-pane-id>` and end the turn. Start Round 2 only when the coordinator requests it. After Round 2, report agreements and remaining disagreements to the same recorded recipient and end the turn. The coordinator reviews the plans; publication and implementation require their own authorization.

If a participant is unresolved, the coordinator uses bounded reconciliation from the skill. `herdr notification show` is a UI fallback for the coordinator/user; it does not deliver a participant message.

Preserve drafts, outputs, tabs, worktrees, and the participant ledger. Do not clean up resources without user authorization for that scope.
```
