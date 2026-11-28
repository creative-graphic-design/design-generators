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
- Participants and direct peer recipients: `<2–6 target slugs with role names `impl-plan-<target-slug>`, agent names, and pane ids>`.
- Coordinator: `<coordinator-agent-name>` on pane `<coordinator-pane-id>`.
- Chair: `<agent name>`.
- Assigned worktree and tab: `<paths/ids from the coordinator's ledger>`.
- Writable draft: `<shared-drafts-dir>/<target-slug>.md`.
- Chair's unified specification: `<shared-drafts-dir>/unified-interface.md`.

Follow the read-only, assigned-worktree, and draft-writing constraints in the council skill.

## Acceptance

<target-specific acceptance criteria>
```
