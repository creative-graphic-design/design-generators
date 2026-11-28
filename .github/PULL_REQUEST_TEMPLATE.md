## Summary

<!-- What changed and why. Include `Closes #N` or `Refs #N` for the implementation issue. The implementation checklist alone does not satisfy the PR metadata check. -->

- TODO

## Changes

- TODO

## Shared Library Changes

<!-- Required when touching cross-cutting paths such as `lib/*/src`, `docs/**`, or `.agents/skills/**`. Explain why the shared change belongs in this PR; otherwise write `N/A`. -->

- TODO

## Verification

<!-- Record commands and results, including passes and skips. For rendered-site inspection, follow the [documentation skill](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#verification) and list opened page URLs with the checks performed. -->

- TODO

## Checklist

Full checklist: see [docs/implementation-checklist.md](https://github.com/creative-graphic-design/design-generators/blob/main/docs/implementation-checklist.md) (source of truth).

<!-- Unchecked template boxes are permitted; use [x] only for verified items. Ready-for-review evidence is checked in the Completion Gate below. -->

- [ ] Confirmed the applicable implementation checklist items.
- [ ] Verify the issue reference, issue metadata, PR labels, and issue status follow [repository tracking rules](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#tracking).
- [ ] Verify model plans and amendments using the [repository source-of-truth rule](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#sources-of-truth), if this is a model PR.
- [ ] See the [model-conversion procedure](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity).
- [ ] Keep vendor sources read-only.
- [ ] Do not commit generated fixtures, weights, images, or downloaded artifacts.
- [ ] Did not push Hub repositories or model artifacts unless explicitly requested.
- [ ] Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.
- [ ] README reproduction steps are copy-pasteable commands, if README docs changed.
- [ ] See the [documentation skill](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#model-readme-and-model-card).
- [ ] Documented any deviations from the plan, checklist, or repository conventions below.

## Completion Gate

<!-- Draft PRs may leave these pending. Ready-for-review PRs must satisfy each item and replace every placeholder. Cite claimed datasets, training/evaluation seed scope, and S0-S4 evidence before S5. Replace the S5 template line with evidence or a concrete N/A reason. Infrastructure-only work must state that full-run reproduction is not claimed. Independent adversarial review must be checked; other items allow the reasons stated in their wording. -->

- [ ] Vendor parity verified, or gated-pending: <independent rerun, cases, criterion, and artifact scope; or state why parity is not applicable; otherwise name the blocker>.
- [ ] Training S5 reproduction complete, or N/A: <datasets, training/evaluation seeds, S0-S4 evidence, or N/A reason>.
- [ ] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)

<!-- Optional for complete PRs that must remain draft:
## Draft Reason
State the blocker and the resolution trigger that will make this ready for review.
-->

## Deviations / Follow-ups

<!-- Model PRs: summarize parity, checkpoint, dataset, and Hub-card deviations here. Infra/docs PRs: summarize scope limits, skipped commands, or follow-up cleanup here. -->

- TODO
