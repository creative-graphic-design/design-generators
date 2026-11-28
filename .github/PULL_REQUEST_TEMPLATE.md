## Summary

<!-- What changed and why. Include `Closes #N` or `Refs #N` for the implementation issue. The implementation checklist alone does not satisfy the PR metadata check. -->

- TODO

## Changes

- TODO

## Shared Library Changes

<!-- Required when touching cross-cutting paths such as `lib/*/src`, `docs/**`, or `.agents/skills/**`. Explain why the shared change belongs in this PR; otherwise write `N/A`. -->

- TODO

## Verification

<!-- Record commands and results, including passes and skips. For rendered-site inspection, follow the [documentation skill](../.agents/skills/design-generators-documentation/SKILL.md#verification) and list opened page URLs with the checks performed. -->

- TODO

## Checklist

Full checklist: see [docs/implementation-checklist.md](https://github.com/creative-graphic-design/design-generators/blob/main/docs/implementation-checklist.md) (source of truth).

<!-- Unchecked template boxes are permitted; use [x] only for verified items. Ready-for-review evidence is checked in the Completion Gate below. -->

- [ ] Confirmed the applicable implementation checklist items.
- [ ] Verify the issue reference, issue metadata, PR labels, and issue status follow [repository tracking rules](../AGENTS.md#tracking).
- [ ] Verify model plans and amendments using the [repository source-of-truth rule](../AGENTS.md#sources-of-truth), if this is a model PR.
- [ ] Follow the [model-conversion parity procedure](../.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity) for vendor sources and generated artifacts.
- [ ] Did not push Hub repositories or model artifacts unless explicitly requested.
- [ ] Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.
- [ ] Follow the [documentation skill](../.agents/skills/design-generators-documentation/SKILL.md#model-readme-and-model-card) for README reproduction guidance, if README docs changed.
- [ ] Documented any deviations from the plan, checklist, or repository conventions below.

## Completion Gate

<!-- Draft PRs may leave these pending. Ready-for-review PRs must satisfy each item, replace every template placeholder, and replace the untouched S5 template text with evidence or a concrete N/A reason. Independent adversarial review must be checked; other items allow the reasons stated in their wording. -->

- [ ] Vendor parity verified, or gated-pending: <independent rerun, cases, criterion, and artifact scope; or state why parity is not applicable; otherwise name the blocker>.
- [ ] Training S5 reproduction complete, or N/A: <claimed datasets and training/evaluation seed scope, with S0-S4 evidence cited before S5; or N/A with reason>. Infrastructure-only work must state that full-run reproduction is not claimed.
- [ ] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)

<!-- Optional for complete PRs that must remain draft:
## Draft Reason
State the blocker and the resolution trigger that will make this ready for review.
-->

## Deviations / Follow-ups

<!-- Model PRs: summarize parity, checkpoint, dataset, and Hub-card deviations here. Infra/docs PRs: summarize scope limits, skipped commands, or follow-up cleanup here. -->

- TODO
