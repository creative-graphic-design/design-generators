## Summary

<!-- What changed and why. Include `Closes #N` or `Refs #N` for the implementation issue. The implementation checklist alone does not satisfy the PR metadata check. -->

- TODO

## Changes

- TODO

## Shared Library Changes

<!-- Required when touching cross-cutting paths such as `lib/*/src`, `docs/**`, or `.agents/skills/**`. Explain why the shared change belongs in this PR; otherwise write `N/A`. -->

- TODO

## Verification

<!-- Record executed commands and results. Include coverage and parity numbers when applicable, distinguishing passes from skips. For docs/skills, report the affected rules and the existing checks; do not claim model parity from documentation checks. When changing `mkdocs.yml`, `docs/api/**`, or `docs/stylesheets/**`, list each page URL opened in `zensical serve` or the built site and the checks performed on that page. Example: `http://127.0.0.1:8000/api/models/layout-dm/` — navigation, rendered headings, API members, code blocks, and links. -->

- TODO

## Checklist

Full checklist: see [docs/implementation-checklist.md](https://github.com/creative-graphic-design/design-generators/blob/main/docs/implementation-checklist.md) (source of truth).

- [ ] Confirmed the applicable implementation checklist items.
- [ ] Referenced the implementation issue with `Closes #N` or `Refs #N` in the Summary; the standing umbrella issue and implementation checklist alone do not satisfy this.
- [ ] Confirmed the implementation issue has a milestone and native Priority field set.
- [ ] Applied the same lane/topic labels as the implementation issue to this PR; status labels such as `plan-agreed`, `in-progress`, and `parity-verified` stay on the issue.
- [ ] Read the model plan and amendment comments, if this is a model PR.
- [ ] Left `vendor/` read-only and did not commit generated fixtures, weights, images, or downloaded artifacts.
- [ ] Did not push Hub repositories or model artifacts unless explicitly requested.
- [ ] Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.
- [ ] README reproducibility steps are copy-pasteable commands, if README docs changed.
- [ ] Documented any deviations from the plan, checklist, or repository conventions below.

## Completion Gate

<!-- Draft PRs may leave these pending. Ready-for-review PRs must either satisfy each item, or keep an actionable blocker/reason in the item text. -->

- [ ] Vendor parity verified, or gated-pending: <independent rerun, cases, criterion, and artifact scope; or state why parity is not applicable; otherwise name the blocker>.
- [ ] Training S5 reproduction complete, or N/A: <claimed datasets and training/evaluation seed scope, with S0-S4 evidence cited before S5; or N/A with reason>. Infrastructure-only work must state that full-run reproduction is not claimed.
- [ ] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)

<!-- Optional for complete PRs that must remain draft:
## Draft Reason
State the blocker and the resolution trigger that will make this ready for review.
-->

## Deviations / Follow-ups

<!-- Model PRs: summarize parity, checkpoint, dataset, and Hub-card deviations here. Infra/docs PRs: summarize scope limits, skipped commands, or follow-up cleanup here. -->

- TODO
