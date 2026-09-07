> [!NOTE]
> After reading this AGENTS.md, say: 🤖 I read the AGENTS.md for creative-graphic-design/design-generators.

# Agent Instructions

## Project

`design-generators` ports layout, poster, and graphic-design generation research repositories into Transformers/Diffusers-style packages that can load converted weights with `from_pretrained` and run inference immediately. Keep repository rules here stable and concise; procedural implementation guidance lives in repo-local skills.

## Sources Of Truth

- Umbrella policy, target table, execution order, interface decisions, data policy, and status tracking live in [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2).
- The living implementation checklist is [issue #60 (implementation checklist)](https://github.com/creative-graphic-design/design-generators/issues/60). Check it before starting a model package and quote verification results in the PR body.
- Shared library structure is [issue #64 (shared library structure)](https://github.com/creative-graphic-design/design-generators/issues/64): workspace members are `lib/*` and `models/*`; shared layout helpers import from `laygen.common`; poster helpers import from `posgen.common` when needed.
- A model issue's plan comment plus all later amendment comments define that model's design. Amendments override earlier plan text.

## Workspace

- The root uv workspace uses members `["lib/*", "models/*"]`.
- Run member-specific commands with the member package selected: `uv run --package <name> ...`. Examples: `uv run --package laygen pytest`, `uv run --package layout-dm pytest`.
- Do not run plain root `uv run` against a member path when the command depends on that member's extras, dependency source mapping, or package metadata.
- Do not commit host-specific absolute filesystem paths. Pass runtime absolute paths through environment variables or CLI arguments; repository defaults must be repo-root-relative.

## Repo-Local Skills

- Skill source of truth is `.agents/skills/<name>/SKILL.md`; repo-local skills are named with the `design-generators-` prefix.
- `.claude/skills` is a relative symlink to `.agents/skills` for Claude compatibility.
- Codex should read the relevant `.agents/skills/<name>/SKILL.md` directly when a repo-local skill applies.
- For model conversion, use the `design-generators-model-conversion` skill (read `.agents/skills/design-generators-model-conversion/SKILL.md`).
- For parity, use the `design-generators-model-conversion` skill (read `.agents/skills/design-generators-model-conversion/SKILL.md`).
- For data sources, use the `design-generators-model-conversion` skill (read `.agents/skills/design-generators-model-conversion/SKILL.md`).
- For documentation, use the `design-generators-documentation` skill (read `.agents/skills/design-generators-documentation/SKILL.md`).
- For training reproduction, use the `design-generators-training-reproduction` skill (read `.agents/skills/design-generators-training-reproduction/SKILL.md`).

## Naming And Core Implementation

- Model package names and Hub repo ids use the method name known in the literature, not necessarily the vendor repository slug.
- Example: vendor `const-layout` becomes package `layoutganpp` and Hub ids such as `creative-graphic-design/layoutganpp-rico`.
- Shared packages are `laygen.common` for layout-generation utilities and `posgen.common` for poster/content-aware utilities.
- Keep `__init__` bodies to variable initialization only.
- Make data-holding classes `dataclass`es and put unavoidable initialization logic in `__post_init__` or a classmethod factory.
- Consider pydantic models at serialization boundaries where runtime validation pays for itself; do not duplicate validation LightningCLI/jsonargparse already performs. Classes bound by framework constructor contracts (`PretrainedConfig`, `PreTrainedModel`, `LightningModule`, ...) follow the framework idiom.
- Prefer guard clauses: return or raise early for simple or invalid cases so the main path reads at minimal nesting; do not build tail-return pyramids.
- Do not weaken annotations to satisfy checkers. Replacing precise annotations with `object`, bare containers, or similarly less informative types is prohibited; annotations must move toward more precise types.

## Tracking

- If the user asks to create or file an issue first, treat issue creation as a hard gate: create the issue and record its URL before implementation, commits, or PR work. Do not substitute a PR for the requested issue.
- Every implementation PR must reference its implementation issue in the PR summary with `Closes #N` or `Refs #N`. The standing checklist issue #60 does not count as the implementation issue.
- Every PR must carry the same lane/topic labels as its implementation issue, such as `ready-heavy`, `documentation`, or `meta`; status labels stay on issues only and must not be added to PRs.
- PR bodies must be built by filling in `.github/PULL_REQUEST_TEMPLATE.md`; do not replace the template when creating PRs with `gh pr create --body`.
- Priority labels select the work lane; status labels move in this order: `plan-agreed` -> `in-progress` -> `parity-verified` -> published/closed.
- Complete draft PRs must be marked ready for review or carry a `## Draft Reason` section.
- When creating any issue, set both the milestone and the native Priority issue field; do not leave either unset. Set the native Priority field through GraphQL `setIssueFieldValue` when the CLI surface is insufficient.
- Add `in-progress` when work on a model issue begins.
- Add `parity-verified` only after the coordinator independently reruns the parity suite and confirms the results.
- Close a model issue only after every planned Hub repo for that issue has a passing `from_pretrained` smoke test.
- Milestones are execution phases: `v0.1` foundation and pilot wave, `v0.2` ready-light completion, `v0.3` ready-heavy, `v0.4` LLM recipes and Pydantic AI, and `v0.5` train-ourselves.

## Self-Improvement

- Treat `AGENTS.md`, repo-local skills, PR templates, and checklist issues as living documents. When work exposes incorrect, stale, or missing guidance, fix it in the same PR if the change is small and in scope; otherwise open or propose a focused `meta` issue.
- Do not silently work around guidance known to be wrong. If the same kind of mistake is raised repeatedly, add or revise a rule, template item, or check so future work can catch it mechanically.
- Keep PR diffs minimal for the stated task. Do not move dependencies between core `dependencies` and `[project.optional-dependencies]`, or add/modify `[build-system]`, unless the task requires it and the PR explains why.

## Reader-First Documentation

- Reader-facing documents (package `TRAINING.md` and `README.md` files, `docs/*.md`, PR and issue bodies) are written for a first-time reader with no knowledge of this repository's history. Lead with the claim or outcome; define or link internal terms, roles, and stage codes at first use; do not open with corrections to earlier states the reader has never seen. Before writing or editing such a document, declare the intended reader and judge every sentence by its value to that reader.
- Do not hard-wrap markdown prose mid-sentence at a column width; write each bullet, paragraph, and table cell as one logical line, breaking only at structural boundaries.

## Machine-Checked Conventions

- `scripts/check_committed_paths.py` rejects host-specific absolute paths in tracked files, with its documented exclusions.
- `scripts/check_src_vendor_language.py`, `scripts/check_jaxtyping_annotations.py`, and `scripts/check_config_defaults.py` enforce source-language boundaries, shaped-annotation/baseline rules, and explicit-config construction.
- `scripts/check_semantic_blank_lines.py` and `scripts/check_module_naming.py` enforce semantic, raise-block, and compound-suite blank lines plus core module-name prefixes, suffixes, and allowlists.
- `scripts/check_model_readmes.py`, `scripts/check_readme_badges.py`, `scripts/check_readme_links.py`, and `scripts/check_reader_facing_references.py` enforce README/model-card, badge, repository-link, and reader-reference contracts.
- `scripts/check_training_doc_template.py` and `scripts/check_training_stage_evidence.py` enforce training-document structure and S0-S5 claims.
- `scripts/check_pr_issue_reference.py`, `scripts/check_changed_urls.py`, and `scripts/check_draft_prs.py` enforce PR issue/checklist/completion gates, excluding standing issues #2 and #60, changed-URL status, and draft completion; their CI entry points are `.github/workflows/ci.yml`, `.github/workflows/draft-pr-audit.yml`, and `.github/workflows/link-check.yml`.
- `.github/workflows/ci.yml` resolves members with `uv sync --all-packages`, runs pre-commit with `SKIP=uv-lock`, `ty`, root tests without package coverage, `scripts/gen_ref_pages.py` over packages below each member's `src/`, strict Zensical, and workspace-member tests; `scripts/run_member_tests.sh` excludes `vendor_parity` and `integration`, measures each member separately without combined coverage, and enforces separate 90% floors; workspace tests include frontmatter and layout-output schema checks.

## CI Policy

- Keep `uv-lock` local because this environment bakes global uv options into `uv.lock`; do not add `uv lock --check` or uv-lock to CI.
