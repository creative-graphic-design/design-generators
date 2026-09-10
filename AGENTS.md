> [!NOTE]
> After reading this AGENTS.md, say: 🤖 I read the AGENTS.md for creative-graphic-design/design-generators.

# Agent Instructions

## Project

`design-generators` ports layout, poster, and graphic-design generation research repositories into Transformers/Diffusers-style packages that can load converted weights with `from_pretrained` and run inference immediately. Keep repository rules here stable and concise; procedural implementation guidance lives in repo-local skills.

## Sources Of Truth

- Stable targets and execution order live in [docs/roadmap.md](docs/roadmap.md), and cross-package dataset policy and sources live in [docs/data-sources.md](docs/data-sources.md); public interface policy remains in [docs/conventions.md](docs/conventions.md), shared-library structure remains in [docs/architecture.md](docs/architecture.md), and repository workflow remains here. Historical discussion remains preserved in [issue #2](https://github.com/creative-graphic-design/design-generators/issues/2).
- The implementation checklist is [docs/implementation-checklist.md](docs/implementation-checklist.md). Check it before starting a model package and quote verification results in the PR body; historical checklist discussion remains preserved in [issue #60](https://github.com/creative-graphic-design/design-generators/issues/60).
- Shared library structure is defined in [docs/architecture.md](docs/architecture.md): workspace members are `lib/*` and `models/*`; shared layout helpers import from `laygen.common`; poster helpers import from `posgen.common` when needed. Historical discussion remains preserved in [issue #64 (shared library structure)](https://github.com/creative-graphic-design/design-generators/issues/64).
- A model issue's plan comment plus all later amendment comments define that model's design. Amendments override earlier plan text.

## Workspace

- The root uv workspace uses members `["lib/*", "models/*"]`.
- Run member-specific commands with the member package selected: `uv run --package <name> ...`. Examples: `uv run --package laygen pytest`, `uv run --package layout-dm pytest`.
- Do not run plain root `uv run` against a member path when the command depends on that member's extras, dependency source mapping, or package metadata.
- Do not commit host-specific absolute filesystem paths. Pass runtime absolute paths through environment variables or CLI arguments; repository defaults must be repo-root-relative.

## Repo-Local Skills

- Skill source of truth is `.agents/skills/<name>/SKILL.md`.
- `.claude/skills` is a relative symlink to `.agents/skills` for Claude compatibility.
- Repo-local skills are named with the `design-generators-` prefix.
- Codex should read the relevant `.agents/skills/<name>/SKILL.md` directly when a repo-local skill applies.

### Model Conversion

- For model conversion, parity, data sources, public interfaces, and layout output schemas, including `laygen`/`posgen` shared-library work, use the `design-generators-model-conversion` skill (read `.agents/skills/design-generators-model-conversion/SKILL.md`).
- Scope: this skill owns model conversion, vendor parity, approved data sources, public interfaces, and layout output schemas.

### Documentation

- For documentation, use the `design-generators-documentation` skill (read `.agents/skills/design-generators-documentation/SKILL.md`).
- Scope: this skill owns documentation, README and model-card, API-docstring, reproduction-instruction, and docs-site guidance.

### Training Reproduction

- For training reproduction, use the `design-generators-training-reproduction` skill (read `.agents/skills/design-generators-training-reproduction/SKILL.md`).
- Scope: this skill owns package-local S0-S5 training reproduction and training-document guidance.

## Naming And Core Implementation

### Naming

- Model package names and Hub repo ids use the method name known in the literature, not necessarily the vendor repository slug.
- Example: vendor `const-layout` becomes package `layoutganpp` and Hub ids such as `creative-graphic-design/layoutganpp-rico`.
- Shared packages are `laygen.common` for layout-generation utilities and `posgen.common` for poster/content-aware utilities.

### Class Design And Code Style

- Keep `__init__` bodies to variable initialization only.
- Make data-holding classes `dataclass`es and put unavoidable initialization logic in `__post_init__` or a classmethod factory.
- Consider pydantic models at serialization boundaries where runtime validation pays for itself; do not duplicate validation LightningCLI/jsonargparse already performs. Classes bound by framework constructor contracts (`PretrainedConfig`, `PreTrainedModel`, `LightningModule`, ...) follow the framework idiom.
- Prefer guard clauses: return or raise early for simple or invalid cases so the main path reads at minimal nesting; do not build tail-return pyramids.
- Do not weaken annotations to satisfy checkers. Replacing precise annotations with `object`, bare containers, or similarly less informative types is prohibited; annotations must move toward more precise types.
- Within function bodies, separate semantic units (configuration branches, submodule construction, transformations, and return preparation) with single blank lines; always leave a blank line after a raise block when ordinary code follows; leave one blank line after an if, for, while, try, or with suite when ordinary code follows at the enclosing indentation.

## Tracking

### Issues

- If the user asks to create or file an issue first, treat issue creation as a hard gate: create the issue and record its URL before implementation, commits, or PR work. Do not substitute a PR for the requested issue.

### Pull Requests

- Every implementation PR must reference its implementation issue in the PR summary with `Closes #N` or `Refs #N`. The implementation checklist document does not count as the implementation issue.
- Every PR must carry the same lane/topic labels as its implementation issue, such as `ready-heavy`, `documentation`, or `meta`; status labels stay on issues only and must not be added to PRs.
- PR bodies must be built by filling in `.github/PULL_REQUEST_TEMPLATE.md`; do not replace the template when creating PRs with `gh pr create --body`.
- Complete draft PRs must be marked ready for review or carry a `## Draft Reason` section.

### Status Labels And Milestones

- Priority labels select the work lane; status labels move in this order: `plan-agreed` -> `in-progress` -> `parity-verified` -> published/closed.
- When creating any issue, set both the milestone and the native Priority issue field; do not leave either unset. Set the native Priority field through GraphQL `setIssueFieldValue` when the CLI surface is insufficient.
- Add `in-progress` when work on a model issue begins.
- Add `parity-verified` only after the coordinator independently reruns the parity suite and confirms the results.
- Serialize converted model artifacts with the standard `save_pretrained`/`from_pretrained` pair.
- Treat publication to the `creative-graphic-design` Hugging Face organization as a separate approved step.
- Track publication separately from local serialization and parity verification.
- Close a model issue only after its implementation is merged to `main`, every planned checkpoint, dataset, or task repository has a passing local `save_pretrained` to `from_pretrained` smoke test, and vendor parity is independently verified.
- Hub publishing is deferred and is not part of the per-issue close condition.
- Milestones are execution phases: `v0.1` foundation and pilot wave, `v0.2` ready-light completion, `v0.3` ready-heavy, `v0.4` LLM recipes and Pydantic AI, and `v0.5` train-ourselves.

## Self-Improvement

- Treat `AGENTS.md`, repo-local skills, PR templates, and checklist issues as living documents. When work exposes incorrect, stale, or missing guidance, fix it in the same PR if the change is small and in scope; otherwise open or propose a focused `meta` issue.
- Do not silently work around guidance known to be wrong. If the same kind of mistake is raised repeatedly, add or revise a rule, template item, or check so future work can catch it mechanically.
- Keep PR diffs minimal for the stated task. Do not move dependencies between core `dependencies` and `[project.optional-dependencies]`, or add/modify `[build-system]`, unless the task requires it and the PR explains why.

## Reader-First Documentation

- Reader-facing documents (package `TRAINING.md` and `README.md` files, `docs/*.md`, PR and issue bodies) are written for a first-time reader with no knowledge of this repository's history. Lead with the claim or outcome; define or link internal terms, roles, and stage codes at first use; do not open with corrections to earlier states the reader has never seen. Before writing or editing such a document, declare the intended reader and judge every sentence by its value to that reader.
- Do not hard-wrap markdown prose mid-sentence at a column width; write each bullet, paragraph, and table cell as one logical line, breaking only at structural boundaries.

## Machine-Checked Conventions

### Source Checks

- `scripts/check_committed_paths.py` rejects host-specific absolute paths in tracked files, with its documented exclusions.
- `scripts/check_src_vendor_language.py` enforces source-language boundaries and the documented `laygen.common.vendor` exception.
- `scripts/check_jaxtyping_annotations.py` enforces shaped-annotation and baseline rules.
- `scripts/check_config_defaults.py` enforces explicit-config construction.
- `scripts/check_semantic_blank_lines.py` checks raise-block and compound-suite blank lines.
- `scripts/check_module_naming.py` enforces core module-name prefixes, suffixes, and allowlists.

### Documentation Checks

- `scripts/check_model_readmes.py` enforces README and model-card contracts.
- `scripts/check_readme_badges.py` enforces README badge contracts.
- `scripts/check_readme_links.py` enforces repository-relative README links.
- `scripts/check_reader_facing_references.py` enforces reader-facing reference contracts.
- `scripts/check_training_doc_template.py` enforces training-document structure.
- `scripts/check_training_stage_evidence.py` enforces S0-S5 claims.
- `tests/test_docs_generation.py` checks that every `docs/*.md` page has icon and tags frontmatter.
- `lib/laygen/tests/test_common.py` checks the shared layout-output schema.

### PR And CI Gates

- `scripts/check_pr_issue_reference.py` enforces PR issue and checklist references, excluding the standing roadmap/data-source issue and historical checklist issue.
- `scripts/check_changed_urls.py` enforces changed-URL status in `.github/workflows/ci.yml`, and `.github/workflows/link-check.yml` checks full Markdown links.
- `scripts/check_draft_prs.py` enforces draft completion, and `.github/workflows/draft-pr-audit.yml` runs it daily.
- `.github/workflows/ci.yml` is the CI entry point for pre-commit with `SKIP=uv-lock`, `ty`, root tests, generated API pages, strict Zensical, and workspace-member tests; `.github/workflows/ci.yml` resolves members with `uv sync --all-packages`.
- `scripts/run_member_tests.sh` excludes `vendor_parity` and `integration` tests from regular member-test runs.
- CI runs root pytest without coverage because the root has no import package, and each workspace member is measured separately without combined coverage.
- Coverage has a 90% floor for every workspace member; do not lower `fail_under` below 90; member-specific overrides may only raise the floor.

## CI Policy

- Keep `uv-lock` local because this environment bakes global uv options into `uv.lock`; do not add `uv lock --check` or uv-lock to CI.
