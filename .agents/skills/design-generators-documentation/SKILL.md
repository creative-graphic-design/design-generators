---
name: design-generators-documentation
description: Use when writing or reviewing design-generators documentation, model cards, README files, API docstrings, reproduction instructions, or docs-site pages for a first-time reader.
---

# Documentation

Write for a first-time package user or contributor. State the intended reader before editing. Paths below are repository-root-relative unless stated otherwise.

## Read by document

| Document being changed          | Read when needed                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Model README or Hub card        | `.agents/skills/design-generators-model-conversion/references/model-readme-template.md`, package metadata and measured results |
| Inference reproduction commands | Package `REPRODUCING.md`, scripts and CLI help for the commands being documented                                               |
| Training procedure or result    | `design-generators-training-reproduction` skill, `docs/training-reproduction.md`, and `docs/templates/TRAINING.template.md`    |
| Shared API or ownership         | `docs/conventions.md`, `docs/architecture.md`, and the implementation being described                                          |
| Docs navigation or API pages    | `mkdocs.yml` and `tests/test_docs_generation.py`                                                                               |

Read the model issue and amendments when changing its scope, interface, or evidence claims. A typo or link correction does not require reading unrelated plans, external model pages, or training artifacts.

## Reader-first guidance

- Before finishing a package `README.md`, `TRAINING.md`, PR body, or issue evidence comment, ask a reader who has not seen the artifact to read only its opening, without additional context. Confirm that they can identify the question it addresses, the method used, the result, and what the result means or changes. If any part is missing, rewrite the opening before publishing.
- Follow the reader-first and no-hard-wrap rules in the repository's core `AGENTS.md`.
- Before publishing or approving a document, run a reduction pass: for every table column, table row, and sentence, state what a first-time reader would lose if it were deleted, and delete anything that loses nothing. A column that restates another column, a row whose owner cell already says what the row would add, and a sentence that repeats the section's opening principle are the usual casualties.
- Write facts and rules as reader-facing prose; do not expose work-history or evidence-receipt language without context.
- Put a section where its subject belongs; do not grow a section merely where related work happened.
- Give every issue or pull-request reference a descriptive Markdown link, including references in headings.
- The reader-facing reference checker enforces the linked-reference rule; it does not judge prose, structure, or terminology.
- Treat excessive emotional intensity, redundant enumeration, and hedging as model-review targets for the slop review (the review of reader-facing text for machine-writing patterns); do not add them to this checker.
- An exact result is agreement from explicitly matched inputs, evaluator, checkpoint-selection rule, and a comparison criterion of bitwise or token-identical equality, with the criterion met; a tolerance criterion supports a within-tolerance claim, not an exact one.
- A confirmatory p-value is the result of a test and threshold named in the model issue plan or an amendment before the run.
- An exploratory p-value is any other p-value; label it exploratory, name the test and sample or seed scope, state that the result covers only the named samples or seeds and that a non-significant p-value does not show equivalence, and do not present it as confirmatory evidence.
- State rounding once per table or paragraph. Rounded values are not exact, and equal rounded values do not establish exact agreement. For example: `Values are rounded to two decimal places. Package and vendor alignment are 0.81 and 0.81. Exploratory two-sided Welch test, 10 training seeds per system: p = 0.42; this covers only these seeds and does not show equivalence.`

## Model README and model card

Use the README template for required metadata and headings. Keep its overview, paper link, and usage clear; use `devharness check model-readmes` as the executable check. Do not run a new model campaign to fill a missing result: report unavailable evidence explicitly.

- Human review checks that the first `python` usage fence in each model README is followed, after optional blank lines, by a `text` fence containing the output produced when that snippet is run as written; the captured output is provenance for the snippet, not a hand-written example.

- Each model package README has exactly one structured classification line in the form `- **Model type:** <content>; task: <task>; conditioning: <comma-separated canonical values in conventions order>.`, with lowercase content values `content-agnostic` or `content-aware`, task values `task-agnostic`, `task-aware`, or `single-task` for generators, and `evaluation` or `saliency` for non-generation roles; `single-task` is used iff exactly one conditioning value is accepted, while `task-agnostic` and `task-aware` require two or more conditioning values; the conditioning values `evaluation`, `saliency`, and `none` appear alone.
- Use pip direct-reference installation for package users, including unpublished workspace dependencies in the same command. Preserve clone/uv setup in `How to Get Started with the Model` when a local converted checkpoint or prompt configuration is required before Hub publication.
- `### Parity Results` under `## Evaluation` contains measured cases, criteria, and results. `## Reproducibility` opens with how to rerun agreement checks and links to `models/<slug>/REPRODUCING.md`.
- Put the ordered command walkthrough in `REPRODUCING.md`: prerequisites, cache locations, asset download, original-code reference generation, package comparison, conversion or prompt-configuration serialization, and local loading. Include expected outputs and required extras. A metadata export alone is not a generated model reference.
- For an acceptance rerun, use `PARITY_REQUIRE=1` and one selected CUDA device where applicable; distinguish passes from skips. Preserve package-specific dataset and condition sweeps.
- Keep planned Hub ids distinct from published checkpoints. Hub cards use `laygen.common.model_card` and the official Hugging Face template; package metadata ownership is in `docs/architecture.md`. Publication follows the separate user authorization in `AGENTS.md`.
- Environment-specific documentation must distinguish observed verification conditions from general requirements. Write "the currently verified setup is ..." or equivalent when only one machine/GPU/driver combination has been tested; do not present that setup as the package's inherent training environment.

## API and docs site

Public API docstrings feed the API reference. Use Google-style `Args`, `Returns`, `Raises`, and `Examples` as applicable. Examples should be runnable doctests when they need no heavyweight downloads or credentials.

Every `docs/*.md` page needs `icon: lucide/...` and non-empty `tags` frontmatter. The API reference uses committed `docs/api/<group>/<package>.md` stubs with the package README included through `pymdownx.snippets`, an `## API Reference` heading, `::: <import_name>`, and `show_submodules: true`, plus navigation in `mkdocs.yml`. Do not commit locally generated alternative API trees.

- Model READMEs wrap everything after the YAML front matter in `<!-- --8<-- [start:card] -->` and `<!-- --8<-- [end:card] -->` so the front matter stays off the docs page.
- Markdown code fences must be tagged. Use `bash` for executable shell commands and `text` for non-executable output, logs, or examples.

Package documents (`lib/*/README.md`, `models/*/README.md`, `models/*/TRAINING.md`, and `models/*/REPRODUCING.md`) link repository files with absolute GitHub URLs under the repository's `blob/main/` or `tree/main/` prefix because the same text is read on GitHub, included on the docs site, and reused on the Hub. The root README and skills use repo-root-relative links.

When adding a conference badge, take its color from the venue's official branding.

## Verification

Run the documentation checks and strict site build in `docs/implementation-checklist.md#verification-commands`, plus required pre-commit gates. Use existing checker diagnostics to resolve structure, badge, link, training-evidence, and navigation failures. Syntax checks establish document consistency, not scientific reproduction.

- When a change touches `mkdocs.yml`, `docs/api/**`, or `docs/stylesheets/**`, inspect the rendered site with `zensical serve` or the built site and list each opened page URL in the PR `Verification` section with the navigation targets, rendered headings, API members, code blocks, and links checked on that page.

When editing templates or copyable examples, check required headings, status values, and prefixes against the consuming checker, and validate a filled example. Checking existing package documents alone does not validate their template.

Before review, check that the commands match the actual scripts and the claims match recorded evidence. Report unrun heavyweight commands and unavailable assets in the PR rather than presenting them as passes.

- Run the documentation and root gates through uv, and run the full pre-commit suite before opening a PR. The PR must use `.github/PULL_REQUEST_TEMPLATE.md` and include the implementation issue, checklist verification, tests, and shared-library rationale when applicable.
