---
name: design-generators-documentation
description: Use when writing or reviewing design-generators documentation, model cards, README files, API docstrings, reproduction instructions, or docs-site pages for a first-time reader.
---

# Design Generators Documentation

Use this skill for repository documentation and model-card work. Read `AGENTS.md`, the [implementation checklist](docs/implementation-checklist.md), and the relevant model issue before editing. The intended reader is a first-time agent or contributor with no knowledge of this repository's history.

## Reader-first contract

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

## Repository documentation rules

- The docs site uses `zensical` and `mkdocstrings[python]`; build it with:
  ```bash
  uv run --group docs zensical build --strict -f mkdocs.yml
  ```
- The API reference has one committed stub under `docs/api/` for each workspace member under `lib/*` and `models/*`; each stub includes the package README with `pymdownx.snippets` (`--8<-- "models/<slug>/README.md:card"` for models, `--8<-- "lib/<slug>/README.md"` for libraries), then an `## API Reference` heading and `::: <import-name>` with `show_submodules: true`. Model READMEs wrap everything after the YAML front matter in `<!-- --8<-- [start:card] -->` and `<!-- --8<-- [end:card] -->` so the front matter stays off the docs page.
- Public API docstrings are the source text for the API reference. Use google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples` sections for public pipelines, tokenizers, processors, configs, `laygen.common` modules, `posgen.common` modules, and agents.
- `Examples` in public API docstrings should be doctest-ready snippets whenever the API can run without heavyweight assets, downloads, or credentials.
- Every `docs/*.md` page needs YAML frontmatter with `icon: lucide/...` and non-empty `tags`.
- Each model package README uses a model-card style: overview, install/usage snippet, supported checkpoints/Hub ids, datasets, reproducibility summary with vendor-parity numbers that compare against the original implementation, license, citation, and original implementation link.
- Each model package README has exactly one structured classification line in the form `- **Model type:** <content>; task: <task>; conditioning: <comma-separated canonical values in conventions order>.`, with lowercase content values `content-agnostic` or `content-aware`, task values `task-agnostic`, `task-aware`, or `single-task` for generators, and `evaluation` or `saliency` for non-generation roles; the conditioning values `evaluation`, `saliency`, and `none` appear alone.
- Each package README's install snippet uses `pip install "pkg @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=<path>"`, co-specifying required workspace libraries such as `laygen` and `posgen` in the same command; clone + uv flows are for development and `REPRODUCING` docs.
- README and model-card repository/source links must be copied from `.gitmodules` or the implementation issue, then checked for a resolving HTTP response before commit. Do not write upstream repository, project-page, dataset, or source links from memory. PR CI mechanically verifies newly added external URLs and rejects added 404/410 links.
- Each README includes `Reproducibility`, opening with one sentence that states how to reproduce the original-implementation agreement checks, followed by copy-pasteable commands for download, vendor reference generation, parity tests, conversion, and `from_pretrained` smoke tests.
- When adding a new conference venue badge to the root README, check the conference's official site, logo, or style assets first and use a badge color that matches that venue rather than choosing an arbitrary generic color.
- Markdown code fences must be tagged. Use `bash` for executable shell commands and `text` for non-executable output, logs, or examples.
- Docs and READMEs link the first mention of external projects and repositories. Do not use internal validation stage codes such as `S0-S2` in reader-facing docs unless that page defines them in place or links directly to the definition.
- Environment-specific documentation must distinguish observed verification conditions from general requirements. Write "the currently verified setup is ..." or equivalent when only one machine/GPU/driver combination has been tested; do not present that setup as the package's inherent training environment.
- Package documents (`lib/*/README.md`, `models/*/README.md`, `models/*/TRAINING.md`, and `models/*/REPRODUCING.md`) link repository files with absolute GitHub URLs under the repository's `blob/main/` (or `tree/main/` for directories) prefix, because the same text is read on GitHub, included on the docs site, and reused on the Hub; relative links break in at least one of those places. The root README and skills use repo-root-relative links such as `docs/training-reproduction.md`.

## Model README and Hub model-card procedure

Start `models/<slug>/README.md` from `.agents/skills/design-generators-model-conversion/references/model-readme-template.md`, which follows the Hugging Face Hub model card metadata spec and the `huggingface_hub` official `modelcard_template.md` headings. Fill every placeholder with the target issue's concrete model, checkpoint, dataset, parity, license, and citation details. Keep the final README in model-card style. Include:

- overview and original implementation link
- install and `from_pretrained` usage
- supported checkpoints and intended Hub ids
- datasets and pinned configs
- reproducibility summary with vendor-parity numbers
- license status and citation
- `Reproducibility` link to `models/<slug>/REPRODUCING.md`

The user-facing install snippet must use pip direct references to this repository's package subdirectories. Include workspace libraries that are not published on PyPI, such as `laygen` or `posgen`, in the same command as the model package. Preserve clone + uv commands only for development or `REPRODUCING.md` workflows.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "<package-name> @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/<slug>"
```

Omit `posgen` when the model does not depend on it, and keep extras on the shared package requirement when the model depends on one, for example `laygen[agents]`.

Every model README must include a `### Parity Results` section under `## Evaluation`. Put the vendor-parity summary in a numeric table that states what was compared, the number of cases, the match criterion, and the result, and state rounding once per table when values are rounded. Prose inside `## Reproducibility` is not enough. Put the mechanical walkthrough in `REPRODUCING.md`; reviewers run that file's download, reference-generation, parity, conversion, and smoke-test commands. `.agents/skills/design-generators-model-conversion/references/model-readme-template.md` is the reference format.

The `Reproducibility` section must open with one sentence that states how to reproduce the original-implementation agreement checks. The remaining commands must be copy-pasteable and ordered: download vendor assets, generate vendor references with `CUDA_VISIBLE_DEVICES`, run `pytest -m vendor_parity`, convert checkpoints, and run `from_pretrained` smoke tests.

- Hub model cards are generated through `laygen.common.model_card` using the official Hugging Face model-card template.
- Do not push model weights or Hub repos from ordinary implementation PRs unless the coordinator (the maintainer who owns the model issue, distinct from whoever produced the evidence) explicitly asks for publish.

## Machine-checked companions

- The docs site uses hand-written `docs/api/<group>/<pkg>.md` stubs that include the package README and then `::: <pkg>` with `show_submodules: true`, explicit navigation in `mkdocs.yml`, and strict Zensical in `.github/workflows/ci.yml`; adding a package requires its stub and nav line, and `test_api_stubs_match_workspace_members_and_nav` enforces the project name with hyphens changed to underscores in the stub and nav, plus the README include. `scripts/check_readme_links.py` enforces the package-document link form.
- Model README structure, install commands, parity sections, tagged fences, first external links, and reproducibility commands are checked by `devharness check model-readmes` and `tools/devharness/tests/test_model_readmes.py`.
- Changed external URLs are checked by `scripts/check_changed_urls.py` in `.github/workflows/ci.yml`, and full Markdown links are checked by `.github/workflows/link-check.yml`.
- Reader-facing internal references are checked by `scripts/check_reader_facing_references.py`.
- Repository-relative README links are checked by `scripts/check_readme_links.py`.
- Keep rules enforced by checkers in the machine-checked conventions section of `AGENTS.md` rather than duplicating their full procedures here.

## Validation

Run the documentation and root gates through uv, and run the full pre-commit suite before opening a PR. The PR must use `.github/PULL_REQUEST_TEMPLATE.md` and include the implementation issue, checklist verification, tests, and shared-library rationale when applicable.
