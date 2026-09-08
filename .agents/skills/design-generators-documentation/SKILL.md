---
name: design-generators-documentation
description: Use when writing or reviewing design-generators documentation, model cards, README files, API docstrings, reproduction instructions, or docs-site pages for a first-time reader.
---

# Design Generators Documentation

Use this skill for repository documentation and model-card work. Read `AGENTS.md`, issue #60, and the relevant model issue before editing. The intended reader is a first-time agent or contributor with no knowledge of this repository's history.

## Reader-first contract

- Follow the reader-first and no-hard-wrap rules in the repository's core `AGENTS.md`.

## Repository documentation rules

- The docs site uses `zensical`, `mkdocstrings[python]`, and generated API pages; build it with:
  ```bash
  uv run --group docs python scripts/gen_ref_pages.py
  uv run --group docs zensical build --strict -f mkdocs.generated.yml
  ```
- The API reference is generated from workspace members under `lib/*` and `models/*`, using Python packages found below each member's `src/` directory.
- Public API docstrings are the source text for the API reference. Use google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples` sections for public pipelines, tokenizers, processors, configs, `laygen.common` modules, `posgen.common` modules, and agents.
- `Examples` in public API docstrings should be doctest-ready snippets whenever the API can run without heavyweight assets, downloads, or credentials.
- Every `docs/*.md` page needs YAML frontmatter with `icon: lucide/...` and non-empty `tags`.
- Each model package README uses a model-card style: overview, install/usage snippet, supported checkpoints/Hub ids, datasets, reproducibility summary with vendor-parity numbers, license, citation, and original implementation link.
- Each package README's install snippet uses `pip install "pkg @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=<path>"`, co-specifying required workspace libraries such as `laygen` and `posgen` in the same command; clone + uv flows are for development and `REPRODUCING` docs.
- README and model-card repository/source links must be copied from `.gitmodules` or the implementation issue, then checked for a resolving HTTP response before commit. Do not write upstream repository, project-page, dataset, or source links from memory. PR CI mechanically verifies newly added external URLs and rejects added 404/410 links.
- Each README includes `Reproducibility`, opening with one sentence that states how to reproduce the original-implementation agreement checks, followed by copy-pasteable commands for download, vendor reference generation, parity tests, conversion, and `from_pretrained` smoke tests.
- When adding a new conference venue badge to the root README, check the conference's official site, logo, or style assets first and use a badge color that matches that venue rather than choosing an arbitrary generic color.
- Markdown code fences must be tagged. Use `bash` for executable shell commands and `text` for non-executable output, logs, or examples.
- Docs and READMEs link the first mention of external projects and repositories. Do not use internal validation stage codes such as `S0-S2` in reader-facing docs unless that page defines them in place or links directly to the definition.
- Environment-specific documentation must distinguish observed verification conditions from general requirements. Write "the currently verified setup is ..." or equivalent when only one machine/GPU/driver combination has been tested; do not present that setup as the package's inherent training environment.
- Package READMEs reference repository docs with repo-root-relative links such as `docs/training-reproduction.md`, not `../` or `../../` relative escapes.

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

Every model README must include a `### Parity Results` section under `## Evaluation`. Put the vendor-parity summary in a numeric table that states what was compared, the number of cases, the match criterion, and the result. Prose inside `## Reproducibility` is not enough. Put the mechanical walkthrough in `REPRODUCING.md`; reviewers run that file's download, reference-generation, parity, conversion, and smoke-test commands. `.agents/skills/design-generators-model-conversion/references/model-readme-template.md` is the reference format.

The `Reproducibility` section must open with one sentence that states how to reproduce the original-implementation agreement checks. The remaining commands must be copy-pasteable and ordered: download vendor assets, generate vendor references with `CUDA_VISIBLE_DEVICES`, run `pytest -m vendor_parity`, convert checkpoints, and run `from_pretrained` smoke tests.

- Hub model cards are generated through `laygen.common.model_card` using the official Hugging Face model-card template.
- Do not push model weights or Hub repos from ordinary implementation PRs unless the coordinator explicitly asks for publish.

## Machine-checked companions

- The docs site uses `scripts/gen_ref_pages.py` and strict Zensical in `.github/workflows/ci.yml`.
- Model README structure, install commands, parity sections, tagged fences, first external links, and reproducibility commands are checked by `scripts/check_model_readmes.py` and `tests/test_readme_contracts.py`.
- Changed external URLs are checked by `scripts/check_changed_urls.py` in `.github/workflows/ci.yml`, and full Markdown links are checked by `.github/workflows/link-check.yml`.
- Reader-facing internal references are checked by `scripts/check_reader_facing_references.py`.
- Repository-relative README links are checked by `scripts/check_readme_links.py`.
- Keep checker-owned clauses in the machine-checked conventions section of `AGENTS.md` rather than duplicating their full procedures here.

## Validation

Run the documentation and root gates through uv, and run the full pre-commit suite before opening a PR. The PR must use `.github/PULL_REQUEST_TEMPLATE.md` and include the implementation issue, checklist verification, tests, and shared-library rationale when applicable.
