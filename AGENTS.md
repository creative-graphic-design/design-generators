> [!NOTE]
> After reading this AGENTS.md, say: 🤖 I read the AGENTS.md for creative-graphic-design/design-generators.

# Agent Instructions

## Project

`design-generators` ports layout, poster, and graphic-design generation research
repositories into Transformers/Diffusers-style packages that can load converted
weights with `from_pretrained` and run inference immediately. Keep repository
rules here stable and concise; procedural implementation guidance lives in
repo-local skills such as `.agents/skills/model-conversion/SKILL.md`.

## Sources Of Truth

- Umbrella policy, target table, execution order, interface decisions, data
  policy, and status tracking live in
  [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2).
- The living implementation checklist is
  [issue #60 (implementation checklist)](https://github.com/creative-graphic-design/design-generators/issues/60).
  Check it before starting a model package and quote verification results in
  the PR body.
- Shared library structure is
  [issue #64 (shared library structure)](https://github.com/creative-graphic-design/design-generators/issues/64):
  workspace members are `lib/*` and `models/*`; shared layout helpers import
  from `laygen.common`; poster helpers import from `posgen.common` when needed.
- A model issue's plan comment plus all later amendment comments define that
  model's design. Amendments override earlier plan text.

## Workspace

- The root uv workspace uses members `["lib/*", "models/*"]`.
- Run member-specific commands with the member package selected:
  `uv run --package <name> ...`. Examples: `uv run --package laygen pytest`,
  `uv run --package layout-dm pytest`.
- Do not run plain root `uv run` against a member path when the command depends
  on that member's extras, dependency source mapping, or package metadata.
- Keep original implementations under `vendor/` read-only. Isolate their
  dependencies behind a model package's `vendor` optional extra.

## Repo-Local Skills

- Skill source of truth is `.agents/skills/<name>/SKILL.md`.
- `.claude/skills` is a relative symlink to `.agents/skills` for Claude
  compatibility.
- Codex should read the relevant `.agents/skills/<name>/SKILL.md` directly
  when a repo-local skill applies.

## Naming

- Model package names and Hub repo ids use the method name known in the
  literature, not necessarily the vendor repository slug.
- Example: vendor `const-layout` becomes package `layoutganpp` and Hub ids such
  as `creative-graphic-design/layoutganpp-rico`.
- Shared packages are `laygen.common` for layout-generation utilities and
  `posgen.common` for poster/content-aware utilities.
- Core model-package modules under `models/*/src/<pkg>/` must follow Hugging
  Face-style filenames with the package suffix, such as
  `configuration_<pkg>.py`, `modeling_<pkg>.py`, `pipeline_<pkg>.py`,
  `scheduling_<pkg>.py`, `processing_<pkg>.py`, `tokenization_<pkg>.py`,
  `image_processing_<pkg>.py`, and `generation_<pkg>.py`. Repository convention
  files and domain helpers must be explicitly allowed by
  `scripts/check_module_naming.py`; do not add new ad hoc core names.

## Tracking

- If the user asks to create or file an issue first, treat issue creation as a
  hard gate: create the issue and record its URL before implementation, commits,
  or PR work. Do not substitute a PR for the requested issue.
- Every implementation PR must reference its implementation issue in the PR
  summary with `Closes #N` or `Refs #N`. The standing checklist issue #60 does
  not count as the implementation issue.
- Every PR must carry the same lane/topic labels as its implementation issue,
  such as `ready-heavy`, `documentation`, or `meta`; status labels stay on
  issues only and must not be added to PRs.
- Priority labels select the work lane; status labels move in this order:
  `plan-agreed` -> `in-progress` -> `parity-verified` -> published/closed.
- When creating any issue, set both the milestone and the native Priority issue
  field; do not leave either unset. Set the native Priority field through
  GraphQL `setIssueFieldValue` when the CLI surface is insufficient.
- Add `in-progress` when work on a model issue begins.
- Add `parity-verified` only after the coordinator independently reruns the
  parity suite and confirms the results.
- Close a model issue only after every planned Hub repo for that issue has a
  passing `from_pretrained` smoke test.
- Milestones are execution phases: `v0.1` foundation and pilot wave, `v0.2`
  ready-light completion, `v0.3` ready-heavy, `v0.4` LLM recipes and Pydantic AI,
  and `v0.5` train-ourselves.

## Self-Improvement

- Treat `AGENTS.md`, repo-local skills, PR templates, and checklist issues as
  living documents. When work exposes incorrect, stale, or missing guidance,
  fix it in the same PR if the change is small and in scope; otherwise open or
  propose a focused `meta` issue.
- Do not silently work around guidance known to be wrong. If the same kind of
  mistake is raised repeatedly, add or revise a rule, template item, or check so
  future work can catch it mechanically.

## Public Interface

- Return the common layout schema: `bbox`, `labels`, `mask`, and `id2label`,
  with optional `sequences`, `scores`, `trajectory`, and `intermediates`.
- Transformers models return `laygen.modeling_outputs.LayoutGenerationOutput`;
  Diffusers pipelines return `laygen.pipelines.pipeline_output.LayoutGenerationOutput`.
  Both are explicit dataclasses and schema tests assert field names, order, and
  defaults stay aligned.
- Transformers-side layout pipelines subclass
  `laygen.pipelines.LayoutGenerationPipeline`; plain classes and
  `transformers.Pipeline` subclasses are non-conforming.
- Public `bbox` is normalized center `xywh` in `[0, 1]`, regardless of vendor
  internals such as `ltwh`, `ltrb`, bins, analog bits, or text tokens.
- Public `mask=True` means a valid element. Padding is represented by `mask`,
  never by reserving public label id `0`.
- `labels` are dataset-local integer ids unless a model explicitly documents
  request-local open-vocabulary ids.
- Persist `id2label` in config/model cards and return it with outputs.
- `generator` is the exact reproducibility API and takes precedence over
  `seed`.
- Canonical `condition_type` names are v1 `unconditional`, `label`,
  `label_size`, `completion`, `refinement`; v2 adds `text`, `content_image`,
  `relation`, `hierarchical`, `retrieval`. Normalize vendor aliases and raise
  explicit errors for unsupported modes.
- Discrete-vocabulary layout tokenizers subclass
  `transformers.PreTrainedTokenizer`; serialize auxiliary data with tokenizer
  files. Use a custom class only when the base class truly conflicts and document
  the reason.
- Store hierarchy, retrieval, open-vocabulary, attention, and other auxiliary
  data in `intermediates`; do not add an `extras` field.

## Parity

- Golden parity fixtures are generated by running vendor code, not handwritten.
- Do not commit golden tensors, images, weights, or large downloaded artifacts.
  Commit only metadata needed to regenerate them: seeds, conditions, environment
  notes, config hashes, and script arguments.
- Run parity generation on one explicitly selected GPU with fixed seeds.
- Coordinator parity reruns must set `PARITY_REQUIRE=1` so missing local
  assets fail loudly instead of turning an all-skip run into an apparent
  success.
- For LLM API or in-context methods, parity means prompt bytes, exemplar
  selection, parser behavior, and repair policy match the paper/vendor path.
- Keep vendor-parity and heavyweight integration tests outside regular CI and
  behind explicit pytest markers.

## Data

- Prefer datasets hosted by the `creative-graphic-design` Hugging Face org.
  Check
  [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2)
  before adding a new data source.
- Use `creative-graphic-design/Rico` with
  `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` for RICO25;
  the default config is metadata-only. RICO13 needs a vendor-derived mapping.
- PubLayNet is `creative-graphic-design/PubLayNet`; avoid any test path that
  could download the full dataset.
- Crello uses `cyberagent/crello` as the canonical source until an org mirror
  exists; `creative-graphic-design/Desigen` is not a Crello substitute.
- Respect pinned dataset quirks from
  [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2)
  and
  [issue #60 (implementation checklist)](https://github.com/creative-graphic-design/design-generators/issues/60):
  Magazine is polygon-based and train-only, PKU has an `INVALID` class and pixel
  `ltrb` boxes, and CGL-v2 needs `ralf-style` for validation/saliency use cases.

## Documentation

- The docs site uses `zensical`, `mkdocstrings[python]`, and generated API
  pages; build it with:
  ```bash
  uv run --group docs python scripts/gen_ref_pages.py
  uv run --group docs zensical build --strict -f mkdocs.generated.yml
  ```
- The API reference is generated from workspace members under `lib/*` and
  `models/*`, using Python packages found below each member's `src/` directory.
- Public API docstrings are the source text for the API reference. Use
  google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples`
  sections for public pipelines, tokenizers, processors, configs,
  `laygen.common` modules, `posgen.common` modules, and agents.
- `Examples` in public API docstrings should be doctest-ready snippets whenever
  the API can run without heavyweight assets, downloads, or credentials.
- Each model package README uses a model-card style: overview, install/usage
  snippet, supported checkpoints/Hub ids, datasets, reproducibility summary
  with vendor-parity numbers, license, citation, and original implementation
  link.
- Each README includes `Reproducibility`, opening with one sentence that states
  how to reproduce the original-implementation agreement checks, followed by
  copy-pasteable commands for download, vendor reference generation, parity
  tests, conversion, and `from_pretrained` smoke tests.
- Markdown code fences must be tagged. Use `bash` for executable shell commands
  and `text` for non-executable output, logs, or examples.
- Hub model cards are generated through `laygen.common.model_card` using the
  official Hugging Face model-card template.

## Training

- Train-ourselves models use PyTorch Lightning through the root `training` extra
  and LightningCLI with YAML configs plus CLI overrides.
- Keep `LightningModule`, `LightningDataModule`, and `configs/*.yaml` inside the
  model package.

## CI

- CI resolves workspace members with `uv sync --all-packages`, then runs
  pre-commit with `SKIP=uv-lock`, `ty`, root tests, and workspace-member tests
  that are not marked `vendor_parity` or `integration`.
- CI runs root pytest without coverage because the root has no import package;
  each workspace member is measured separately and coverage is not combined
  across members.
- Coverage has a 90% floor for every workspace member. Do not lower
  `fail_under` below 90; member-specific overrides may only raise the floor.
- Do not add `uv lock --check` or uv-lock to CI; local uv global options are
  intentionally baked into `uv.lock` in this environment.
