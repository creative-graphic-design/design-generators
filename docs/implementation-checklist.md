---
icon: lucide/list-checks
tags:
  - Contributors
  - Implementation
  - Checklist
---

# Implementation checklist

Use this checklist when implementing a model package. Complete each applicable item before requesting review, and quote any deviation in the pull request description. A workspace member is a package included in the repository's root `uv` workspace.

## Before starting

- [ ] Create a fresh worktree from the current `origin/main`; local checkouts may be stale, and vendor submodules must remain untouched.
- [ ] Read the [project roadmap](roadmap.md), [shared data sources](data-sources.md), [public conventions](conventions.md), and [shared-library architecture](architecture.md) before applying the current repository contracts; historical umbrella discussion remains preserved in [issue 2](https://github.com/creative-graphic-design/design-generators/issues/2).
- [ ] Read the model issue's plan comment and every later amendment comment; later amendments override earlier plan text when they conflict.
- [ ] Add the `in-progress` label to the model issue before implementation begins.

## Interface compliance

- [ ] Import `LayoutGenerationOutput` from `laygen.modeling_outputs` for Transformers models and from `laygen.pipelines.pipeline_output` for Diffusers pipelines, preserve all eight fields (`bbox`, `labels`, `mask`, `id2label`, `sequences`, `scores`, `trajectory`, and `intermediates`), use the canonical `transformers.utils.ModelOutput`-based class for Transformers models, keep Diffusers output based on `diffusers.utils.BaseOutput`, build both from one field specification, keep no local copies, do not add an `extras` field, and put auxiliary data in `intermediates`.
- [ ] Return normalized center `xywh` boxes in `[0, 1]`, and represent padding only with `mask` rather than a reserved public label id.
- [ ] Return `id2label` with outputs and persist it in the config and model card; batched open-vocabulary output uses one batch-local union, with per-example maps in `intermediates["id2label_per_example"]`.
- [ ] Make `generator` take precedence over `seed`, and verify that generation is reproducible from the seed or generator.
- [ ] Use canonical `condition_type` names (`unconditional`, `label`, `label_size`, `completion`, `refinement`, `text`, `content_image`, `relation`, `hierarchical`, and `retrieval`); normalize original-implementation aliases before dispatch and raise explicitly for unsupported conditions instead of falling back silently.
- [ ] Expose the full agreed v1 (initial interface) pipeline `__call__` signature and the relevant v2 (later interface) additions even when the model rejects some inputs.
- [ ] Make discrete-vocabulary layout tokenizers subclass `transformers.PreTrainedTokenizer`, use synthetic token strings and standard `pad_token` and `mask_token` values, expose `encode_layout()` and `decode_layout()` as the primary API, serialize auxiliary data such as cluster centers with tokenizer files, preserve float64 decode paths required for agreement checks, and use a custom class only when a documented conflict requires it.
- [ ] Ensure every `transformers.PreTrainedModel` subclass implements `forward`; if its computation cannot be represented as one forward pass, compose the stages in the package pipeline instead of using `PreTrainedModel` for the composite.
- [ ] Expose only standard model entry points (`forward` and token-level `generate`) on model classes; put processor encoding, generation, decoding, layout-level orchestration, and the `LayoutGenerationOutput` result in the pipeline's `__call__`, and do not add `generate_layout`-style model methods. Vendor-specific constrained decoding that cannot be expressed as a stateless `LogitsProcessor` may remain as a model-side helper called by the pipeline, but it is not a public generation API.
- [ ] Do not override `from_pretrained` or `save_pretrained` in a way that bypasses standard loading and serialization; document the reason in the pull request description if an override is unavoidable.
- [ ] Use upstream class suffixes only when the class satisfies the upstream contract; for example, `ForConditionalGeneration` requires seq2seq-style `forward` and `generate` methods.
- [ ] Before applying `plan-agreed`, document and justify any novel public method or override on a Hugging Face base class, and have the coordinator, meaning the maintainer who owns the model issue and is distinct from the evidence producer, check that justification.
- [ ] Make Transformers-side layout pipelines subclass `laygen.pipelines.LayoutGenerationPipeline` rather than `transformers.Pipeline`; the shared base owns config and subfolder loading, serialization, device and dtype handling, `generator`-over-`seed` behavior, and the canonical layout-output contract.

## Package layout

- [ ] Create `models/<slug>/` as a `uv` workspace member with its own `pyproject.toml`, `src/<pkg>/`, `scripts/`, `tests/`, and `tests/vendor_parity/` directories.
- [ ] Isolate original-implementation dependencies in the package's `vendor` optional extra so the package itself stays light.
- [ ] Put shared logic in `lib/laygen` (`laygen.common`) or poster-side `lib/posgen` (`posgen.common`) and import it rather than copying it between model packages.
- [ ] Run workspace-member commands with `uv run --package <member-name> ...`, such as `uv run --package layout-dm pytest` or `uv run --package laygen pytest`, so the member's dependencies and extras resolve instead of running plain root `uv run` against a member path.

## Data

- [ ] Use organization datasets under `creative-graphic-design/*` as the primary source, and put all loading behind the processor.
- [ ] Respect the pinned dataset configurations: Rico uses `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` because the default configuration is metadata-only; RICO13 needs a vendor-derived mapping, PKU filters or reserves `INVALID` and uses pixel `ltrb` boxes, Magazine converts polygons to boxes and is train-only, and CGL-v2 uses `ralf-style` for validation and saliency.
- [ ] Keep tests from triggering large dataset downloads; PubLayNet is about 107 GB, so use builders, streaming, synthetic rows, or tiny local fixtures.
- [ ] When an organization dataset is missing, use the original implementation's dataset source and record the migration TODO in the model issue.

## Parity and tests

- [ ] Regenerate golden fixtures with the reference-generation script on one explicitly selected GPU and fixed seeds with `CUDA_VISIBLE_DEVICES` set to one free GPU; never commit the fixtures, and commit only seeds, environment notes, config hashes, and script arguments needed to regenerate them.
- [ ] Require exact token or id matches for deterministic generation and tolerance-based comparison only for logits, gate parity tests behind a pytest marker, and skip them cleanly when weights are absent.
- [ ] Reach at least 90% coverage per package under the CI selection `-m "not vendor_parity and not integration"` with real unit tests such as tiny random-weight CPU configurations; never lower the gate or add broad pragma exclusions.
- [ ] Run root pytest with `--import-mode=importlib` from the root `pyproject.toml` `addopts` setting, and preserve that setting when resolving pyproject merge conflicts because packages share test basenames; adding `tests/__init__.py` does not fix import mode.
- [ ] Keep unit tests independent of weights and network access, and do not add `uv lock --check` to CI because the environment uses specific lock options.
- [ ] Pass a local `save_pretrained` to `from_pretrained` round-trip test.

## Training for train-ourselves models

Models whose weights this repository trains itself are called train-ourselves models.

- [ ] Use PyTorch Lightning through the `training` extra with LightningCLI, YAML configurations, and CLI overrides, and keep the `LightningModule`, `LightningDataModule`, and `configs/*.yaml` files in the model package.

## Hub and licensing

- [ ] Name Hugging Face Hub repositories `creative-graphic-design/<model-slug>-<dataset>`, adding a task suffix only for incompatible task-specific checkpoints.
- [ ] Use the method name known in the literature for `<model-slug>`, such as `layoutganpp` for vendor `const-layout`, rather than the vendor repository slug when they differ, and keep the vendor slug in the model card for traceability.
- [ ] Verify the license before uploading weights, and obtain explicit approval for AGPL, GPL, or CC-NC models.
- [ ] Ship a `README.md` for every library package under `lib/laygen`, `lib/posgen`, and future `lib/*` packages that explains its purpose, module map, key API examples, design rules, single-field-spec and no-`extras` constraints, extraction criteria, and links to the [project roadmap](roadmap.md), [shared data sources](data-sources.md), and [shared library architecture](architecture.md).
- [ ] Write READMEs for package users rather than reviewers; omit compliance narration and internal tooling walkthroughs, and state only what users need to know about what exists and how to use it.
- [ ] Before writing or reviewing model documentation, read the [Transformers contributing guide](https://huggingface.co/docs/transformers/en/contributing), [modular Transformers](https://huggingface.co/docs/transformers/en/modular_transformers), and the usage-first [DETR](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/detr), [LayoutLMv3](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/layoutlmv3), [LLaVA](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/llava), [Llama](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/llama), [GPT-2](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/gpt2), [T5](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/t5), and [BERT](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/bert) and [ViT](https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/vit) model pages; each model README should open with a short overview, paper link, key idea, and early copy-pasteable usage example before tips, limitations, and reference material.
- [ ] Give every model README a top-level `## Reproducibility` section that opens by stating how to reproduce agreement checks against the original implementation and links to `models/<pkg>/REPRODUCING.md`; that required file contains copy-pasteable commands for download, reference or golden generation with `CUDA_VISIBLE_DEVICES`, `pytest -m vendor_parity`, checkpoint conversion, and `from_pretrained` smoke tests, with prerequisites, cache locations, and expected artifacts. Prose mentions do not satisfy this contract.
- [ ] Take dataset identifiers and condition types from shared enums: `laygen.common.DatasetName` for layout datasets, `posgen.common.DatasetName` for poster and content datasets, and `laygen.common.ConditionType` with the central original-implementation alias resolver; do not redefine package-local enums or alias tables, and use canonical condition names in Hub task suffixes such as `-label` rather than `-gen-t`.
- [ ] Apply the typing rules in the code and review safeguards below to closed sets such as `box_format`, `condition_type`, `output_type`, sampling modes, and dataset or vocabulary keys, as well as exhaustive branches, module constants, public signatures, and structured specification data.
- [ ] Apply `ruff` docstring rules (`D`) to all `src/` code without adding per-file ignores for `lib/*/src` or `models/*/src`; write the docstrings instead, while keeping test and script exemptions where they already apply.
- [ ] Give public pipelines, tokenizers, processors, configs, `laygen.common` modules, and agents google-style docstrings with `Args`, `Returns`, `Raises`, and runnable doctest-style `Examples`; these docstrings feed the generated API reference.
- [ ] Do not commit machine-specific absolute paths that contain a developer's local checkout directory; resolve script defaults relative to the repository root and provide an explicit CLI override, and use repository-relative paths such as `./vendor/<repo>` in documentation.
- [ ] Give every script under `models/<pkg>/scripts/` a module docstring and an argparse `--help` description for every argument and default, with defaults that work from a clean checkout.
- [ ] Ship every model package README in model-card style with an overview, install and usage snippet using `from_pretrained` or a pipeline call, supported Hub ids, datasets, a numeric original-implementation agreement summary, and the original implementation's license and citation.
- [ ] Give every model README a `### Parity Results` section under `## Evaluation` with a numeric table stating what was compared, the number of cases, the match criterion, and the result. Prose mentions do not satisfy this contract; use `models/layout-dm/README.md` as the reference format.
- [ ] Give every Hub model repository a model card based on the [official Hugging Face model-card template](https://huggingface.co/docs/hub/model-card-annotated) through `huggingface_hub.ModelCard.from_template`, with YAML metadata for `license`, `library_name` (`transformers` or `diffusers`), `pipeline_tag`, `tags` including `layout-generation`, and organization dataset ids, plus model details, intended uses and limitations, a `from_pretrained` example, training data, numeric agreement results, citation BibTeX, and a link to the original implementation.
- [ ] Close a model issue only after the implementation is merged to `main`, agreement has been independently verified, and a local `save_pretrained` to `from_pretrained` smoke test passes; keep Hub publishing separate from implementation PRs.

## Process

- [ ] Treat completion as a pull request with green CI or a documented CI blocker; do not self-merge because the user makes the merge decision.
- [ ] Apply the same lane or topic labels as the implementation issue to the pull request, and keep status labels such as `plan-agreed`, `in-progress`, and `parity-verified` on the issue rather than the pull request.
- [ ] Build the pull request description from `.github/PULL_REQUEST_TEMPLATE.md`, keep it as the single current summary of the pull request, and keep progress reports out of pull request comments.
- [ ] Include the pull request URL, checklist verification with deviations, agreement results, and follow-ups in progress reports.

## Code and review safeguards

- [ ] Use `StrEnum` with `auto()` for closed string sets rather than bare `str`, `tuple[str, ...]`, or string-literal unions; type alias tables as `dict[SomeAliasEnum, SomeEnum]`, and use shared dataset and condition enums instead of local redefinitions.
- [ ] Use `typing.assert_never` for exhaustive enum dispatch.
- [ ] Annotate module constants with `Final[...]`, including tokens, thresholds, and paths.
- [ ] Annotate every public signature and structured specification; use `NamedTuple` or `TypedDict` for structured tuples and dictionaries, and use `Literal` or enums for closed parameter sets.
- [ ] Do not import private underscore-prefixed modules across modules; make a needed module public instead.
- [ ] Use Hugging Face base classes where they exist, including `PreTrainedTokenizer` for tokenizers and `transformers.ProcessorMixin` for processors, rather than hand-written save and load logic.
- [ ] Do not use `*args` or `**kwargs` grab-bag signatures on public APIs; expose explicit keyword-only parameters and reject unsupported call shapes at the signature level.
- [ ] Do not use `sys.path` hacks in tests or conftest files; workspace and package execution must resolve imports.
- [ ] Accept `str` at public boundaries only when the boundary normalizes it immediately to a shared enum; use the enum type internally.
- [ ] Check shared helpers before writing local copies of bbox, label, or serialization utilities, and list extraction candidates for LayoutDM-like logic instead of duplicating it.

## Maintainer confirmation needed

- [ ] Confirm the model-issue closure rule before closing model issues: the source checklist text requires merge to `main`, independently verified agreement, and a passing local round-trip, while current repository guidance additionally requires a passing `from_pretrained` smoke test for every planned Hub repository.
