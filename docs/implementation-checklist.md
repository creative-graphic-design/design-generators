---
icon: lucide/list-checks
tags:
  - Contributors
  - Implementation
  - Checklist
---

# Implementation checklist

Use this checklist when implementing or reviewing a model package. For maintenance or documentation, select the affected sections and explain non-applicable evidence in the PR. Complete the applicable items before requesting review, and quote deviations in the pull request description. A workspace member is a package included in the repository's root `uv` workspace.

## Before starting

- [ ] Start new implementation in a task worktree from current `origin/main`, or continue the existing worktree for that task; vendor submodules remain read-only.
- [ ] Use the [roadmap](roadmap.md) for scope, [data sources](data-sources.md) for dataset choices, [conventions](conventions.md) for interfaces/code, and [architecture](architecture.md) for shared ownership. Read the parts affected by the task.
- [ ] Read the model issue's plan comment and every later amendment comment; later amendments override earlier plan text when they conflict.
- [ ] Add the `in-progress` label to the model issue before implementation begins.

## Interface compliance

- [ ] Use the canonical [layout output classes](conventions.md#public-outputs) with their eight aligned fields and framework bases. Keep package-local copies and an `extras` field out; schema tests check that the two explicit dataclasses agree.
- [ ] Return normalized center `xywh` boxes in `[0, 1]`, and represent padding only with `mask` rather than a reserved public label id.
- [ ] Return `id2label` with outputs and persist it in the config and model card; batched open-vocabulary output uses one batch-local union, with per-example maps in `intermediates["id2label_per_example"]`.
- [ ] Make `generator` take precedence over `seed`, and verify that generation is reproducible from the seed or generator.
- [ ] Use canonical `condition_type` names (`unconditional`, `label`, `label_size`, `completion`, `refinement`, `text`, `content_image`, `relation`, `hierarchical`, and `retrieval`); normalize original-implementation aliases before dispatch and raise explicitly for unsupported conditions instead of falling back silently.
- [ ] Expose the full agreed v1 (initial interface) pipeline `__call__` signature and the relevant v2 (later interface) additions even when the model rejects some inputs.
- [ ] Make discrete-vocabulary layout tokenizers subclass [`transformers.PreTrainedTokenizer`](https://huggingface.co/docs/transformers/main_classes/tokenizer), use synthetic token strings and standard `pad_token` and `mask_token` values, expose `encode_layout()` and `decode_layout()` as the primary API, serialize auxiliary data such as cluster centers with tokenizer files, preserve float64 decode paths required for agreement checks, and use a custom class only when a documented conflict requires it.
- [ ] Ensure every [`transformers.PreTrainedModel`](https://huggingface.co/docs/transformers/main_classes/model) subclass implements `forward`; if its computation cannot be represented as one forward pass, compose the stages in the package pipeline instead of using `PreTrainedModel` for the composite.
- [ ] Expose only standard model entry points (`forward` and token-level `generate`) on model classes; put processor encoding, generation, decoding, layout-level orchestration, and the `LayoutGenerationOutput` result in the pipeline's `__call__`, and do not add `generate_layout`-style model methods. Vendor-specific constrained decoding that cannot be expressed as a stateless `LogitsProcessor` may remain as a model-side helper called by the pipeline, but it is not a public generation API.
- [ ] Do not override [`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) or [`save_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) in a way that bypasses standard loading and serialization; document the reason in the pull request description if an override is unavoidable.
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
- [ ] Apply the [model-conversion parity contract](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity): exact deterministic tokens/ids and bitwise floating comparison by default, with a measured numerical justification for any tolerance. Regular tests may skip absent assets; acceptance uses `PARITY_REQUIRE=1` and reports pass/skip counts.
- [ ] Reach at least 90% coverage per package under the CI selection `-m "not vendor_parity and not integration"` with real unit tests such as tiny random-weight CPU configurations; never lower the gate or add broad pragma exclusions.
- [ ] Run root pytest with `--import-mode=importlib` from the root `pyproject.toml` `addopts` setting, and preserve that setting when resolving pyproject merge conflicts because packages share test basenames; adding `tests/__init__.py` does not fix import mode.
- [ ] Keep unit tests independent of weights and network access. CI checks committed lockfile freshness with `uv lock --check`; keep host-specific uv options out of `uv.lock` and use the existing workflow as the command authority.
- [ ] Pass a local `save_pretrained` to `from_pretrained` round-trip test.

## Training for train-ourselves models

Models whose weights this repository trains itself are called train-ourselves models.

- [ ] Use PyTorch Lightning through the `training` extra with [`LightningCLI`](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), YAML configurations, and CLI overrides, and keep the [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and `configs/*.yaml` files in the model package.
- [ ] Follow the [training protocol](training-reproduction.md) and [training document template](templates/TRAINING.template.md) for stage order, launch prerequisites, source/config provenance, per-dataset results, seed scope, and evidence invalidation. Inference parity or a passing document checker is not training reproduction.

## Hub and licensing

- [ ] Name Hugging Face Hub repositories `creative-graphic-design/<model-slug>-<dataset>`, adding a task suffix only for incompatible task-specific checkpoints.
- [ ] Use the method name known in the literature for `<model-slug>`, such as `layoutganpp` for vendor `const-layout`, rather than the vendor repository slug when they differ, and keep the vendor slug in the model card for traceability.
- [ ] Verify the license before uploading weights, and obtain explicit approval for AGPL, GPL, or CC-NC models.
- [ ] Ship a `README.md` for every library package under `lib/laygen`, `lib/posgen`, and future `lib/*` packages that explains its purpose, module map, key API examples, design rules, single-field-spec and no-`extras` constraints, extraction criteria, and links to the [project roadmap](roadmap.md), [shared data sources](data-sources.md), and [shared library architecture](architecture.md).
- [ ] Write READMEs for package users rather than reviewers; omit compliance narration and internal tooling walkthroughs, and state only what users need to know about what exists and how to use it.
- [ ] Open model documentation with an overview, paper link, key idea, and copy-pasteable usage. Consult the relevant official framework documentation when an API or documentation convention is uncertain; there is no required tour of unrelated model pages for every edit.
- [ ] Give every model README a top-level `## Reproducibility` section that opens by stating how to reproduce agreement checks against the original implementation and links to `models/<pkg>/REPRODUCING.md`; that required file contains copy-pasteable commands for download, reference or golden generation with `CUDA_VISIBLE_DEVICES`, `pytest -m vendor_parity`, checkpoint conversion, and `from_pretrained` smoke tests, with prerequisites, cache locations, and expected artifacts. Prose mentions do not satisfy this contract.
- [ ] Take dataset identifiers and canonical condition types from shared enums. Follow [runtime ownership](architecture.md#runtime-ownership) for canonical aliases versus package-specific alias interpretation and validation, and use canonical condition names in Hub task suffixes such as `-label` rather than `-gen-t`.
- [ ] Apply the typing rules in the code and review safeguards below to closed sets such as `box_format`, `condition_type`, `output_type`, sampling modes, and dataset or vocabulary keys, as well as exhaustive branches, module constants, public signatures, and structured specification data.
- [ ] Apply `ruff` docstring rules (`D`) to all `src/` code without adding per-file ignores for `lib/*/src` or `models/*/src`; write the docstrings instead, while keeping test and script exemptions where they already apply.
- [ ] Give public pipelines, tokenizers, processors, configs, `laygen.common` modules, and agents google-style docstrings with `Args`, `Returns`, `Raises`, and runnable doctest-style `Examples`; these docstrings feed the generated API reference.
- [ ] Do not commit machine-specific absolute paths that contain a developer's local checkout directory; resolve script defaults relative to the repository root and provide an explicit CLI override, and use repository-relative paths such as `./vendor/<repo>` in documentation.
- [ ] Give every script under `models/<pkg>/scripts/` a module docstring and an argparse `--help` description for every argument and default, with defaults that work from a clean checkout.
- [ ] Ship every model package README in model-card style with an overview, install and usage snippet using `from_pretrained` or a pipeline call, supported Hub ids, datasets, a numeric original-implementation agreement summary, and the original implementation's license and citation.
- [ ] Give every model README a `### Parity Results` section under `## Evaluation` with a numeric table stating what was compared, the number of cases, the match criterion, and the result. Prose mentions do not satisfy this contract; use `models/layout-dm/README.md` as the reference format.
- [ ] Give every Hub model repository a model card based on the [official Hugging Face model-card template](https://huggingface.co/docs/hub/model-card-annotated) through `huggingface_hub.ModelCard.from_template`, with YAML metadata for `license`, `library_name` (`transformers` or `diffusers`), `pipeline_tag`, `tags` including `layout-generation`, and organization dataset ids, plus model details, intended uses and limitations, a `from_pretrained` example, training data, numeric agreement results, citation BibTeX, and a link to the original implementation.
- [ ] Follow the issue-closure rule in [AGENTS.md](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#issue-and-pr-lifecycle): merge, independent agreement checks, and a local save/load smoke test for every planned checkpoint, dataset, or task repository. Hub publication is separate.

## Process

- [ ] Treat completion as a pull request with green CI or a documented CI blocker; do not self-merge because the user makes the merge decision.
- [ ] Apply the same lane or topic labels as the implementation issue to the pull request, and keep status labels such as `plan-agreed`, `in-progress`, and `parity-verified` on the issue rather than the pull request.
- [ ] Build the pull request description from `.github/PULL_REQUEST_TEMPLATE.md`, keep it as the single current summary of the pull request, and keep progress reports out of pull request comments.
- [ ] Include the pull request URL, checklist verification with deviations, agreement results, and follow-ups in progress reports.

## Code and review safeguards

- [ ] Apply [code conventions](conventions.md#code-style) for closed vocabularies, including shared enums, typed alias tables, and `Literal` for small fixed parameter sets.
- [ ] Use `typing.assert_never` for exhaustive enum dispatch.
- [ ] Annotate module constants with `Final[...]`, including tokens, thresholds, and paths.
- [ ] Annotate every public signature and structured specification; use `NamedTuple` or `TypedDict` for structured tuples and dictionaries, and use `Literal` or enums for closed parameter sets.
- [ ] Do not import private underscore-prefixed modules across modules; make a needed module public instead.
- [ ] Use Hugging Face base classes where they exist, including `PreTrainedTokenizer` for tokenizers and [`transformers.ProcessorMixin`](https://huggingface.co/docs/transformers/main_classes/processors) for processors, rather than hand-written save and load logic.
- [ ] Do not use `*args` or `**kwargs` grab-bag signatures on public APIs; expose explicit keyword-only parameters and reject unsupported call shapes at the signature level.
- [ ] Do not use `sys.path` hacks in tests or conftest files; workspace and package execution must resolve imports.
- [ ] Accept `str` at public boundaries only when the boundary normalizes it immediately to a shared enum; use the enum type internally.
- [ ] Check shared helpers before writing local copies of bbox, label, or serialization utilities, and list extraction candidates for LayoutDM-like logic instead of duplicating it.

## Verification commands

Run the checks for the changed scope once, then rerun affected checks if a fix changes their inputs. Record commands, results, and unavailable checks in the PR. The local hooks and CI remain required; documentation-only work does not require new GPU runs or scientific claims.

For package behavior, use its member environment and the regular test selection:

```bash
uv run --package <member> pytest <member-path>/tests -m "not vendor_parity and not integration"
```

For documentation and instruction changes, the existing root tests check README contracts, docs navigation, and checker behavior:

```bash
uv run --package design-generators --group dev pytest tests -m "not integration"
uv run --package design-generators devharness check model-readmes
uv run --package design-generators scripts/check_training_doc_template.py
uv run --package design-generators scripts/check_training_stage_evidence.py
```

The strict site build needs installed workspace packages for API imports. Preserve that environment for the build:

```bash
uv sync --all-packages --group docs
uv run --no-sync --package design-generators zensical build --strict -f mkdocs.yml
```

Run the repository's pre-commit suite in the full-workspace environment above before opening the PR; its member-test hooks expect installed packages. The documented `uv-lock` exclusion prevents a validation run from rewriting the lockfile; dependency changes still require a deliberate lock update and CI freshness check.

```bash
SKIP=uv-lock uv run --no-sync --package design-generators pre-commit run --all-files
```

Use the package's `REPRODUCING.md` or `TRAINING.md` for asset-dependent acceptance commands. An independent reviewer reports actual comparisons with `PARITY_REQUIRE=1`, not an all-skip result.
