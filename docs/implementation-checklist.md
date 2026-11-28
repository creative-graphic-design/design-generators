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

- [ ] Use the [roadmap](roadmap.md) for scope, [data sources](data-sources.md) for dataset choices, [conventions](conventions.md) for interfaces/code, and [architecture](architecture.md) for shared ownership. Read the parts affected by the task.
- [ ] Apply the model issue plan and amendments according to the [repository rule](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#sources-of-truth).
- [ ] Apply issue status labels according to [repository tracking rules](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#tracking).

## Interface compliance

- [ ] Verify output classes, fields, coordinates, labels, conditions, reproducibility, and pipeline arguments against the [public interface conventions](conventions.md#user-interface).
- [ ] Verify tokenizers, model entry points, serialization, and config handling against the [model and serialization rules](conventions.md#model-and-serialization-rules).
- [ ] Before applying `plan-agreed`, follow the coordinator review rule in the [model-conversion skill](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#select-the-relevant-rules).

## Package layout

- [ ] Apply the [model framework](conventions.md#choosing-a-model-framework) and [module-name](conventions.md#model-package-module-names) conventions to packages added or maintained here.
- [ ] Create `models/<slug>/` as a `uv` workspace member with its own `pyproject.toml`, `src/<pkg>/`, `scripts/`, `tests/`, and `tests/vendor_parity/` directories.
- [ ] Follow the [model-conversion package rules](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#package-and-interface) for original-implementation dependencies.
- [ ] Put shared logic in `lib/laygen` (`laygen.common`) or poster-side `lib/posgen` (`posgen.common`) and import it rather than copying it between model packages.
- [ ] Select the member environment for member-specific commands as described in [Workspace](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#workspace).

## Data

- [ ] Follow the approved sources and dataset policy in [Shared Data Sources](data-sources.md); isolate loading behind the package processor.

## Parity and tests

- [ ] Follow [parity rules](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity).
- [ ] Verify coverage against the [repository CI floor](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#pr-and-ci-gates).
- [ ] Run root pytest with `--import-mode=importlib` from the root `pyproject.toml` `addopts` setting, and preserve that setting when resolving pyproject merge conflicts because packages share test basenames; adding `tests/__init__.py` does not fix import mode.
- [ ] Keep unit tests independent of weights and network access.
- [ ] Pass a local `save_pretrained` to `from_pretrained` round-trip test.

## Training for train-ourselves models

Models whose weights this repository trains itself are called train-ourselves models.

- [ ] Follow [training reproduction guidance](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-training-reproduction/SKILL.md).
- [ ] Follow the [training protocol](training-reproduction.md) and [training document template](templates/TRAINING.template.md) for stage order, launch prerequisites, source/config provenance, per-dataset results, seed scope, and evidence invalidation. Inference parity or a passing document checker is not training reproduction.

## Hub and licensing

- [ ] Name Hugging Face Hub repositories `creative-graphic-design/<model-slug>-<dataset>`, adding a task suffix only for incompatible task-specific checkpoints.
- [ ] Follow the [repository naming convention](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#naming) for model package names and Hub IDs.
- [ ] Verify the license before uploading weights, and obtain explicit approval for AGPL, GPL, or CC-NC models.
- [ ] Give each `lib/*` package a README describing its purpose, modules, API examples, design rules, extraction criteria, and relevant project guidance. The `laygen` and `posgen` READMEs also document their layout-schema and no-`extras` contracts.
- [ ] Follow [model README guidance](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#model-readme-and-model-card).
- [ ] Take dataset identifiers and canonical condition types from shared enums. Follow [runtime ownership](architecture.md#runtime-ownership) for canonical aliases versus package-specific alias interpretation and validation, and use canonical condition names in Hub task suffixes such as `-label` rather than `-gen-t`.
- [ ] Apply the typing rules in the code and review safeguards below to closed sets such as `box_format`, `condition_type`, `output_type`, sampling modes, and dataset or vocabulary keys, as well as exhaustive branches, module constants, public signatures, and structured specification data.
- [ ] Apply `ruff` docstring rules (`D`) to all `src/` code without adding per-file ignores for `lib/*/src` or `models/*/src`; write the docstrings instead, while keeping test and script exemptions where they already apply.
- [ ] Follow the [documentation skill's public API docstring guidance](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#api-and-docs-site).
- [ ] Follow the [repository path rule](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#workspace) for committed paths and runtime path overrides.
- [ ] Give every script under `models/<pkg>/scripts/` a module docstring and an argparse `--help` description for every argument and default, with defaults that work from a clean checkout.
- [ ] Set model README Hub status using [Hub publication ownership](architecture.md#hub-publication-ownership).
- [ ] Follow [checkpoint-card guidance](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/references/model-readme-template.md).
- [ ] Close model issues according to [repository tracking rules](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#status-labels-and-milestones).

## Process

- [ ] Treat completion as a pull request with green CI or a documented CI blocker; do not self-merge because the user makes the merge decision.
- [ ] Apply [repository issue and pull-request tracking rules](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#tracking).
- [ ] Use `.github/PULL_REQUEST_TEMPLATE.md` when preparing the pull request.
- [ ] Include the pull request URL, checklist verification with deviations, agreement results, and follow-ups in progress reports.
- [ ] Use `[x]` only for verified items; unfinished or non-applicable items may remain unchecked.

## Code and review safeguards

- [ ] Follow [Class Design and Code Style](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#class-design-and-code-style).
- [ ] Prefer `StrEnum` with `auto()` for runtime closed sets; `Literal` is also accepted for static parameter sets. Type alias tables as `dict[SomeAliasEnum, SomeEnum]` and use shared dataset and condition enums.
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

Run the checks for the changed scope once, then rerun affected checks if a fix changes their inputs. Record commands, results, and unavailable checks in the PR. Use the repository's [dependency environments](architecture.md#dependency-environments) for package, root-tooling, full-workspace, and documentation command selection.

- Run affected member tests with `uv run --package <member> pytest`.
- Run the full hook set with `uv run --locked pre-commit run --all-files`.
- Build the documentation site strictly with `uv run --locked --group docs zensical build --strict -f mkdocs.yml`.

Use the package's `REPRODUCING.md` or `TRAINING.md` for asset-dependent commands. Follow the [model-conversion acceptance rule](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity) for reruns and pass criteria.
