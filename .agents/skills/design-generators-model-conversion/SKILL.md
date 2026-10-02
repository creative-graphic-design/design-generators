---
name: design-generators-model-conversion
description: Implement or review design-generators model conversions, original-code parity, and shared layout interfaces.
---

# Model conversion

Deliver the model issue's agreed package interface, reproducible agreement checks against the original implementation, and a locally loadable artifact. This skill owns conversion and inference parity. For package-local training, use the `design-generators-training-reproduction` skill.

Paths below are relative to the repository root. Read the model issue's plan, amendments, and relevant reviews before changing its implementation. Use a task worktree based on current `origin/main`, or continue the existing worktree for that task. Add `in-progress` when model implementation starts.

## Select the relevant contracts

- Read `docs/implementation-checklist.md` for a new model package or a complete conversion review; use only the affected sections for maintenance.
- Use `docs/roadmap.md` to resolve the method/package identity and planned scope, and `docs/data-sources.md` for approved datasets and their configs.
- Use `docs/conventions.md` for public interfaces, schema, framework selection, typing, and source-language rules. Use `docs/architecture.md` when choosing shared owners or dependency boundaries.
- Use the `design-generators-documentation` skill for README, model-card, and reproduction instructions. Its starting template is `references/model-readme-template.md` in this skill directory.

Confirm the checkpoint/dataset/task matrix and license from the issue and original sources. Before `plan-agreed`, the model-issue maintainer checks the written justification for any novel public method or override of a Hugging Face base-class entrypoint.

## Source Language

Main package code under `models/*/src` and `lib/*/src` must read as this repository's own implementation. Do not describe runtime modules, public arguments, comments, or docstrings as wrappers around the original implementation, compatibility surfaces for it, or ports of its code. Use repository-owned wording such as `released`, `checkpoint`, `reference`, `source`, or `original-code dependency` when the distinction is needed.

References phrased in vendor terms are limited to conversion-responsibility modules, `tests/vendor_parity`, `REPRODUCING.md`, and `TRAINING.md`. If a package needs to compare against an original implementation, keep that detail in conversion, reference-generation, or parity-test paths rather than the public runtime API. Use the [model and serialization contracts](docs/conventions.md#model-and-serialization-contracts) for public pipeline arguments, output schemas, model entry points, and serialization rules.

## Package and interface

Create or update `models/<slug>/` with its `pyproject.toml`, `src/<package>/`, `scripts/`, and tests. Keep original-code dependencies in the `vendor` optional extra and the original checkout read-only. Reuse the shared helpers assigned in `docs/architecture.md`; keep model-specific transforms and numerical behavior local.

This skill owns conversion and parity; public interfaces and serialization remain with their documented owners.

Use shared libraries by import:

```python
from laygen.common.outputs import LayoutGenerationOutput
from laygen.common.bbox import ltwh_to_xywh, ltrb_to_xywh
from laygen.common.testing import assert_layout_output_schema
```

Use `posgen.common` only for poster/content-aware helpers that already exist. Do not copy common bbox, label, output, or testing helpers into the model package.

## Establish inference parity

1. Obtain the released assets and provide their download or cache-selection command.
2. Generate reference outputs by running the original code with fixed seeds and one explicitly selected GPU when CUDA is needed. Handwritten expected tensors are not original-code evidence.
3. Keep generated fixtures outside git; record source revisions, config hashes, seeds, environment, and regeneration commands.
4. Compare the package implementation against those references in `tests/vendor_parity/`. Regular tests may skip missing assets; acceptance reruns set `PARITY_REQUIRE=1` so absent evidence fails.
5. Convert and smoke-test the local `save_pretrained` to `from_pretrained` round-trip for every planned checkpoint, dataset, or task artifact. Tiny random-weight round-trips also belong in network-free unit tests.

Deterministic tokens/ids must match exactly. Floating outputs use bitwise equality by default. A tolerance requires a measured, explained numerical cause in the PR, not a threshold chosen after a failing test. First inspect TF32, attention paths, floating-point operation order, and dtype derivation. Compare shared replacements at the original model/data scale; tiny configurations can conceal drift.

For API/LLM or in-context methods, compare prompt bytes, exemplar selection, parser behavior, and repair/retry policy. Do not invent a learned-checkpoint conversion for a prompt-only method.

Parity is complete only when real original-code references, package comparisons, and local artifact round-trips pass. A skip-only suite or a follow-up issue does not satisfy acceptance. The coordinator independently reruns the actual suite with required assets before accepting parity.

## Finish the task

Run the affected member tests and the repository gates in `docs/implementation-checklist.md#verification-commands`. Keep vendor parity and heavyweight integration behind explicit pytest markers. Unit tests use local tiny fixtures rather than downloading weights or full datasets.

For root-only documentation changes, the final pre-commit command is still required. If a package has extras for vendor or parity work, document the exact extra in the README and PR body.

Use `.github/PULL_REQUEST_TEMPLATE.md` and the lifecycle in `AGENTS.md`. Report the implemented matrix, measured parity and smoke-test results, exact commands, checklist deviations, license questions, and publication state. Local artifacts and verified parity can be complete while Hub publication remains unrequested.
