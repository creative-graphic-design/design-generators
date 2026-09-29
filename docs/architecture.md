---
icon: lucide/boxes
tags:
  - Architecture
  - Contributors
  - Shared Libraries
---

# Shared library architecture

The repository keeps reusable layout and poster-generation code in shared workspace libraries, while model packages keep model-specific behavior local.

## Workspace layout

A workspace member is a package included in the root `uv` workspace, so contributors can develop it alongside the model packages that consume it.

The workspace contains shared libraries under `lib/*` and model packages under `models/*`.

```text
lib/
  laygen/
  posgen/
  traingen/
  traingen-parity/
models/
  <model-package>/
```

The root workspace declaration is:

```toml
[tool.uv.workspace]
members = ["lib/*", "models/*"]
```

The `lib/` and `models/` directories are separate ownership boundaries: shared libraries provide reusable contracts and helpers, while a model package owns its model-specific processing, inference, conversion, training, and configuration code.

## Dependency environments

A member-scoped environment installs one selected workspace package with `uv sync --package <name>` and is the environment used by the member test matrix for package-local development and verification. Member environments resolve package dependencies through the workspace source mappings.

A root-tooling environment installs the root package and its tooling group with `uv sync --package design-generators --group dev`, and evaluation-only commands add the root `evaluation` extra when they run `scripts/verify_evaluate_layout_metrics.py`. The root project installs no runtime dependencies by default; it owns the workspace-wide `transformers` and `diffusers` version floors and the `dev` and `docs` tooling groups, while workspace members own their runtime dependencies.

| Surface                  | Owner                                    | Consumer or contract                                                                                      | Why root owns it                                                                                                        |
| ------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Workspace version policy | Root `[tool.uv].constraint-dependencies` | `transformers>=5.0.0` and `diffusers>=0.36.0`; members declare what they install and satisfy these ranges | The root coordinates workspace-wide `transformers` and `diffusers` versions without installing runtime packages itself. |
| `evaluation` extra       | Root verifier                            | `scripts/verify_evaluate_layout_metrics.py` and its `evaluate.load` metrics                               | The verifier is root-owned, and its metrics require the evaluation runtime including explicit `opencv-python`.          |
| `dev` group              | Root tooling                             | Root checks, tests, and pre-commit commands                                                               | Repository-wide development tooling is coordinated at the root.                                                         |
| `docs` group             | Root tooling                             | `make setup` (`uv sync --all-packages --group docs`)                                                      | Documentation generation is a repository-wide root workflow.                                                            |

A full-workspace environment installs all workspace members with `uv sync --all-packages` and serves as an explicit compatibility check for cross-member development and CI. Add `--group docs` when preparing the full checkout for documentation work, as in `make setup`.

## Training ownership

Training infrastructure follows the current package boundaries below. Model packages retain dataset, numerical, scheduling, configuration, and model-specific parity behavior.

| Concern                                                | Current owner            | Notes                                                                                                                             |
| ------------------------------------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| LightningCLI/bootstrap                                 | `traingen`               | Model-agnostic repository training entrypoint used by package-scoped training environments.                                       |
| Generic Lightning logging/reduction helpers            | `laygen.common.training` | Used by LayoutDM and LayoutDiffusion; candidate for migration into `traingen` once an import-migration plan exists.               |
| RNG capture/restore and deterministic runtime controls | `traingen-parity`        | Parity-only primitives are used by LayoutDM, LayoutFlow, and LayoutDiffusion adapters; model-specific seed wrappers remain local. |
| Trace construction/comparison/report helpers           | `traingen-parity`        | Generic parity primitives; model packages provide trace schemas and adapter functions.                                            |
| Model-specific trace point names                       | Model package            | Each package owns its trace tuple; trace names are not shared merely for reuse.                                                   |
| Dataset/data transforms                                | Model package            | Dataset formats, preprocessing, ordering, and loader behavior remain package-specific.                                            |
| Loss definitions                                       | Model package            | Loss formulas and reduction semantics remain package-specific.                                                                    |
| Scheduler/sampler policy                               | Model package            | Optimizer cadence, timestep sampling, EMA, and scheduler behavior remain package-specific.                                        |
| Training YAML/config values                            | Model package            | Dataset and model configuration values live with each package's training entrypoints.                                             |
| Model-specific callbacks                               | Model package            | Callbacks remain local unless at least two concrete consumers have identical behavior.                                            |
| Checkpoint conversion                                  | Model package            | Conversion code follows each model's checkpoint and topology contract.                                                            |

The generic helpers currently owned by `laygen.common.training` have these dispositions:

| Helper                 | Current consumers                                                                                     | Disposition                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `ScalarLogger`         | Structural protocol in helper signatures; satisfied by LayoutDM and LayoutDiffusion Lightning modules | Candidate for migration into `traingen` once an import-migration plan exists. |
| `sum_loss_values`      | `layout-dm`, `layoutdiffusion`, and `finish_training_step`                                            | Candidate for migration into `traingen` once an import-migration plan exists. |
| `log_training_losses`  | `finish_training_step` and shared helper tests                                                        | Candidate for migration into `traingen` once an import-migration plan exists. |
| `finish_training_step` | `layout-dm`, `layoutdiffusion`, and shared helper tests                                               | Candidate for migration into `traingen` once an import-migration plan exists. |
| `log_validation_loss`  | `layout-dm`, `layoutdiffusion`, and shared helper tests                                               | Candidate for migration into `traingen` once an import-migration plan exists. |

## Shared package names and imports

`laygen` is the shared layout-generation library; its installable package name and its import name are both `laygen`. `posgen` is the shared poster and content-aware library; its installable package name and its import name are both `posgen`.

Use `laygen.common` for helpers that are reusable across layout-generation packages, and use `posgen.common` for helpers that are reusable across poster or content-aware packages.

## Ownership boundaries

| Concern                                                                                                             | Shared owner            | Model-package responsibility                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Layout boxes, labels, discrete helpers, visualization, serialization, testing, and other proven layout-wide helpers | `laygen.common`         | Apply model-specific coordinate conventions, token order, checkpoint details, and parity behavior.                                     |
| Poster content, poster labels, poster testing, and poster visualization that are shared by concrete consumers       | `posgen.common`         | Keep model-specific image processing, saliency behavior, retrieval, prompt parsing, tokenization, scheduling, and configuration local. |
| Public output types and pipeline contracts                                                                          | `laygen` public modules | Return the repository's common schema while preserving model-specific optional data in the documented fields.                          |

Move a helper into a shared package when at least two packages need the same behavior or when a shared public contract must exist before a second consumer lands. Do not create a speculative abstraction without concrete shared behavior.

## Dependency direction

All model packages may import `laygen` directly. Poster or content-aware model packages may additionally import `posgen`; `posgen` may depend on `laygen` when shared layout primitives are needed, but the current `posgen` package does not. Shared libraries never import model packages.

`laygen` must not import `posgen` or model packages, and `posgen` must not import model packages.

The Transformers-side shared layout output in `laygen.modeling_outputs` is based on `transformers.utils.ModelOutput`, not `diffusers.utils.BaseOutput`. The Diffusers-side output in `laygen.pipelines.pipeline_output` may use the Diffusers base type behind the optional `diffusion` extra. The `laygen` core dependencies must not gain a hard `diffusers` dependency.

```mermaid
graph LR
    laygen["laygen"] --> model_packages["model packages"]
    laygen -. "optional; currently unused" .-> posgen["posgen"]
    posgen --> poster_models["poster or content-aware model packages"]
```

The arrows point from a library to the packages that import it.

`posgen` is a workspace member, and current poster/content-aware consumers use it alongside `laygen`.

## Working with the architecture

Import shared helpers using their public module paths, such as `from laygen.common.bbox import normalize_boxes` or `from posgen.common import PositionContent`.

Keep model-specific code in the model package even when it resembles a shared helper, and add a shared helper only when its behavior and ownership are clear to its consumers.

This page defines where packages live, which package owns each import name, dependency direction, and output dependency boundaries; the current public output fields and pipeline contracts remain documented in [Conventions](conventions/), and API details remain in the generated reference.
