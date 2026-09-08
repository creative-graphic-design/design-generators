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

## Shared package namespaces

`laygen` is the shared layout-generation distribution and Python namespace, and `posgen` is the shared poster and content-aware distribution and Python namespace.

Use `laygen.common` for helpers that are reusable across layout-generation packages, and use `posgen.common` for helpers that are reusable across poster or content-aware packages.

## Ownership boundaries

| Concern | Shared owner | Model-package responsibility |
| --- | --- | --- |
| Layout boxes, labels, discrete helpers, visualization, serialization, testing, and other proven layout-wide helpers | `laygen.common` | Apply model-specific coordinate conventions, token order, checkpoint details, and parity behavior. |
| Poster content, poster labels, poster testing, and poster visualization that are shared by concrete consumers | `posgen.common` | Keep model-specific image processing, saliency behavior, retrieval, prompt parsing, tokenization, scheduling, and configuration local. |
| Public output types and pipeline contracts | `laygen` public modules | Return the repository's common schema while preserving model-specific optional data in the documented fields. |

Move a helper into a shared package when at least two packages need the same behavior or when a shared public contract must exist before a second consumer lands. Do not create a speculative abstraction without concrete shared behavior.

## Dependency direction

All model packages may import `laygen` directly. Poster or content-aware model packages may additionally import `posgen`; `posgen` may depend on `laygen` when shared layout primitives are needed, but the current `posgen` package does not. Shared libraries never import model packages.

`laygen` must not import `posgen` or model packages, and `posgen` must not import model packages.

The Transformers-side shared layout output in `laygen.modeling_outputs` is based on `transformers.utils.ModelOutput`, not `diffusers.utils.BaseOutput`. The Diffusers-side output in `laygen.pipelines.pipeline_output` may use the Diffusers base type behind the optional `diffusion` extra. The `laygen` core dependencies must not gain a hard `diffusers` dependency.

```text
model packages -> laygen
poster or content-aware model packages -> posgen
posgen -> laygen (optional; currently unused)
laygen -> no posgen or model imports
posgen -> no model imports
```

`posgen` is a root workspace dependency because current poster/content-aware consumers use it alongside `laygen`.

## Working with the architecture

Import shared helpers from their public namespace, such as `from laygen.common.bbox import normalize_boxes` or `from posgen.common import PositionContent`.

Keep model-specific code in the model package even when it resembles a shared helper, and add a shared helper only when its behavior and ownership are clear to its consumers.

This page defines package placement, namespace ownership, dependency direction, and output dependency boundaries; the current public output fields and pipeline contracts remain documented in [Conventions](conventions/), and API details remain in the generated reference.
