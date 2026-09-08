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

The accepted package names are plain `laygen` and `posgen`; do not introduce the old `layout_generation_common` or `layout-generation-common` names.

## Ownership boundaries

| Concern | Shared owner | Model-package responsibility |
| --- | --- | --- |
| Layout boxes, labels, discrete helpers, visualization, serialization, testing, and other proven layout-wide helpers | `laygen.common` | Apply model-specific coordinate conventions, token order, checkpoint details, and parity behavior. |
| Poster content, poster labels, poster testing, and poster visualization that are shared by concrete consumers | `posgen.common` | Keep model-specific image processing, saliency behavior, retrieval, prompt parsing, tokenization, scheduling, and configuration local. |
| Public output types and pipeline contracts | `laygen` public modules | Return the repository's common schema while preserving model-specific optional data in the documented fields. |

Move a helper into a shared package when at least two packages need the same behavior or when a shared public contract must exist before a second consumer lands. Do not create a speculative abstraction without concrete shared behavior.

## Dependency direction

Layout model packages may import `laygen`, and poster or content-aware model packages may import `posgen` and, through `posgen`, `laygen`.

`laygen` must not import `posgen` or model packages, and `posgen` must not import model packages.

```text
layout model packages -> laygen
poster or content-aware model packages -> posgen -> laygen
laygen -> no posgen or model imports
posgen -> no model imports
```

`posgen` may exist as a workspace member before it becomes a root project dependency; add it to root dependencies when its first consumer requires it.

## Working with the architecture

Import shared helpers from their public namespace, such as `from laygen.common.bbox import normalize_boxes` or `from posgen.common import PositionContent`.

Keep model-specific code in the model package even when it resembles a shared helper, and add a shared helper only when its behavior and ownership are clear to its consumers.

This page defines package placement, namespace ownership, and dependency direction; the current public output fields and pipeline contracts remain documented in [Conventions](conventions.md), and API details remain in the generated reference.
