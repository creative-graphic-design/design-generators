---
icon: lucide/layout-template
tags:
  - Overview
  - Documentation
---

# design-generators

`design-generators` ports layout, poster, and graphic-design generation research repositories into Transformers- and Diffusers-style packages. The target interface is `from_pretrained`-ready inference with shared output schemas, consistent conditioning names, and reproducible conversion paths.

The repository plan, target model table, execution order, interface decisions, and data policy are tracked in [issue #2](https://github.com/creative-graphic-design/design-generators/issues/2). The living implementation checklist is tracked in [issue #60](https://github.com/creative-graphic-design/design-generators/issues/60).

## Documentation Layout

- [Conventions](conventions.md) describes package, interface, parity, data, and documentation conventions for users and contributors.
- [API Reference](api/index.md) is generated from workspace members under `lib/*` and `models/*`. It is empty on repository baselines that do not yet contain workspace packages, and updates automatically as packages are added.
