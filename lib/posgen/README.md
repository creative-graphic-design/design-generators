# posgen

![package](https://img.shields.io/static/v1?label=package&message=posgen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![status](https://img.shields.io/static/v1?label=status&message=skeleton&color=lightgrey&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square)](https://creative-graphic-design.github.io/design-generators/)

[`posgen`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/posgen) is the shared package reserved for poster-generation and content-aware placement helpers. It is intentionally small until real poster packages create repeated contracts.

`posgen` exists so poster-generation and content-aware placement models can share small, stable contracts once real consumers appear.

## Install

```bash
uv sync --package posgen
```

## Current API

```bash
uv run --package posgen python - <<'PY'
import torch
from posgen.common import (
    PositionContent,
    assert_position_content_schema,
    labels_for_dataset,
    normalize_label,
    render_position_summary,
)

content = PositionContent(
    positions=torch.zeros(1, 2, 2),
    mask=torch.tensor([[True, False]]),
)
assert_position_content_schema(content)
print(normalize_label("Anchor-Point"))
print(render_position_summary(content))
print(labels_for_dataset("crello"))
PY
```

Example output:

```text
anchor_point
1 active positions
('coloredBackground', 'imageElement', 'maskElement', 'svgElement', 'textElement')
```

## Scope

- Keep generic position content containers, label normalization, schema tests, and lightweight summaries here.
- Use `posgen` for poster/content-aware placement concerns.
- Keep saliency, retrieval, ranking, and dataset-specific placement logic in model packages until there is proven reuse.
- Do not add `posgen` to the root project dependencies until the first real consumer needs it.

## Dependency Direction

`posgen` may depend on [`laygen`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/laygen) when poster/content-aware placement packages need shared layout primitives. `laygen` must not depend on `posgen`.

```text
model package -> posgen -> laygen
model package -> laygen
```

## Growth Rule

Add to `posgen` only when a second package needs the same behavior, or when a first consumer needs a stable public contract before other position-generation packages arrive.
