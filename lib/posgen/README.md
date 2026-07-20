# posgen

![package](https://img.shields.io/static/v1?label=package&message=posgen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![status](https://img.shields.io/static/v1?label=status&message=skeleton&color=lightgrey&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)

[`posgen`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/posgen) provides shared utilities for poster-generation and content-aware placement packages. Its current public surface is intentionally small: `DatasetName` identifies supported poster/content datasets, `PositionContent` carries position tensors plus masks, and label helpers normalize dataset-specific class names.

Use `posgen` for poster or content-aware placement concerns such as Crello-style element categories and position-content summaries. Use [`laygen`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/laygen) for general layout-generation schemas, bbox conversion, scheduler adapters, and model-card helpers. The package is still a skeleton until more poster models need shared behavior, so model-specific saliency, retrieval, ranking, and rendering logic should stay in model packages.

## Install

```bash
uv sync --package posgen
```

## Current API

```bash
uv run --package posgen python
```

```python
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

Add to `posgen` only when a second package needs the same behavior, or when a first consumer needs a stable public contract before other poster/content-aware placement packages arrive.

## Pointers

- [Documentation site](https://creative-graphic-design.github.io/design-generators/)
- [API reference](https://creative-graphic-design.github.io/design-generators/api/libraries/posgen/)
