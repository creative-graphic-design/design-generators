# posgen

![package](https://img.shields.io/static/v1?label=package&message=posgen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![status](https://img.shields.io/static/v1?label=status&message=skeleton&color=lightgrey&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square)](https://creative-graphic-design.github.io/design-generators/)

`posgen` is the shared package reserved for poster-generation and content-aware placement helpers. It is intentionally small until real poster packages create repeated contracts.

## Install

```bash
uv sync --package posgen
```

## Current API

```bash
uv run --package posgen python - <<'PY'
import torch
from posgen.common import PositionContent, render_position_summary

content = PositionContent(
    positions=torch.zeros(1, 2, 2),
    mask=torch.tensor([[True, False]]),
)
print(render_position_summary(content))
PY
```

## Dependency Direction

`posgen` may depend on `laygen` when poster/content-aware placement packages need shared layout primitives. `laygen` must not depend on `posgen`.

```text
model package -> posgen -> laygen
model package -> laygen
```

## Growth Rule

Add to `posgen` only when a second package needs the same behavior, or when a first consumer needs a stable public contract before other position-generation packages arrive.
