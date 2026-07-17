# posgen

Shared schemas and utilities reserved for future position-generation packages.

`posgen` is intentionally a thin skeleton today. It exists so poster-generation and content-aware placement models can share small, stable contracts once real consumers appear.

## Scope

- Keep generic position content containers, label normalization, schema tests, and lightweight summaries here.
- Use `posgen` for poster/content-aware placement concerns.
- Keep saliency, retrieval, ranking, and dataset-specific placement logic in model packages until there is proven reuse.
- Do not add `posgen` to the root project dependencies until the first real consumer needs it.

## Dependency Direction

`posgen` can depend on `laygen` later if position-generation packages need shared layout primitives. `laygen` should not depend on `posgen`.

```text
model package -> posgen -> laygen
model package -> laygen
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

## Growth Rule

Add to `posgen` only when a second package needs the same behavior, or when a first consumer needs a stable public contract before other position-generation packages arrive.
