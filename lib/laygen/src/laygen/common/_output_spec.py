from __future__ import annotations

from dataclasses import MISSING

import torch

LAYOUT_GENERATION_OUTPUT_FIELDS = (
    ("bbox", torch.Tensor, MISSING),
    ("labels", torch.Tensor, None),
    ("mask", torch.Tensor, None),
    ("id2label", dict[int, str], None),
    ("sequences", torch.Tensor | None, None),
    ("scores", torch.Tensor | None, None),
    ("trajectory", object | None, None),
    ("intermediates", object | None, None),
)


def dataclass_fields() -> list[tuple[str, object] | tuple[str, object, object]]:
    fields: list[tuple[str, object] | tuple[str, object, object]] = []
    for name, annotation, default in LAYOUT_GENERATION_OUTPUT_FIELDS:
        if default is MISSING:
            fields.append((name, annotation))
        else:
            fields.append((name, annotation, default))
    return fields
