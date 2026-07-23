"""Visualization helpers for House-GAN outputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from PIL import Image, ImageDraw


def render_layout(
    bbox: torch.Tensor,
    labels: torch.Tensor,
    *,
    id2label: Mapping[int, str],
    canvas_size: tuple[int, int] = (256, 256),
) -> Image.Image:
    """Render normalized center ``xywh`` boxes for debugging."""
    width, height = canvas_size
    image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(image)
    for box, label in zip(bbox.detach().cpu(), labels.detach().cpu(), strict=True):
        cx, cy, bw, bh = box.tolist()
        left = (cx - bw / 2.0) * width
        top = (cy - bh / 2.0) * height
        right = (cx + bw / 2.0) * width
        bottom = (cy + bh / 2.0) * height
        _ = id2label.get(int(label.item()), str(int(label.item())))
        draw.rectangle((left, top, right, bottom), outline="black", width=2)
    return image
