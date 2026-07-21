"""Text color helpers ported from the SmartText demo path."""

from __future__ import annotations

import numpy as np
from PIL import Image


def choose_text_color(
    image: Image.Image | np.ndarray,
    crop_bbox_ltrb_px: tuple[int, int, int, int],
    *,
    contrast_threshold: float,
) -> str:
    """Choose black or white foreground for a candidate region.

    Args:
        image: Source RGB image.
        crop_bbox_ltrb_px: Candidate box as ``(left, top, right, bottom)``.
        contrast_threshold: Threshold above which white text is preferred.

    Returns:
        Hex foreground color.

    Examples:
        >>> choose_text_color(Image.new("RGB", (8, 8), "black"), (0, 0, 8, 8), contrast_threshold=5)
        '#ffffff'
    """
    array = np.asarray(
        image.convert("RGB") if isinstance(image, Image.Image) else image
    )
    left, top, right, bottom = crop_bbox_ltrb_px
    crop = array[top:bottom, left:right]
    if crop.size == 0:
        return "#000000"
    luminance = np.dot(crop[..., :3], np.array([0.299, 0.587, 0.114])).mean()
    return "#ffffff" if luminance < contrast_threshold * 25.5 else "#000000"
