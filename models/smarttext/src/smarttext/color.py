"""Text color helpers ported from the SmartText demo path."""

# ruff: noqa: E402
from __future__ import annotations

import os

# Set BLAS/OpenMP defaults before importing scikit-learn.
for _thread_env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

import numpy as np
from jaxtyping import Shaped
from PIL import Image
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from typing import TypedDict


class SmartTextColorCandidate(TypedDict):
    """Foreground color candidate and contrast score."""

    color: Shaped[np.ndarray, "channels"] | list[float] | list[int]
    contrast_rate: float


def dominant_colors(
    image: Shaped[np.ndarray, "height width channels"], clusters: int
) -> list[Shaped[np.ndarray, "channels"]]:
    """Return reference-sorted KMeans dominant colors.

    Args:
        image: RGB image array.
        clusters: Number of KMeans clusters.

    Returns:
        Cluster centers sorted by RGB tuple, matching
        the original ``cal_color.py::cal_domcolor``.
    """
    pixels = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    estimator = KMeans(n_clusters=clusters, max_iter=300, n_init=2)
    with threadpool_limits(limits=1):
        estimator.fit(pixels)
    return sorted(estimator.cluster_centers_, key=lambda row: (row[0], row[1], row[2]))


def rgb_distance(
    rgb: Shaped[np.ndarray, "channels"] | list[float] | list[int],
) -> float:
    """Return the channel-spread distance."""
    return abs(rgb[0] - rgb[1]) + abs(rgb[0] - rgb[2]) + abs(rgb[2] - rgb[1])


def rgb_to_hex(rgb: Shaped[np.ndarray, "channels"] | list[float] | list[int]) -> str:
    """Convert an RGB row to the uppercase hex form."""
    color = "#"
    for channel in rgb:
        color += str(hex(int(channel)))[-2:].replace("x", "0").upper()
    return color


def luminance(rgb: list[float]) -> float:
    """Return WCAG relative luminance using the reference formula."""
    values = list(rgb)
    for index in range(len(values)):
        if values[index] <= 0.03928:
            values[index] = values[index] / 12.92
        else:
            values[index] = pow(((values[index] + 0.055) / 1.055), 2.4)
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def contrast_rate(
    rgb_a: Shaped[np.ndarray, "channels"] | list[float] | list[int],
    rgb_b: Shaped[np.ndarray, "channels"] | list[float] | list[int],
) -> float:
    """Return reference-rounded contrast ratio between two RGB colors."""
    l1 = luminance([rgb_a[0] / 255, rgb_a[1] / 255, rgb_a[2] / 255])
    l2 = luminance([rgb_b[0] / 255, rgb_b[1] / 255, rgb_b[2] / 255])
    if l1 >= l2:
        ratio = (l1 + 0.05) / (l2 + 0.05)
    else:
        ratio = (l2 + 0.05) / (l1 + 0.05)
    return round(ratio * 100) / 100


def best_color_candidates(
    image: Image.Image | Shaped[np.ndarray, "height width channels"],
    crop: Shaped[np.ndarray, "crop_height crop_width channels"],
    *,
    contrast_threshold: float,
    random_seed: int | None = 0,
) -> list[SmartTextColorCandidate]:
    """Return reference-ordered color candidates for a text region.

    Args:
        image: Source RGB image.
        crop: Candidate crop from ``image``.
        contrast_threshold: Minimum contrast ratio accepted before fallback.
        random_seed: Seed used to make the reference KMeans path deterministic.

    Returns:
        Candidate dictionaries with ``color`` and ``contrast_rate`` keys.
    """
    state = np.random.get_state() if random_seed is not None else None
    if random_seed is not None:
        np.random.seed(random_seed)
    try:
        return _best_color_candidates_unseeded(
            image, crop, contrast_threshold=contrast_threshold
        )
    finally:
        if state is not None:
            np.random.set_state(state)


def _best_color_candidates_unseeded(
    image: Image.Image | Shaped[np.ndarray, "height width channels"],
    crop: Shaped[np.ndarray, "crop_height crop_width channels"],
    *,
    contrast_threshold: float,
) -> list[SmartTextColorCandidate]:
    image_array = np.asarray(
        image.convert("RGB") if isinstance(image, Image.Image) else image
    )
    color_candidates = dominant_colors(image_array, 6)
    crop_color = dominant_colors(crop, 1)[0]
    chosen: list[SmartTextColorCandidate] = []
    grey_flag = False
    for color in color_candidates:
        rate = contrast_rate(color, crop_color)
        if rate > contrast_threshold:
            chosen.append({"color": color, "contrast_rate": rate})

    if not chosen:
        grey_flag = True
        for grey_color in ([value, value, value] for value in range(0, 256, 50)):
            rate = contrast_rate(grey_color, crop_color)
            if rate > contrast_threshold:
                chosen.append({"color": grey_color, "contrast_rate": rate})

    if not chosen:
        black_rate = contrast_rate([0, 0, 0], crop_color)
        white_rate = contrast_rate([255, 255, 255], crop_color)
        if black_rate > white_rate:
            chosen.append({"color": [0, 0, 0], "contrast_rate": black_rate})
        else:
            chosen.append({"color": [255, 255, 255], "contrast_rate": white_rate})

    if grey_flag:
        return sorted(chosen, key=lambda row: row["contrast_rate"], reverse=True)
    return sorted(chosen, key=lambda row: rgb_distance(row["color"]), reverse=True)


def choose_text_color(
    image: Image.Image | Shaped[np.ndarray, "height width channels"],
    crop_bbox_ltrb_px: tuple[int, int, int, int],
    *,
    contrast_threshold: float,
) -> str:
    """Choose the SmartText foreground for a candidate region.

    Args:
        image: Source RGB image.
        crop_bbox_ltrb_px: Candidate box as ``(left, top, right, bottom)``.
        contrast_threshold: Threshold above which white text is preferred.

    Returns:
        Hex foreground color in the uppercase form.

    Examples:
        >>> choose_text_color(Image.new("RGB", (8, 8), "black"), (0, 0, 8, 8), contrast_threshold=5)
        '#FFFFFF'
    """
    array = np.asarray(
        image.convert("RGB") if isinstance(image, Image.Image) else image
    )
    left, top, right, bottom = crop_bbox_ltrb_px
    crop = array[top:bottom, left:right]
    if crop.size == 0:
        return "#000000"
    color = best_color_candidates(array, crop, contrast_threshold=contrast_threshold)[
        0
    ]["color"]
    return rgb_to_hex(color)
