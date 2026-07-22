"""Candidate generation helpers for SmartText text placement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .configuration_smarttext import SmartTextConfig

_INF: Final[float] = 1_000_000_007.0


class VendorBoxRow(TypedDict, total=False):
    """JSON-compatible vendor candidate row."""

    idx: int
    xl: int
    yl: int
    xr: int
    yr: int
    tl_cnt: int
    fsz: int
    fontstr: str


@dataclass(frozen=True)
class SmartTextLine:
    """One rendered prompt line within a candidate region."""

    text: str
    font_size: int
    bbox_ltrb_px: tuple[int, int, int, int]


@dataclass(frozen=True)
class SmartTextCandidate:
    """A candidate text block and its line-level boxes."""

    index: int
    bbox_ltrb_px: tuple[int, int, int, int]
    lines: tuple[SmartTextLine, ...]


def split_prompt_lines(prompt: str, ratio_list: Sequence[float]) -> tuple[str, ...]:
    r"""Split prompt text into non-empty lines.

    Args:
        prompt: User text payload.
        ratio_list: Font-size ratio list. The argument is accepted here so tests
            can verify prompt/ratio length behavior at one public boundary.

    Returns:
        Non-empty prompt lines, preserving internal spaces.

    Raises:
        ValueError: If no non-empty line remains.

    Examples:
        >>> split_prompt_lines("A\\nB", (1.0, 0.8))
        ('A', 'B')
    """
    del ratio_list
    lines = tuple(line for line in prompt.splitlines() if line.strip())
    if not lines:
        raise ValueError("SmartText requires at least one prompt line")
    return lines


def generate_candidates(
    image: Image.Image,
    saliency: np.ndarray | torch.Tensor,
    *,
    prompt: str,
    font: str | Path | ImageFont.FreeTypeFont | ImageFont.ImageFont,
    config: SmartTextConfig,
    ratio_list: Sequence[float] | None = None,
) -> list[SmartTextCandidate]:
    """Generate deterministic candidate text boxes from image, saliency, and text.

    Args:
        image: Source RGB image.
        saliency: Saliency map in image space.
        prompt: Text payload split on newlines.
        font: TrueType font path or loaded PIL font object.
        config: SmartText configuration.
        ratio_list: Optional per-line font-size ratios.

    Returns:
        Candidate text regions sorted in deterministic search order.

    Examples:
        >>> img = Image.new("RGB", (64, 64), "white")
        >>> candidates = generate_candidates(
        ...     img,
        ...     np.zeros((64, 64), dtype=np.float32),
        ...     prompt="Hi",
        ...     font=ImageFont.load_default(),
        ...     config=SmartTextConfig(grid_num=16, max_font_size=20),
        ... )
        >>> bool(candidates)
        True
    """
    saliency_np = _saliency_to_numpy(saliency, image.size)
    lines = split_prompt_lines(prompt, ratio_list or config.ratio_list)
    ratios = _expand_ratios(ratio_list or config.ratio_list, len(lines))
    search_map, grid_rsz, grid_csz = _build_search_map(saliency_np, config)
    width, height = image.size
    min_text_area = height * width / config.max_text_area_coef
    max_text_area = height * width / config.min_text_area_coef
    draw = ImageDraw.Draw(image.copy())

    candidates: list[SmartTextCandidate] = []
    index = 0
    for font_size in range(
        config.min_font_size,
        config.max_font_size + 1,
        config.font_inc_unit,
    ):
        line_sizes = [
            _text_size(
                draw,
                text,
                _resolve_font(font, max(1, int(font_size * ratio))),
                config.text_spacing,
            )
            for text, ratio in zip(lines, ratios, strict=True)
        ]
        block_width = max((line_width for line_width, _ in line_sizes), default=0)
        block_height = sum(line_height for _, line_height in line_sizes)
        block_height += config.text_spacing * max(0, len(line_sizes) - 1)
        area = block_width * block_height
        if (
            area > max_text_area
            or area < min_text_area
            or block_width >= width
            or block_height >= height
        ):
            continue
        kernel = (
            max(1, int(block_height / grid_rsz)),
            max(1, int(block_width / grid_csz)),
        )
        for row, col in _top_non_overlapping(search_map, kernel, k=1):
            top = row * grid_rsz
            left = col * grid_csz
            right = left + block_width
            bottom = top + block_height
            if right >= width or bottom >= height:
                continue
            line_rows: list[SmartTextLine] = []
            cursor_top = top
            for text, ratio, (line_width, line_height) in zip(
                lines,
                ratios,
                line_sizes,
                strict=True,
            ):
                line_font_size = max(1, int(font_size * ratio))
                line_rows.append(
                    SmartTextLine(
                        text=text,
                        font_size=line_font_size,
                        bbox_ltrb_px=(
                            left,
                            cursor_top,
                            left + line_width,
                            cursor_top + line_height,
                        ),
                    )
                )
                cursor_top += line_height + config.text_spacing
            candidates.append(
                SmartTextCandidate(
                    index=index,
                    bbox_ltrb_px=(left, top, right, bottom),
                    lines=tuple(line_rows),
                )
            )
            index += 1
    return candidates


def prepare_scorer_batch(
    image: Image.Image,
    candidates: Sequence[SmartTextCandidate],
    *,
    config: SmartTextConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[SmartTextCandidate]]:
    """Prepare scorer image tensor and RoI/RoD boxes.

    Args:
        image: Source image.
        candidates: Candidate boxes.
        config: SmartText configuration.

    Returns:
        ``pixel_values`` shaped ``(1, 3, H, W)``, boxes shaped ``(N, 5)`` with
        batch index in column zero, and the candidate list.
    """
    if config.uses_expanded_region:
        return _prepare_expanded_scorer_batch(image, candidates, config=config)
    pixel_values = _preprocess_scorer_image(image, config)
    width, height = image.size
    scale_x = pixel_values.shape[-1] / width
    scale_y = pixel_values.shape[-2] / height
    box_rows = []
    for candidate in candidates:
        left, top, right, bottom = candidate.bbox_ltrb_px
        box_rows.append(
            [
                0.0,
                float(left) * scale_x,
                float(top) * scale_y,
                float(right) * scale_x,
                float(bottom) * scale_y,
            ]
        )
    boxes = torch.tensor(box_rows, dtype=torch.float32)
    return pixel_values, boxes, list(candidates)


def candidate_to_vendor_json(candidate: SmartTextCandidate) -> list[VendorBoxRow]:
    """Convert a candidate to the vendor JSON row format."""
    left, top, right, bottom = candidate.bbox_ltrb_px
    rows: list[VendorBoxRow] = [
        {
            "idx": candidate.index,
            "xl": top,
            "yl": left,
            "xr": bottom,
            "yr": right,
            "tl_cnt": len(candidate.lines),
        }
    ]
    for line in candidate.lines:
        l_left, l_top, l_right, l_bottom = line.bbox_ltrb_px
        rows.append(
            {
                "xl": l_top,
                "yl": l_left,
                "xr": l_bottom,
                "yr": l_right,
                "fsz": line.font_size,
                "fontstr": line.text,
            }
        )
    return rows


def candidate_from_vendor_json(
    row: Sequence[Mapping[str, object]],
) -> SmartTextCandidate:
    """Convert one vendor JSON candidate row to typed metadata."""
    head = row[0]
    lines = []
    for line in row[1:]:
        lines.append(
            SmartTextLine(
                text=str(line["fontstr"]),
                font_size=int(cast(int | float | str, line["fsz"])),
                bbox_ltrb_px=(
                    int(cast(int | float | str, line["yl"])),
                    int(cast(int | float | str, line["xl"])),
                    int(cast(int | float | str, line["yr"])),
                    int(cast(int | float | str, line["xr"])),
                ),
            )
        )
    return SmartTextCandidate(
        index=int(cast(int | float | str, head["idx"])),
        bbox_ltrb_px=(
            int(cast(int | float | str, head["yl"])),
            int(cast(int | float | str, head["xl"])),
            int(cast(int | float | str, head["yr"])),
            int(cast(int | float | str, head["xr"])),
        ),
        lines=tuple(lines),
    )


def _saliency_to_numpy(
    saliency: np.ndarray | torch.Tensor,
    size: tuple[int, int],
) -> np.ndarray:
    if isinstance(saliency, torch.Tensor):
        array = saliency.detach().cpu().float().numpy()
    else:
        array = np.asarray(saliency, dtype=np.float32)
    array = np.squeeze(array)
    if array.shape != (size[1], size[0]):
        image = Image.fromarray((array * 255).astype(np.uint8)).resize(
            size, Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.max() <= 1.0:
        array = array * 255.0
    return array.astype(np.float32)


def _build_search_map(
    saliency: np.ndarray,
    config: SmartTextConfig,
) -> tuple[np.ndarray, int, int]:
    height, width = saliency.shape
    grid_rsz = max(1, int(height / config.grid_num))
    if grid_rsz > 1 and grid_rsz % 2 == 1:
        grid_rsz -= 1
    grid_csz = grid_rsz
    pooled = F.avg_pool2d(
        torch.as_tensor(saliency).view(1, 1, height, width),
        kernel_size=grid_rsz,
    ).squeeze()
    crop_mat = pooled.numpy() * grid_rsz * grid_csz
    crop_rows, crop_cols = crop_mat.shape
    flat_desc = np.sort(crop_mat.flatten())[::-1]
    kth = min(len(flat_desc) - 1, int(crop_rows * crop_cols / config.saliency_coef))
    threshold = flat_desc[kth]
    matrix_cal = np.empty_like(crop_mat)
    for row in range(crop_rows):
        for col in range(crop_cols):
            if (
                crop_mat[row, col] > threshold
                or row <= 3
                or col <= 3
                or row >= crop_rows - 4
                or col >= crop_cols - 4
            ):
                matrix_cal[row, col] = _INF
            else:
                matrix_cal[row, col] = crop_mat[row, col]
    return _smooth_importance(crop_mat, matrix_cal, flat_desc), grid_rsz, grid_csz


def _smooth_importance(
    matrix: np.ndarray,
    matrix_cal: np.ndarray,
    matrix1d: np.ndarray,
) -> np.ndarray:
    rows, cols = matrix.shape
    remaining = 0
    while remaining < rows * cols:
        for row in range(rows):
            for col in range(cols):
                total = 0.0
                for rr in (row - 1, row, row + 1):
                    for cc in (col - 1, col, col + 1):
                        total += (
                            matrix_cal[rr, cc]
                            if 0 <= rr < rows and 0 <= cc < cols
                            else _INF
                        )
                matrix_cal[row, col] = total / 9.0
                index = row * cols + col
                if (
                    matrix1d[index] != -1
                    and matrix_cal[row, col] - matrix[row, col] > 0.5
                ):
                    matrix1d[index] = -1
                    remaining += 1
        if rows * cols == 0:
            break
    return matrix_cal


def _top_non_overlapping(
    matrix: np.ndarray,
    kernel_size: tuple[int, int],
    *,
    k: int,
) -> list[tuple[int, int]]:
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    height, width = tensor.shape
    kh, kw = kernel_size
    kh = min(kh, height)
    kw = min(kw, width)
    pooled = F.avg_pool2d(
        tensor.view(1, 1, height, width), kernel_size=(kh, kw), stride=(1, 1)
    )
    scores = pooled.view(height - kh + 1, width - kw + 1) * kh * kw
    indexes = np.dstack(
        np.unravel_index(np.argsort(scores.numpy().ravel()), scores.shape)
    )[0]
    selected: list[tuple[int, int]] = []
    for row, col in indexes:
        if len(selected) >= k:
            break
        if all(
            not _intersects((row, col, row + kh, col + kw), (r, c, r + kh, c + kw))
            for r, c in selected
        ):
            selected.append((int(row), int(col)))
    return selected


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0, right - left) * max(0, bottom - top) > 0


def _expand_ratios(ratio_list: Sequence[float], line_count: int) -> tuple[float, ...]:
    ratios = list(ratio_list)
    ratios.extend([1.0] * max(0, line_count - len(ratios)))
    return tuple(ratios[:line_count])


def _resolve_font(
    font: str | Path | ImageFont.FreeTypeFont | ImageFont.ImageFont,
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if isinstance(font, ImageFont.FreeTypeFont):
        return font.font_variant(size=size)
    if isinstance(font, str | Path):
        return ImageFont.truetype(str(font), size, encoding="utf-8")
    return cast(ImageFont.ImageFont, font)


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    spacing: int,
) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font=font, spacing=spacing
        )
        return int(right - left), int(bottom - top)
    return cast(tuple[int, int], draw.textsize(text, font=font, spacing=spacing))


def _preprocess_scorer_image(
    image: Image.Image, config: SmartTextConfig
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = _resize_for_scorer(array, config)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()


def _prepare_expanded_scorer_batch(
    image: Image.Image,
    candidates: Sequence[SmartTextCandidate],
    *,
    config: SmartTextConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[SmartTextCandidate]]:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized_images: list[np.ndarray] = []
    box_rows: list[list[float]] = []
    max_height = 0
    max_width = 0
    for index, candidate in enumerate(candidates):
        crop, relative_box = _find_expanded_region(
            source, candidate, exp_prop=config.exp_prop
        )
        resized = _resize_for_scorer(crop, config)
        max_height = max(max_height, resized.shape[0])
        max_width = max(max_width, resized.shape[1])
        scale_height = crop.shape[0] / float(resized.shape[0])
        scale_width = crop.shape[1] / float(resized.shape[1])
        rel_top, rel_left, rel_bottom, rel_right = relative_box
        box_rows.append(
            [
                float(index),
                float(rel_left) / scale_width,
                float(rel_top) / scale_height,
                float(rel_right) / scale_width,
                float(rel_bottom) / scale_height,
            ]
        )
        resized_images.append(resized)
    padded = []
    for resized in resized_images:
        pad_h = max_height - resized.shape[0]
        pad_w = max_width - resized.shape[1]
        padded.append(
            np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), "constant").transpose(
                2, 0, 1
            )
        )
    if not padded:
        return (
            torch.empty((0, 3, 0, 0), dtype=torch.float32),
            torch.empty((0, 5), dtype=torch.float32),
            [],
        )
    return (
        torch.from_numpy(np.stack(padded)).float(),
        torch.tensor(box_rows, dtype=torch.float32),
        list(candidates),
    )


def _find_expanded_region(
    image: np.ndarray,
    candidate: SmartTextCandidate,
    *,
    exp_prop: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    left, top, right, bottom = candidate.bbox_ltrb_px
    anno_width = int(np.floor(abs(float(right - left))))
    anno_height = int(np.floor(abs(float(bottom - top))))
    src_top = min(top, bottom)
    src_bottom = max(top, bottom)
    src_left = min(left, right)
    src_right = max(left, right)
    new_top = max(0, int(src_top - exp_prop * anno_height))
    new_left = max(0, int(src_left - exp_prop * anno_width))
    new_bottom = min(image.shape[0], int(src_bottom + exp_prop * anno_height))
    new_right = min(image.shape[1], int(src_right + exp_prop * anno_width))
    crop = image[new_top:new_bottom, new_left:new_right].copy()
    return crop, (
        src_top - new_top,
        src_left - new_left,
        src_bottom - new_top,
        src_right - new_left,
    )


def _resize_for_scorer(array: np.ndarray, config: SmartTextConfig) -> np.ndarray:
    height, width = array.shape[:2]
    scale = float(config.image_size) / float(min(height, width))
    resized_h = max(32, int(round(height * scale / 32.0) * 32))
    resized_w = max(32, int(round(width * scale / 32.0) * 32))
    try:
        import cv2  # type: ignore[import-not-found]

        resized = cv2.resize(array, (resized_w, resized_h))
    except ModuleNotFoundError:
        resized = np.asarray(
            Image.fromarray(array).resize(
                (resized_w, resized_h), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    array = resized.astype(np.float32) / 256.0
    mean = np.asarray(config.rgb_mean, dtype=np.float32)
    std = np.asarray(config.rgb_std, dtype=np.float32)
    return (array - mean) / std
