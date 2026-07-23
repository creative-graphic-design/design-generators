"""Dataset helpers for the LayoutDETR Ad Banner checkpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Literal, cast

import json
import torch

from laygen.common.bbox import normalize_boxes

AD_BANNER_LABELS: Final[tuple[str, ...]] = (
    "header",
    "pre-header",
    "post-header",
    "body text",
    "disclaimer / footnote",
    "button",
    "callout",
    "logo",
)


def id2label_for_ad_banner() -> dict[int, str]:
    """Return the LayoutDETR Ad Banner label vocabulary."""
    return dict(enumerate(AD_BANNER_LABELS))


def label2id_for_ad_banner() -> dict[str, int]:
    """Return the inverse Ad Banner label mapping."""
    return {label: index for index, label in id2label_for_ad_banner().items()}


def normalize_vendor_annotation(sample: Mapping[str, object]) -> dict[str, object]:
    """Normalize one vendor annotation row to public normalized center ``xywh``.

    Args:
        sample: Vendor element with ``xyxy_word_fit``, ``label``, ``str``,
            ``width``, and ``height`` values.

    Returns:
        A normalized row with ``bbox``, integer ``label``, and ``text``.

    Raises:
        ValueError: If the label or canvas metadata is invalid.
    """
    label = str(sample["label"])
    label2id = label2id_for_ad_banner()
    if label not in label2id:
        raise ValueError(f"Unknown Ad Banner label: {label}")
    width = int(cast(int | str, sample["width"]))
    height = int(cast(int | str, sample["height"]))
    xyxy = torch.tensor(sample["xyxy_word_fit"], dtype=torch.float32).view(1, 1, 4)
    bbox = normalize_boxes(xyxy, canvas_size=(width, height), box_format="ltrb")
    return {
        "bbox": bbox[0, 0].tolist(),
        "label": label2id[label],
        "text": str(sample.get("str", "")),
    }


def load_ad_banner_dataset(
    root: str | Path,
    *,
    split: Literal["train", "validation"],
    source: Literal["vendor"] = "vendor",
) -> Iterable[dict[str, object]]:
    """Iterate a local vendor Ad Banner directory without downloading assets.

    TODO: switch this adapter to a ``creative-graphic-design`` Hugging Face
    dataset once Ad Banner is imported into the org.
    """
    if source != "vendor":
        raise ValueError("LayoutDETR currently supports only source='vendor'")
    split_name = "val" if split == "validation" else "train"
    root_path = Path(root)
    json_paths = sorted(root_path.glob(f"{split_name}/**/*.json"))
    if not json_paths:
        json_paths = sorted(root_path.glob("*.json"))
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("elements", [])
        yield {"path": str(path), "elements": rows}
