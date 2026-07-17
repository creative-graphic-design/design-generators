"""Dataset constants used by LayoutPrompter prompts and parsing."""

from __future__ import annotations

DATASET_LABELS: dict[str, tuple[str, ...]] = {
    "publaynet": ("text", "title", "list", "table", "figure"),
    "rico": (
        "text",
        "image",
        "icon",
        "list item",
        "text button",
        "toolbar",
        "web view",
        "input",
        "card",
        "advertisement",
        "background image",
        "drawer",
        "radio button",
        "checkbox",
        "multi-tab",
        "pager indicator",
        "modal",
        "on/off switch",
        "slider",
        "map view",
        "button bar",
        "video",
        "bottom navigation",
        "number stepper",
        "date picker",
    ),
    "posterlayout": ("text", "logo", "underlay"),
    "webui": (
        "text",
        "link",
        "button",
        "title",
        "description",
        "image",
        "background",
        "logo",
        "icon",
        "input",
    ),
}

CANVAS_SIZE: dict[str, tuple[int, int]] = {
    "rico": (90, 160),
    "publaynet": (120, 160),
    "posterlayout": (102, 150),
    "webui": (120, 120),
}

LAYOUT_DOMAIN: dict[str, str] = {
    "rico": "android",
    "publaynet": "document",
    "posterlayout": "poster",
    "webui": "web",
}


def id2label(dataset: str) -> dict[int, str]:
    """Return public 0-based dataset-local label mapping."""
    return dict(enumerate(DATASET_LABELS[dataset]))


def label2id(dataset: str) -> dict[str, int]:
    """Return public 0-based dataset-local label ids."""
    return {label: index for index, label in id2label(dataset).items()}
