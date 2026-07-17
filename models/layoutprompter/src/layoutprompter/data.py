"""Dataset constants used by LayoutPrompter prompts and parsing."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class LayoutPrompterDataset(StrEnum):
    """Supported LayoutPrompter dataset vocabularies."""

    PUBLAYNET = "publaynet"
    RICO = "rico"
    POSTERLAYOUT = "posterlayout"
    WEBUI = "webui"


DATASET_LABELS: Final[dict[LayoutPrompterDataset, tuple[str, ...]]] = {
    LayoutPrompterDataset.PUBLAYNET: ("text", "title", "list", "table", "figure"),
    LayoutPrompterDataset.RICO: (
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
    LayoutPrompterDataset.POSTERLAYOUT: ("text", "logo", "underlay"),
    LayoutPrompterDataset.WEBUI: (
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

CANVAS_SIZE: Final[dict[LayoutPrompterDataset, tuple[int, int]]] = {
    LayoutPrompterDataset.RICO: (90, 160),
    LayoutPrompterDataset.PUBLAYNET: (120, 160),
    LayoutPrompterDataset.POSTERLAYOUT: (102, 150),
    LayoutPrompterDataset.WEBUI: (120, 120),
}

LAYOUT_DOMAIN: Final[dict[LayoutPrompterDataset, str]] = {
    LayoutPrompterDataset.RICO: "android",
    LayoutPrompterDataset.PUBLAYNET: "document",
    LayoutPrompterDataset.POSTERLAYOUT: "poster",
    LayoutPrompterDataset.WEBUI: "web",
}


def normalize_dataset(dataset: LayoutPrompterDataset | str) -> LayoutPrompterDataset:
    """Return a supported dataset enum from a public string value."""
    try:
        return LayoutPrompterDataset(dataset)
    except ValueError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def id2label(dataset: LayoutPrompterDataset | str) -> dict[int, str]:
    """Return public 0-based dataset-local label mapping."""
    return dict(enumerate(DATASET_LABELS[normalize_dataset(dataset)]))


def label2id(dataset: LayoutPrompterDataset | str) -> dict[str, int]:
    """Return public 0-based dataset-local label ids."""
    return {label: index for index, label in id2label(dataset).items()}
