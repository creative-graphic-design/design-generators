"""Dataset constants used by LayoutPrompter prompts and parsing."""

from __future__ import annotations

from typing import Final, TypeAlias

from laygen.common import DatasetName, normalize_dataset_name

from layoutprompter.enums import LayoutPrompterDataset


SupportedDataset: TypeAlias = DatasetName | LayoutPrompterDataset


DATASET_LABELS: Final[dict[SupportedDataset, tuple[str, ...]]] = {
    DatasetName.publaynet: ("text", "title", "list", "table", "figure"),
    DatasetName.rico25: (
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
    LayoutPrompterDataset.posterlayout: ("text", "logo", "underlay"),
    LayoutPrompterDataset.webui: (
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

CANVAS_SIZE: Final[dict[SupportedDataset, tuple[int, int]]] = {
    DatasetName.rico25: (90, 160),
    DatasetName.publaynet: (120, 160),
    LayoutPrompterDataset.posterlayout: (102, 150),
    LayoutPrompterDataset.webui: (120, 120),
}

LAYOUT_DOMAIN: Final[dict[SupportedDataset, str]] = {
    DatasetName.rico25: "android",
    DatasetName.publaynet: "document",
    LayoutPrompterDataset.posterlayout: "poster",
    LayoutPrompterDataset.webui: "web",
}


def normalize_dataset(dataset: SupportedDataset | str) -> SupportedDataset:
    """Return a supported dataset enum from a public string value."""
    if isinstance(dataset, DatasetName | LayoutPrompterDataset):
        return dataset
    try:
        shared_dataset = normalize_dataset_name(dataset)
    except ValueError:
        pass
    else:
        if shared_dataset in DATASET_LABELS:
            return shared_dataset
    try:
        return LayoutPrompterDataset(dataset)
    except ValueError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def id2label(dataset: SupportedDataset | str) -> dict[int, str]:
    """Return public 0-based dataset-local label mapping."""
    return dict(enumerate(DATASET_LABELS[normalize_dataset(dataset)]))


def label2id(dataset: SupportedDataset | str) -> dict[str, int]:
    """Return public 0-based dataset-local label ids."""
    return {label: index for index, label in id2label(dataset).items()}
