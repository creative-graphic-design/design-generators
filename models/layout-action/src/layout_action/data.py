"""Dataset metadata for LayoutAction checkpoints."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final


class LayoutActionDatasetName(StrEnum):
    """Dataset names supported by the LayoutAction package."""

    rico13 = auto()
    layout_action_rico13 = auto()
    rico = auto()
    publaynet = auto()
    infoppt = auto()


RICO13_LABELS: Final[tuple[str, ...]] = (
    "Toolbar",
    "Image",
    "Text",
    "Icon",
    "Text Button",
    "Input",
    "List Item",
    "Advertisement",
    "Pager Indicator",
    "Web View",
    "Background Image",
    "Drawer",
    "Modal",
)
PUBLAYNET_LABELS: Final[tuple[str, ...]] = (
    "text",
    "title",
    "list",
    "table",
    "figure",
)
INFOPPT_LABELS: Final[tuple[str, ...]] = (
    "TEXT_BOX",
    "PICTURE",
    "CHART",
    "TABLE",
    "TITLE",
    "SUBTITLE",
)


def normalize_vendor_dataset_name(dataset_name: str | LayoutActionDatasetName) -> str:
    """Normalize public and vendor dataset aliases.

    Args:
        dataset_name: Dataset name or alias.

    Returns:
        Canonical package dataset name.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> normalize_vendor_dataset_name("rico")
        'rico13'
    """
    value = str(dataset_name).lower().replace("-", "_")
    if value in {"rico", "rico13", "layout_action_rico13"}:
        return "rico13"
    if value == "publaynet":
        return "publaynet"
    if value == "infoppt":
        return "infoppt"
    raise ValueError(f"Unsupported LayoutAction dataset_name: {dataset_name}")


def layout_action_labels(
    dataset_name: str | LayoutActionDatasetName,
) -> tuple[str, ...]:
    """Return the exact label order used by the LayoutAction vendor assets."""
    dataset = normalize_vendor_dataset_name(dataset_name)
    if dataset == "rico13":
        return RICO13_LABELS
    if dataset == "publaynet":
        return PUBLAYNET_LABELS
    if dataset == "infoppt":
        return INFOPPT_LABELS
    raise ValueError(f"Unsupported LayoutAction dataset_name: {dataset_name}")


def max_elements_for_layout_action_dataset(
    dataset_name: str | LayoutActionDatasetName,
) -> int:
    """Return the vendor maximum element count for a LayoutAction dataset."""
    dataset = normalize_vendor_dataset_name(dataset_name)
    if dataset in {"rico13", "publaynet"}:
        return 9
    if dataset == "infoppt":
        return 20
    raise ValueError(f"Unsupported LayoutAction dataset_name: {dataset_name}")


def iter_org_rico13_samples() -> None:
    """Placeholder for the org RICO adapter.

    Raises:
        NotImplementedError: Always, until the lightweight streaming adapter is
            wired without full dataset downloads.
    """
    raise NotImplementedError(
        "RICO loading must use creative-graphic-design/Rico with the "
        "ui-screenshots-and-hierarchies-with-semantic-annotations config; "
        "the streaming adapter is not implemented in this lightweight package path."
    )


def iter_org_publaynet_samples() -> None:
    """Placeholder for the org PubLayNet adapter."""
    raise NotImplementedError(
        "PubLayNet loading must avoid full downloads; use tiny builders or "
        "streaming fixtures when the adapter is implemented."
    )


def iter_vendor_infoppt_samples() -> None:
    """Placeholder for vendor-distribution-only InfoPPT loading."""
    raise NotImplementedError(
        "InfoPPT is not available in the creative-graphic-design HF org yet; "
        "use the vendor distribution until it is imported."
    )
