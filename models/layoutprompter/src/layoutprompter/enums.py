"""Closed string vocabularies used by the LayoutPrompter package."""

from __future__ import annotations

from enum import StrEnum, auto


class LayoutPrompterDataset(StrEnum):
    """LayoutPrompter-only dataset names absent from the shared registry."""

    posterlayout = auto()
    webui = auto()


class PromptFormat(StrEnum):
    """Supported LayoutPrompter prompt encodings."""

    SEQ = auto()
    HTML = auto()


class OutputType(StrEnum):
    """Supported return containers for the shared call interface."""

    DATACLASS = auto()
    DICT = auto()


class LayoutPrompterTask(StrEnum):
    """Released task keys supported by LayoutPrompter."""

    gent = auto()
    gents = auto()
    genr = auto()
    completion = auto()
    refinement = auto()
    content = auto()
    text = auto()


def normalize_layoutprompter_task(
    task: LayoutPrompterTask | str,
) -> LayoutPrompterTask:
    """Return a LayoutPrompter task enum from a public or release string."""
    try:
        return LayoutPrompterTask(task)
    except ValueError as exc:
        raise ValueError(f"Unsupported LayoutPrompter task: {task}") from exc
