"""Shared LayoutFormer++ dataset and checkpoint-task helpers."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final

from laygen.common import ConditionType, DatasetName, normalize_condition_type
from laygen.common.labels import normalize_dataset_name


class LayoutFormerPPTask(StrEnum):
    """Supported converted LayoutFormer++ checkpoint variants."""

    ugen = auto()
    gen_t = auto()
    gen_ts = auto()
    gen_r = auto()
    completion = auto()
    refinement = auto()


class OutputType(StrEnum):
    """Supported post-processing return shapes."""

    dataclass = auto()
    dict = auto()


TASK_TO_CONDITION: Final[dict[LayoutFormerPPTask, ConditionType]] = {
    LayoutFormerPPTask.ugen: ConditionType.unconditional,
    LayoutFormerPPTask.gen_t: ConditionType.label,
    LayoutFormerPPTask.gen_ts: ConditionType.label_size,
    LayoutFormerPPTask.gen_r: ConditionType.relation,
    LayoutFormerPPTask.completion: ConditionType.completion,
    LayoutFormerPPTask.refinement: ConditionType.refinement,
}
DEFAULT_TASK_FOR_CONDITION: Final[dict[ConditionType, LayoutFormerPPTask]] = {
    condition: task for task, condition in TASK_TO_CONDITION.items()
}
SUPPORTED_CONDITIONS: Final[frozenset[ConditionType]] = frozenset(
    TASK_TO_CONDITION.values()
)
SUPPORTED_DATASETS: Final[frozenset[DatasetName]] = frozenset(
    {DatasetName.rico25, DatasetName.publaynet}
)
DATASET_TO_VENDOR_SLUG: Final[dict[DatasetName, str]] = {
    DatasetName.rico25: "rico",
    DatasetName.publaynet: "publaynet",
}


def normalize_layoutformerpp_dataset(dataset: DatasetName | str) -> DatasetName:
    """Normalize public dataset aliases to a LayoutFormer++ supported dataset."""
    normalized = normalize_dataset_name(dataset)
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported LayoutFormer++ dataset: {dataset}")
    return normalized


def layoutformerpp_dataset_slug(dataset: DatasetName | str) -> str:
    """Return the dataset slug for a LayoutFormer++ dataset."""
    return DATASET_TO_VENDOR_SLUG[normalize_layoutformerpp_dataset(dataset)]


def normalize_layoutformerpp_task(
    task: LayoutFormerPPTask | ConditionType | str,
) -> LayoutFormerPPTask:
    """Normalize checkpoint variant aliases to the internal task enum."""
    if isinstance(task, LayoutFormerPPTask):
        return task
    try:
        return LayoutFormerPPTask(task)
    except ValueError:
        condition = normalize_condition_type(task)
    try:
        return DEFAULT_TASK_FOR_CONDITION[condition]
    except KeyError as exc:
        raise ValueError(f"Unsupported LayoutFormer++ task: {task}") from exc
