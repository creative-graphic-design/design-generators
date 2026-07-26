"""Shared public APIs for layout-generation packages."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BoxFormat",
    "ConditionType",
    "ConditionAlias",
    "DatasetName",
    "ParityMetric",
    "RICO25_INTERACTION_LABEL_NAMES",
    "SamplingMode",
    "WhitespaceTokenizerMixin",
    "WEBUI_BASE_LABEL_NAMES",
    "build_layout_model_card",
    "build_token_maps",
    "convert_id_to_token",
    "convert_token_to_id",
    "join_tokens",
    "layoutdm_model_card",
    "max_elements_for_dataset",
    "normalize_box_format",
    "normalize_condition_type",
    "normalize_dataset_name",
    "normalize_enum_value",
    "normalize_sampling_mode",
    "save_json_vocabulary",
    "sanitize_for_yaml",
    "split_whitespace_tokens",
]

if TYPE_CHECKING:
    from .bbox import BoxFormat, normalize_box_format
    from .conditions import ConditionAlias, ConditionType, normalize_condition_type
    from .discrete import SamplingMode, normalize_sampling_mode
    from .enums import normalize_enum_value
    from .labels import (
        RICO25_INTERACTION_LABEL_NAMES,
        WEBUI_BASE_LABEL_NAMES,
        DatasetName,
        max_elements_for_dataset,
        normalize_dataset_name,
    )
    from .model_card import ParityMetric, build_layout_model_card, layoutdm_model_card
    from .serialization import sanitize_for_yaml
    from .tokenization import (
        WhitespaceTokenizerMixin,
        build_token_maps,
        convert_id_to_token,
        convert_token_to_id,
        join_tokens,
        save_json_vocabulary,
        split_whitespace_tokens,
    )

_EXPORT_MODULES = {
    "BoxFormat": "bbox",
    "ConditionAlias": "conditions",
    "ConditionType": "conditions",
    "DatasetName": "labels",
    "ParityMetric": "model_card",
    "RICO25_INTERACTION_LABEL_NAMES": "labels",
    "SamplingMode": "discrete",
    "WhitespaceTokenizerMixin": "tokenization",
    "WEBUI_BASE_LABEL_NAMES": "labels",
    "build_layout_model_card": "model_card",
    "build_token_maps": "tokenization",
    "convert_id_to_token": "tokenization",
    "convert_token_to_id": "tokenization",
    "join_tokens": "tokenization",
    "layoutdm_model_card": "model_card",
    "max_elements_for_dataset": "labels",
    "normalize_box_format": "bbox",
    "normalize_condition_type": "conditions",
    "normalize_dataset_name": "labels",
    "normalize_enum_value": "enums",
    "normalize_sampling_mode": "discrete",
    "save_json_vocabulary": "tokenization",
    "sanitize_for_yaml": "serialization",
    "split_whitespace_tokens": "tokenization",
}


def __getattr__(
    name: str,
) -> type | Callable[..., None] | tuple[str, ...] | list[str]:
    """Lazily load public re-exports without pre-importing submodules."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
