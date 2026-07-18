from __future__ import annotations

from typing import cast

import pytest

from parse_then_place import ParseThenPlaceConfig
from parse_then_place.labels import (
    ParseThenPlaceDatasetName,
    Stage2Mode,
    canvas_size_for_dataset,
    label2id_for_dataset,
    normalize_dataset_name,
    normalize_stage2_mode,
)


def test_config_roundtrip_preserves_dataset_defaults() -> None:
    config = ParseThenPlaceConfig(dataset_name="web", stage2_mode="pretrain")

    restored = ParseThenPlaceConfig.from_dict(config.to_dict())

    assert restored.dataset_name == "web"
    assert restored.stage2_mode == "pretrain"
    assert restored.canvas_size == (120, 120)
    id2label = cast(dict[int, str], restored.id2label)
    assert id2label[11] == "textarea"
    assert restored.placement_subfolder == "placement"
    assert restored.parser_subfolder == "semantic_parser"


def test_dataset_and_stage2_name_normalization() -> None:
    assert (
        normalize_dataset_name(ParseThenPlaceDatasetName.rico)
        is ParseThenPlaceDatasetName.rico
    )
    assert normalize_dataset_name("webui") is ParseThenPlaceDatasetName.web
    assert normalize_stage2_mode(Stage2Mode.pretrain) is Stage2Mode.pretrain
    assert normalize_stage2_mode("finetune") is Stage2Mode.finetune
    assert canvas_size_for_dataset("web") == (120, 120)
    assert label2id_for_dataset("rico")["text"] == 4


def test_invalid_dataset_and_stage2_mode_raise() -> None:
    with pytest.raises(ValueError, match="Unsupported Parse-Then-Place dataset"):
        normalize_dataset_name("publaynet")

    with pytest.raises(ValueError, match="Unsupported stage2_mode"):
        normalize_stage2_mode("best")
