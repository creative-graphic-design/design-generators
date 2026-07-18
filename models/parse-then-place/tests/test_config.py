from __future__ import annotations

from typing import cast

from parse_then_place import ParseThenPlaceConfig


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
