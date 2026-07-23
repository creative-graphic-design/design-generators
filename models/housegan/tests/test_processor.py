import pytest
import torch
from typing import cast

from housegan import HouseGanConfig, HouseGanProcessor
from laygen.modeling_outputs import LayoutGenerationOutput
from housegan.processing_housegan import mask_to_ltrb


def test_processor_relation_payload_and_mask_postprocess():
    processor = HouseGanProcessor(config=HouseGanConfig())
    encoded = processor(
        condition_type="graph",
        scene_graph={
            "nodes": [{"id": 0, "label": "living_room"}, {"id": 1, "label": "kitchen"}],
            "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
        },
    )
    assert encoded["node_features"].shape == (2, 10)
    assert encoded["edges"].tolist() == [[0, 1, 1]]
    masks = torch.zeros(2, 32, 32)
    masks[0, 1:4, 2:6] = 1
    masks[1, 10:12, 11:15] = 1
    output = processor.post_process_masks(
        masks,
        labels=encoded["labels"],
        edges=encoded["edges"],
        return_intermediates=True,
    )
    output = cast(LayoutGenerationOutput, output)
    assert tuple(output.bbox.shape) == (1, 2, 4)
    assert output.mask.tolist() == [[True, True]]
    assert "room_masks" in cast(dict[str, object], output.intermediates)


def test_unsupported_condition_raises():
    with pytest.raises(NotImplementedError):
        HouseGanProcessor(config=HouseGanConfig())(
            condition_type="unconditional", labels=[0]
        )


def test_mask_to_ltrb_vendor_inclusive_behavior():
    mask = torch.zeros(1, 32, 32)
    mask[0, 4:6, 7:9] = 1
    assert mask_to_ltrb(mask).tolist() == [[7.0, 4.0, 9.0, 6.0]]


def test_processor_bbox_relations_and_save_load(tmp_path):
    processor = HouseGanProcessor(
        config=HouseGanConfig(), default_missing_relation="error"
    )
    with pytest.raises(ValueError):
        processor(labels=[0, 1])
    encoded = processor(
        labels=[0, 1],
        bbox=[[0.25, 0.25, 0.5, 0.5], [0.75, 0.25, 0.5, 0.5]],
        box_format="xywh",
    )
    assert encoded["edges"].shape == (1, 3)
    processor.save_pretrained(tmp_path)
    loaded = HouseGanProcessor.from_pretrained(tmp_path)
    assert loaded.id2label[0] == "living_room"
