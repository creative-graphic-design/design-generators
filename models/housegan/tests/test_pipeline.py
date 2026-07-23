import pytest
import torch
from typing import cast

from housegan import HouseGanConfig, HouseGanGenerator, HouseGanPipeline
from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput


def _pipe():
    return HouseGanPipeline(HouseGanGenerator(HouseGanConfig()))


def test_pipeline_smoke_and_schema():
    output = _pipe()(
        condition_type="relation",
        scene_graph={
            "nodes": [{"id": 0, "label": "living_room"}, {"id": 1, "label": "kitchen"}],
            "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
        },
        seed=0,
    )
    output = cast(LayoutGenerationOutput, output)
    assert_layout_output_schema(output, batch_size=1)
    assert tuple(output.labels.shape) == (1, 2)


def test_pipeline_generator_wins_over_seed():
    graph = {
        "nodes": [{"id": 0, "label": 0}, {"id": 1, "label": 1}],
        "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
    }
    gen_a = torch.Generator(device="cpu").manual_seed(123)
    gen_b = torch.Generator(device="cpu").manual_seed(123)
    out_a = _pipe()(scene_graph=graph, generator=gen_a, seed=1)
    out_b = _pipe()(scene_graph=graph, generator=gen_b, seed=999)
    out_a = cast(LayoutGenerationOutput, out_a)
    out_b = cast(LayoutGenerationOutput, out_b)
    assert torch.equal(cast(torch.Tensor, out_a.bbox), cast(torch.Tensor, out_b.bbox))


def test_save_pretrained_from_pretrained(tmp_path):
    pipe = _pipe()
    pipe.save_pretrained(tmp_path)
    loaded = HouseGanPipeline.from_pretrained(tmp_path, local_files_only=True)
    output = loaded(labels=[0, 1], relations=[(0, "adjacent", 1)], seed=0)
    output = cast(LayoutGenerationOutput, output)
    assert tuple(output.labels.shape) == (1, 2)


def test_pipeline_batch_and_dict_output():
    graph = {
        "nodes": [{"id": 0, "label": 0}, {"id": 1, "label": 1}],
        "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
    }
    output = _pipe()(scene_graph=graph, batch_size=2, seed=4, output_type="dict")
    assert isinstance(output, dict)
    assert cast(torch.Tensor, output["bbox"]).shape == (2, 2, 4)
    with pytest.raises(ValueError):
        _pipe()(scene_graph=graph, batch_size=0)
