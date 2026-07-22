import numpy as np
import torch

from housegan.datasets import (
    build_edges_from_bboxes,
    load_housegan_numpy,
    normalize_vendor_graph_row,
    split_target_set,
)
from housegan.graph_schema import HouseGanSceneGraph
from housegan.visualization import render_layout


def test_dataset_adapters_and_split(tmp_path):
    row = [[1, 2], np.array([[0, 0, 64, 64], [64, 0, 128, 64]], dtype=np.float32)]
    data_path = tmp_path / "train_data.npy"
    array = np.empty(1, dtype=object)
    array[0] = row
    np.save(data_path, array)
    loaded = load_housegan_numpy(data_path)
    graph = normalize_vendor_graph_row(loaded[0])
    assert isinstance(graph, HouseGanSceneGraph)
    assert [node.label for node in graph.nodes] == [0, 1]
    assert split_target_set([graph], target_set="A", split="eval") == [graph]
    assert split_target_set([graph], target_set="A", split="train") == []


def test_bbox_relations_and_render_layout():
    relations = build_edges_from_bboxes(
        np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]], dtype=np.float32)
    )
    assert relations[0].adjacent
    image = render_layout(
        torch.tensor([[0.25, 0.25, 0.5, 0.5]]),
        torch.tensor([0]),
        id2label={0: "living_room"},
        canvas_size=(32, 32),
    )
    assert image.size == (32, 32)
