import torch
import pytest

from housegan.graph_schema import (
    HouseGanRelation,
    complete_signed_edges,
    graph_to_node_features,
    normalize_scene_graph,
)


def test_scene_graph_from_labels_and_complete_edges():
    graph = normalize_scene_graph(
        None,
        labels=["living_room", "kitchen", "bedroom"],
        relations=[HouseGanRelation(0, 1, True)],
        id2label={0: "living_room", 1: "kitchen", 2: "bedroom"},
    )
    edges = complete_signed_edges(graph.nodes, graph.relations)
    assert edges.tolist() == [[0, 1, 1], [0, -1, 2], [1, -1, 2]]
    node_features = graph_to_node_features(
        graph.nodes,
        label2id={"living_room": 0, "kitchen": 1, "bedroom": 2},
        num_labels=3,
    )
    assert torch.equal(node_features.argmax(dim=-1), torch.tensor([0, 1, 2]))


def test_mapping_relations_accept_predicate_aliases():
    graph = normalize_scene_graph(
        {
            "nodes": [{"id": 0, "label": 0}, {"id": 1, "label": 1}],
            "edges": [{"source": 0, "target": 1, "predicate": "not_adjacent"}],
        },
        labels=None,
        relations=None,
        id2label={0: "a", 1: "b"},
    )
    assert complete_signed_edges(graph.nodes, graph.relations).tolist() == [[0, -1, 1]]


def test_graph_schema_error_paths_and_tuple_signs():
    with pytest.raises(ValueError):
        normalize_scene_graph(None, labels=None, relations=None, id2label={0: "a"})
    with pytest.raises(ValueError):
        normalize_scene_graph(
            {"nodes": [{"id": 0}]}, labels=None, relations=None, id2label={0: "a"}
        )
    with pytest.raises(ValueError):
        normalize_scene_graph(
            {
                "nodes": [{"id": 0, "label": 0}, {"id": 1, "label": 1}],
                "edges": [(0, "bad", 1)],
            },
            labels=None,
            relations=None,
            id2label={0: "a", 1: "b"},
        )
    graph = normalize_scene_graph(
        {
            "nodes": [{"id": 0, "label": 0}, {"id": 1, "label": 1}],
            "edges": [(0, -1, 1)],
        },
        labels=None,
        relations=None,
        id2label={0: "a", 1: "b"},
    )
    assert complete_signed_edges(
        graph.nodes, graph.relations, default_adjacent=True
    ).tolist() == [[0, -1, 1]]
    with pytest.raises(ValueError):
        graph_to_node_features(graph.nodes, label2id={"a": 0}, num_labels=1)
