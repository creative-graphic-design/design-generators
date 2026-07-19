from layout_transformer import LayoutObject, LayoutRelation, LayoutTransformerProcessor


def test_processor_serializes_relation_graph():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person", 2: "table"},
        relation_id2label={1: "left of"},
        max_sequence_length=8,
    )

    encoded = processor(
        objects=[
            LayoutObject(id="a", label="person"),
            LayoutObject(id="b", label="table"),
        ],
        relations=[LayoutRelation(subject="a", predicate="left of", object="b")],
    )

    assert encoded["input_token"].shape == (1, 8)
    assert encoded["input_token"][0, 0].item() == processor.tokenizer.cls_token_id
    assert encoded["input_obj_id"][0, :5].tolist() == [0, 1, 0, 2, 0]
    assert encoded["token_type"][0, :5].tolist() == [0, 1, 2, 3, 0]
    assert encoded["src_mask"][0, 0, :5].tolist() == [True, True, True, True, True]


def test_processor_rejects_unsupported_condition():
    processor = LayoutTransformerProcessor.from_config()

    try:
        processor(objects=[LayoutObject(id="a", label=0)], condition_type="label")
    except ValueError as exc:
        assert "only supports" in str(exc)
    else:
        raise AssertionError("expected unsupported condition error")


def test_processor_accepts_mapping_scene_graph_and_numpy_output():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person", 2: "table"},
        relation_id2label={1: "left of"},
        max_sequence_length=8,
    )
    encoded = processor(
        scene_graph={
            "nodes": [
                {"id": "a", "label": "person"},
                {"id": "b", "label": "table"},
            ],
            "edges": [
                {"source": "a", "predicate": "left of", "target": "b", "score": 1.0}
            ],
        },
        return_tensors="np",
    )

    assert encoded["input_token"].shape == (1, 8)


def test_processor_rejects_missing_graph():
    processor = LayoutTransformerProcessor.from_config()

    try:
        processor()
    except ValueError as exc:
        assert "scene_graph or objects" in str(exc)
    else:
        raise AssertionError("expected missing graph error")
