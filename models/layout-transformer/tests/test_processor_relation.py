from typing import Literal, cast

import torch

from laygen.modeling_outputs import LayoutGenerationOutput
from layout_transformer import (
    LayoutObject,
    LayoutRelation,
    LayoutTransformerModelOutput,
    LayoutTransformerProcessor,
    LayoutTransformerRelationTokenizer,
)


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


def test_processor_accepts_integer_ids_and_overrides_max_length():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person", 2: "table"},
        relation_id2label={1: "left of"},
        max_sequence_length=8,
    )

    encoded = processor(
        objects=[LayoutObject(id="a", label=1), LayoutObject(id="b", label=2)],
        relations=[LayoutRelation(subject="a", predicate=1, object="b")],
        max_sequence_length=6,
    )

    assert encoded["input_token"].shape == (1, 6)
    assert processor.max_sequence_length == 8


def test_processor_rejects_unknown_labels_and_tensor_type():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person"},
        relation_id2label={1: "left of"},
    )

    try:
        processor(objects=[LayoutObject(id="a", label="missing")])
    except ValueError as exc:
        assert "Unknown object label" in str(exc)
    else:
        raise AssertionError("expected unknown object label error")

    try:
        processor(
            objects=[LayoutObject(id="a", label="person")],
            return_tensors=cast(Literal["pt", "np"], "jax"),
        )
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("expected return_tensors error")

    try:
        processor(
            objects=[LayoutObject(id="a", label="person")],
            relations=[LayoutRelation(subject="a", predicate="missing", object="a")],
        )
    except ValueError as exc:
        assert "Unknown relation label" in str(exc)
    else:
        raise AssertionError("expected unknown relation label error")


def test_processor_postprocess_coarse_boxes_and_intermediates():
    processor = LayoutTransformerProcessor.from_config(id2label={0: "__image__"})
    model_outputs = LayoutTransformerModelOutput(
        coarse_box=torch.full((1, 3, 4), 0.5),
        vocab_logits=torch.zeros(1, 3, 4),
        obj_id_logits=torch.zeros(1, 3, 4),
        token_type_logits=torch.zeros(1, 3, 4),
    )

    output = cast(
        LayoutGenerationOutput,
        processor.post_process_layout_generation(
            model_outputs,
            input_obj_id=torch.tensor([[0, 1, 0]]),
            token_type=torch.tensor([[0, 1, 0]]),
            return_intermediates=True,
        ),
    )

    assert output.bbox.shape == (1, 1, 4)
    intermediates = cast(dict[str, object], output.intermediates)
    assert intermediates["coarse_box"] is model_outputs.coarse_box


def test_processor_postprocess_requires_boxes():
    processor = LayoutTransformerProcessor.from_config()

    try:
        processor.post_process_layout_generation(
            LayoutTransformerModelOutput(),
            input_obj_id=torch.tensor([[0]]),
            token_type=torch.tensor([[0]]),
        )
    except ValueError as exc:
        assert "coarse_box or refine_box" in str(exc)
    else:
        raise AssertionError("expected missing boxes error")


def test_processor_object_reduce_deduplicates_repeated_relations():
    tokenizer = LayoutTransformerRelationTokenizer(tokens=["person", "table"])
    model_outputs = LayoutTransformerModelOutput(
        refine_box=torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3, 0.4],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.9, 1.0, 1.1, 1.2],
                ]
            ]
        )
    )
    input_obj_id = torch.tensor([[0, 1, 0, 1, 0, 2]])
    token_type = torch.tensor([[0, 1, 2, 3, 0, 1]])
    input_token = torch.tensor([[0, 11, 0, 12, 0, 21]])

    outputs = {}
    for reduce in ("first", "last", "mean"):
        processor = LayoutTransformerProcessor(
            tokenizer=tokenizer,
            id2label={0: "__image__", 11: "person", 21: "table"},
            object_reduce=reduce,
        )
        outputs[reduce] = cast(
            LayoutGenerationOutput,
            processor.post_process_layout_generation(
                model_outputs,
                input_token=input_token,
                input_obj_id=input_obj_id,
                token_type=token_type,
            ),
        )

    assert torch.allclose(
        outputs["first"].bbox[0, 0], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    assert outputs["first"].labels[0, 0].item() == 11
    assert torch.allclose(
        outputs["last"].bbox[0, 0], torch.tensor([0.5, 0.6, 0.7, 0.8])
    )
    assert outputs["last"].labels[0, 0].item() == 12
    assert torch.allclose(
        outputs["mean"].bbox[0, 0], torch.tensor([0.3, 0.4, 0.5, 0.6])
    )
    assert outputs["mean"].labels[0, 0].item() == 11
    assert outputs["mean"].mask.tolist() == [[True, True]]


def test_processor_rejects_missing_graph():
    processor = LayoutTransformerProcessor.from_config()

    try:
        processor()
    except ValueError as exc:
        assert "scene_graph or objects" in str(exc)
    else:
        raise AssertionError("expected missing graph error")
