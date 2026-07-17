"""Tests for LayoutGPT exemplar selection."""

import json
from pathlib import Path

import pytest

from layout_gpt.enums import LayoutGPTSetting
from layout_gpt.exemplars import (
    LayoutExample,
    load_nsr_examples,
    select_fixed_random,
    select_k_similar,
)


def test_vendor_records_load_counting_and_spatial_settings(tmp_path: Path) -> None:
    counting = LayoutExample.from_vendor_record(
        {
            "id": 1,
            "prompt": "one clock",
            "object_list": [("clock", [0.1, 0.2, 0.3, 0.4])],
        },
        setting=LayoutGPTSetting.counting,
    )
    spatial = LayoutExample.from_vendor_record(
        {
            "id": "spatial-1",
            "prompt": "clock left of cat",
            "obj1": ("clock", [0.1, 0.2, 0.3, 0.4]),
            "obj2": ("cat", [0.5, 0.6, 0.7, 0.8]),
        },
        setting="spatial",
    )

    assert counting.objects == (("clock", (0.1, 0.2, 0.3, 0.4)),)
    assert spatial.objects == (
        ("clock", (0.1, 0.2, 0.3, 0.4)),
        ("cat", (0.5, 0.6, 0.7, 0.8)),
    )

    examples_path = tmp_path / "counting.json"
    examples_path.write_text(
        json.dumps(
            [
                {
                    "id": 2,
                    "prompt": "one chair",
                    "object_list": [("chair", [0.0, 0.1, 0.2, 0.3])],
                }
            ]
        )
    )
    loaded = load_nsr_examples(examples_path, setting="counting")

    assert loaded[0].id == 2
    assert loaded[0].objects == (("chair", (0.0, 0.1, 0.2, 0.3)),)


def test_fixed_random_matches_vendor_seed_strategy() -> None:
    examples = [
        LayoutExample(id=index, prompt=str(index), objects=(), metadata={})
        for index in range(5)
    ]

    assert [example.id for example in select_fixed_random(examples, k=3)] == [3, 1, 2]


def test_k_similar_uses_embedding_ranking_without_clip_dependency() -> None:
    examples = [
        LayoutExample(id="x", prompt="x", objects=(), metadata={}),
        LayoutExample(id="y", prompt="y", objects=(), metadata={}),
        LayoutExample(id="z", prompt="z", objects=(), metadata={}),
    ]

    selected = select_k_similar(
        examples,
        query="query",
        k=2,
        query_embedding=lambda _query: (0.0, 1.0),
        example_embeddings=[(1.0, 0.0), (0.0, 0.9), (0.0, 0.8)],
    )

    assert [example.id for example in selected] == ["y", "z"]


def test_k_similar_validates_embedding_inputs() -> None:
    examples = [LayoutExample(id="x", prompt="x", objects=(), metadata={})]

    with pytest.raises(ValueError, match="length"):
        select_k_similar(
            examples,
            query="query",
            k=1,
            query_embedding=lambda _query: (1.0,),
            example_embeddings=[],
        )

    with pytest.raises(ValueError, match="non-zero"):
        select_k_similar(
            examples,
            query="query",
            k=1,
            query_embedding=lambda _query: (0.0,),
            example_embeddings=[(1.0,)],
        )
