"""Tests for LayoutGPT exemplar selection."""

from layout_gpt.exemplars import LayoutExample, select_fixed_random, select_k_similar


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
