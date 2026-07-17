"""LayoutGPT exemplar loading, selection, and serialization."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never, cast

from layout_gpt.enums import LayoutGPTSetting, coerce_enum


EmbeddingProvider = Callable[[str], Sequence[float]]
DEFAULT_FIXED_RANDOM_SEED: Final[int] = 42


@dataclass(frozen=True)
class LayoutExample:
    """One NSR-1K 2D LayoutGPT exemplar."""

    id: int | str
    prompt: str
    objects: tuple[tuple[str, tuple[float, float, float, float]], ...]
    metadata: dict[str, object]

    @classmethod
    def from_vendor_record(
        cls, record: Mapping[str, object], *, setting: str | LayoutGPTSetting
    ) -> LayoutExample:
        """Build an exemplar from the original NSR-1K JSON record."""
        normalized_setting = coerce_enum(setting, LayoutGPTSetting)
        if normalized_setting is LayoutGPTSetting.counting:
            raw_objects = cast(
                Sequence[tuple[str, Sequence[float]]], record["object_list"]
            )
        elif normalized_setting is LayoutGPTSetting.spatial:
            raw_objects = cast(
                Sequence[tuple[str, Sequence[float]]],
                [record["obj1"], record["obj2"]],
            )
        else:
            assert_never(normalized_setting)
        objects = tuple(
            (
                str(label),
                (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
            )
            for label, bbox in raw_objects
        )
        return cls(
            id=cast(int | str, record["id"]),
            prompt=str(record["prompt"]),
            objects=objects,
            metadata=dict(record),
        )


def load_nsr_examples(
    path: str | Path, *, setting: str | LayoutGPTSetting
) -> list[LayoutExample]:
    """Load LayoutGPT NSR-1K examples from a vendor-style JSON file."""
    records = cast(Sequence[Mapping[str, object]], json.loads(Path(path).read_text()))
    return [
        LayoutExample.from_vendor_record(record, setting=setting) for record in records
    ]


def select_fixed_random(
    examples: Sequence[LayoutExample],
    *,
    k: int,
    seed: int = DEFAULT_FIXED_RANDOM_SEED,
) -> list[LayoutExample]:
    """Select exemplars with the vendor's fixed ``random.seed(42)`` strategy."""
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:k]


def select_k_similar(
    examples: Sequence[LayoutExample],
    *,
    query: str,
    k: int,
    query_embedding: EmbeddingProvider,
    example_embeddings: Sequence[Sequence[float]],
) -> list[LayoutExample]:
    """Select top-k exemplars by CLIP-style cosine similarity.

    The original code computes ``softmax(100 * query @ train.T)`` and then
    ``topk``. Softmax preserves ranking, so this implementation keeps the same
    order without requiring torch or CLIP in the core package.
    """
    if len(examples) != len(example_embeddings):
        msg = "example_embeddings length must match examples"
        raise ValueError(msg)
    query_vector = _normalize(query_embedding(query))
    scores = [
        sum(q * e for q, e in zip(query_vector, _normalize(embedding), strict=True))
        for embedding in example_embeddings
    ]
    top_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return [examples[index] for index in top_indices]


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        msg = "embedding vectors must be non-zero"
        raise ValueError(msg)
    return tuple(value / norm for value in values)
