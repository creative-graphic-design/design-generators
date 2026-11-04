from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType

import pytest
import torch
from structure_diffusion.processing_structure_diffusion import (
    NounPhrase,
    align_phrase_context,
    constituency_noun_phrases,
    normalize_prompt,
    scene_graph_noun_phrases,
    token_span_from_char_span,
)


class WordTokenizer:
    model_max_length = 8

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.lower().split()


@dataclass
class FakeWord:
    start_char: int
    end_char: int


@dataclass
class FakeSentence:
    constituency: str
    words: list[FakeWord]


@dataclass
class FakeDocument:
    sentences: list[FakeSentence]


class FakeStanzaPipeline:
    def __init__(self, document: FakeDocument) -> None:
        self.document = document

    def __call__(self, text: str) -> FakeDocument:
        del text
        return self.document


def test_normalize_prompt_matches_source_preprocessing() -> None:
    assert normalize_prompt("  A Red Car. ") == "a red car"


def test_constituency_parser_extracts_phrase_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "a red car and a white sheep"
    offsets = [(0, 1), (2, 5), (6, 9), (10, 13), (14, 15), (16, 21), (22, 27)]
    document = FakeDocument(
        sentences=[
            FakeSentence(
                constituency=(
                    "(S (NP (DT a) (JJ red) (NN car)) (CC and) "
                    "(NP (DT a) (JJ white) (NN sheep)))"
                ),
                words=[FakeWord(*span) for span in offsets],
            )
        ]
    )
    monkeypatch.setattr(
        "structure_diffusion.processing_structure_diffusion._constituency_pipeline",
        lambda: FakeStanzaPipeline(document),
    )

    assert constituency_noun_phrases(prompt) == [
        NounPhrase("a red car", 0, 9),
        NounPhrase("a white sheep", 14, 27),
    ]


def test_constituency_parser_returns_empty_for_no_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "structure_diffusion.processing_structure_diffusion._constituency_pipeline",
        lambda: FakeStanzaPipeline(FakeDocument(sentences=[])),
    )
    assert constituency_noun_phrases(" ") == []


def test_constituency_parser_rejects_mismatched_tree_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(
        sentences=[
            FakeSentence(
                constituency="(S (NP (NN car) (NN sheep)))",
                words=[FakeWord(0, 3)],
            )
        ]
    )
    monkeypatch.setattr(
        "structure_diffusion.processing_structure_diffusion._constituency_pipeline",
        lambda: FakeStanzaPipeline(document),
    )
    with pytest.raises(ValueError, match="different lengths"):
        constituency_noun_phrases("car sheep")


def test_token_span_uses_real_bpe_positions_and_clamps_eos() -> None:
    tokenizer = WordTokenizer()
    prompt = "a red car and a white sheep"

    assert token_span_from_char_span(tokenizer, prompt, 6, 9, 8) == (3, 4)
    assert token_span_from_char_span(tokenizer, prompt, 22, 27, 6) is None


def test_token_span_handles_capitalized_prompt_with_lowercasing_tokenizer() -> None:
    prompt = "A red car and a white sheep"

    assert token_span_from_char_span(WordTokenizer(), prompt, 6, 9, 8) == (3, 4)


def test_token_span_rejects_invalid_character_ranges() -> None:
    with pytest.raises(ValueError, match="non-empty prompt span"):
        token_span_from_char_span(WordTokenizer(), "car", -1, 2, 8)


def test_align_phrase_context_copies_only_phrase_tokens() -> None:
    prompt_context = torch.zeros((1, 8, 2))
    phrase_context = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)

    aligned = align_phrase_context(
        prompt_context,
        phrase_context,
        (2, 4),
        phrase_token_count=2,
    )

    assert torch.equal(aligned[:, 2:4], phrase_context[:, 1:3])
    assert torch.equal(aligned[:, :2], prompt_context[:, :2])
    assert torch.equal(aligned[:, 4:], prompt_context[:, 4:])


def test_align_phrase_context_rejects_short_phrase_and_batch_mismatch() -> None:
    prompt_context = torch.zeros((1, 8, 2))
    phrase_context = torch.zeros((2, 8, 2))
    with pytest.raises(ValueError, match="fewer real tokens"):
        align_phrase_context(prompt_context, prompt_context, (2, 4), 1)
    with pytest.raises(ValueError, match="batches must match"):
        align_phrase_context(prompt_context, phrase_context, (2, 4), 2)


def test_align_phrase_context_rejects_invalid_span() -> None:
    with pytest.raises(ValueError, match="outside the prompt context"):
        align_phrase_context(torch.zeros((1, 8, 2)), torch.zeros((1, 8, 2)), (0, 2), 2)


def test_scene_graph_parser_handles_entities_and_deduplicates_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_graph_parser = ModuleType("sng_parser")

    def parse(prompt: str) -> dict[str, list[dict[str, int | tuple[int, int]]]]:
        del prompt
        return {
            "entities": [
                {"span_bounds": (0, 2)},
                {"span_bounds": (4, 7)},
            ],
            "relations": [{"subject": 0, "object": 1}],
        }

    setattr(scene_graph_parser, "parse", parse)
    monkeypatch.setitem(sys.modules, "sng_parser", scene_graph_parser)
    assert scene_graph_noun_phrases("a red car and a white sheep") == [
        NounPhrase("a red", 0, 5),
        NounPhrase("a white sheep", 14, 27),
    ]
