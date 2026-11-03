"""Prompt parsing and CLIP token alignment for structured diffusion guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, cast

import torch
from jaxtyping import Float
from nltk.tree import Tree


class _StanzaWord(Protocol):
    start_char: int | None
    end_char: int | None


class _StanzaSentence(Protocol):
    constituency: str
    words: list[_StanzaWord]


class _StanzaDocument(Protocol):
    sentences: list[_StanzaSentence]


class _StanzaPipeline(Protocol):
    def __call__(self, text: str) -> _StanzaDocument: ...


class _Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class NounPhrase:
    """A parsed noun phrase with character offsets into normalized prompt text."""

    text: str
    start: int
    end: int


def normalize_prompt(prompt: str) -> str:
    """Normalize casing and trailing punctuation before parsing and token alignment."""
    return prompt.lower().strip().strip(".").strip()


def constituency_noun_phrases(prompt: str) -> list[NounPhrase]:
    """Parse a prompt and return its multi-token noun phrases.

    The full prompt is represented separately by the pipeline. Returned offsets
    refer to normalized prompt text so they can be aligned to CLIP BPE tokens.
    Stanza models must be installed before this function runs.
    """
    document = _constituency_pipeline()(prompt)
    if not document.sentences:
        return []

    sentence = document.sentences[0]
    tree = Tree.fromstring(str(sentence.constituency))
    word_offsets: list[tuple[int, int]] = []
    for word in sentence.words:
        if word.start_char is None or word.end_char is None:
            raise ValueError("The constituency parser did not return word offsets.")

        word_offsets.append((word.start_char, word.end_char))

    if len(tree.leaves()) != len(word_offsets):
        raise ValueError(
            "The constituency tree and parser word offsets have different lengths."
        )

    phrases: list[NounPhrase] = []

    def visit(node: Tree | str, word_start: int) -> int:
        if isinstance(node, str):
            return 1

        cursor = word_start
        for child in node:
            cursor += visit(child, cursor)

        if node.label() == "NP" and cursor - word_start > 1:
            char_start = word_offsets[word_start][0]
            char_end = word_offsets[cursor - 1][1]
            text = prompt[char_start:char_end].strip()
            if text and text != prompt:
                phrases.append(NounPhrase(text=text, start=char_start, end=char_end))

        return cursor - word_start

    visit(tree, 0)
    return phrases


def scene_graph_noun_phrases(prompt: str) -> list[NounPhrase]:
    """Return entity and relation spans from the optional scene-graph parser."""
    try:
        import sng_parser
    except ImportError as exc:
        raise ImportError(
            "Install structurediffusion[scene-graph] to use parser_type='scene_graph'."
        ) from exc

    word_offsets = [
        (match.start(), match.end()) for match in re.finditer(r"\S+", prompt)
    ]
    graph = sng_parser.parse(prompt)
    spans: list[tuple[int, int]] = []

    for entity in graph["entities"]:
        start, end = entity["span_bounds"]
        if end - start > 1 and 0 <= start < end <= len(word_offsets):
            spans.append((start, end))

    for relation in graph["relations"]:
        subject_start, subject_end = graph["entities"][relation["subject"]][
            "span_bounds"
        ]
        object_start, object_end = graph["entities"][relation["object"]]["span_bounds"]
        start = min(subject_start, object_start)
        end = max(subject_end, object_end)
        if end - start > 1 and 0 <= start < end <= len(word_offsets):
            spans.append((start, end))

    phrases: list[NounPhrase] = []
    seen: set[tuple[int, int]] = set()
    for start, end in spans:
        if (start, end) in seen:
            continue

        seen.add((start, end))
        char_start = word_offsets[start][0]
        char_end = word_offsets[end - 1][1]
        text = prompt[char_start:char_end].strip()
        if text and text != prompt:
            phrases.append(NounPhrase(text=text, start=char_start, end=char_end))

    return phrases


def token_span_from_char_span(
    tokenizer: _Tokenizer,
    prompt: str,
    start: int,
    end: int,
    sequence_length: int,
) -> tuple[int, int] | None:
    """Map prompt character offsets to a BOS-shifted, truncated CLIP token span."""
    if start < 0 or end <= start or end > len(prompt):
        raise ValueError("Phrase character offsets must be a non-empty prompt span.")

    # CLIP reserves token index 0 for BOS, so shift text-token offsets by one.
    start_token = len(tokenizer.tokenize(prompt[:start])) + 1
    end_token = len(tokenizer.tokenize(prompt[:end])) + 1
    end_token = min(end_token, sequence_length - 1)
    if start_token >= end_token:
        return None

    return start_token, end_token


def align_phrase_context(
    prompt_context: Float[torch.Tensor, "batch tokens dim"],
    phrase_context: Float[torch.Tensor, "batch tokens dim"],
    token_span: tuple[int, int],
    phrase_token_count: int,
) -> Float[torch.Tensor, "batch tokens dim"]:
    """Copy only actual phrase-token embeddings into the matching prompt span."""
    start, end = token_span
    span_length = end - start
    if start < 1 or end > prompt_context.shape[1] or span_length < 1:
        raise ValueError("The aligned token span is outside the prompt context.")

    if phrase_token_count < span_length:
        raise ValueError("The phrase encoding has fewer real tokens than its span.")

    if phrase_context.shape[0] != prompt_context.shape[0]:
        raise ValueError("Prompt and phrase context batches must match.")

    aligned = prompt_context.clone()
    # Skip the phrase encoding's BOS token and copy only the aligned phrase tokens.
    aligned[:, start:end] = phrase_context[:, 1 : 1 + span_length]
    return aligned


@lru_cache(maxsize=1)
def _constituency_pipeline() -> _StanzaPipeline:
    """Build a reusable English constituency parser without downloading models."""
    import stanza

    return cast(
        _StanzaPipeline,
        stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,constituency",
            download_method=None,
            use_gpu=False,
            verbose=False,
        ),
    )
