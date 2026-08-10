"""Training vocabulary helpers for LayoutDiffusion."""

from __future__ import annotations

import json
from pathlib import Path

from ..configuration_layoutdiffusion import LayoutDiffusionConfig
from ..tokenization_layoutdiffusion import LayoutDiffusionTokenizer

_SPECIAL_TOKENS = {"START", "END", "UNK", "PAD", "|", "MASK"}


def build_training_tokenizer(
    config: LayoutDiffusionConfig, *, vocab_file: str | None = None
) -> LayoutDiffusionTokenizer:
    """Build a tokenizer while keeping training config vocabulary fields aligned."""
    if vocab_file is None:
        return LayoutDiffusionTokenizer(config)
    vocab_path = Path(vocab_file)
    if not vocab_path.is_file():
        raise FileNotFoundError(vocab_path)
    raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = {str(token): int(index) for token, index in raw_vocab.items()}
    if "MASK" not in vocab:
        vocab["MASK"] = max(vocab.values()) + 1
    id2label = _id2label_from_vocab(vocab)
    config.vocab = vocab
    config.id2label = id2label
    config.vocab_size = max(vocab.values()) + 1
    config.register_to_config(
        vocab=vocab,
        id2label={str(index): label for index, label in id2label.items()},
        vocab_size=config.vocab_size,
    )
    return LayoutDiffusionTokenizer(config)


def _id2label_from_vocab(vocab: dict[str, int]) -> dict[int, str]:
    labels = [
        token
        for token, _ in sorted(vocab.items(), key=lambda item: item[1])
        if token not in _SPECIAL_TOKENS and not token.isdigit()
    ]
    return dict(enumerate(labels))
