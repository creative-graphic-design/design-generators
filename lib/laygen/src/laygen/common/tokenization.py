"""Shared helpers for lightweight whitespace layout tokenizers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast


def build_token_maps(
    *,
    vocab_file: str | PathLike[str] | None,
    tokens: Sequence[str] | None,
    base_tokens: Sequence[str],
    numeric_id_vocab: bool = False,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build token/id maps from a JSON vocabulary file or synthetic token list."""
    if vocab_file is not None:
        with Path(vocab_file).open() as f:
            raw_vocab = cast(dict[str, object], json.load(f))
        if numeric_id_vocab and all(str(key).isdigit() for key in raw_vocab):
            id2token = {int(key): str(value) for key, value in raw_vocab.items()}
            return {value: key for key, value in id2token.items()}, id2token
        token2id = {str(key): int(str(value)) for key, value in raw_vocab.items()}
        return token2id, {value: key for key, value in token2id.items()}

    token2id = {token: idx for idx, token in enumerate(base_tokens)}
    for token in tokens or ():
        if token not in token2id:
            token2id[token] = len(token2id)
    return token2id, {idx: token for token, idx in token2id.items()}


def split_whitespace_tokens(text: str) -> list[str]:
    """Split a layout token string on whitespace."""
    return text.strip().split()


def convert_token_to_id(token2id: dict[str, int], token: str, unk_token_id: int) -> int:
    """Convert a token to an id using a tokenizer-local unknown-token id."""
    return token2id.get(token, unk_token_id)


def convert_id_to_token(id2token: dict[int, str], index: int, unk_token: str) -> str:
    """Convert an id to a token using a tokenizer-local unknown token string."""
    return id2token.get(int(index), unk_token)


def join_tokens(tokens: Sequence[str]) -> str:
    """Join already-tokenized layout tokens with spaces."""
    return " ".join(tokens)


def save_json_vocabulary(
    *,
    save_directory: str | PathLike[str],
    filename: str,
    data: dict[str, int] | dict[str, str],
    filename_prefix: str | None = None,
) -> tuple[str]:
    """Save tokenizer vocabulary JSON and return the generated path."""
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = filename if filename_prefix is None else f"{filename_prefix}-{filename}"
    out_path = out_dir / name
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return (str(out_path),)


class WhitespaceTokenizerMixin:
    """Mixin for tokenizers backed by tokenizer-local id dictionaries."""

    _token2id: dict[str, int]
    _id2token: dict[int, str]
    unk_token_id: int
    unk_token: str

    @property
    def vocab_size(self) -> int:
        """Return the number of vocabulary entries."""
        return len(self._token2id)

    def get_vocab(self) -> dict[str, int]:
        """Return token-to-id mapping."""
        return dict(self._token2id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        _ = kwargs
        return split_whitespace_tokens(text)

    def _convert_token_to_id(self, token: str) -> int:
        return convert_token_to_id(self._token2id, token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return convert_id_to_token(self._id2token, index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join layout tokens with spaces."""
        return join_tokens(tokens)
