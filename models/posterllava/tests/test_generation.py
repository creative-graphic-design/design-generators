from __future__ import annotations

import torch
from typing import Literal, cast
from transformers import PreTrainedTokenizerBase

from posterllava.generation_posterllava import (
    IMAGE_TOKEN_INDEX,
    StopStringCriteria,
    build_stopping_criteria,
    infer_conversation_mode,
    tokenizer_image_token,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [ord(char) % 31 + 1 for char in text]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}

    def batch_decode(self, ids, skip_special_tokens: bool = True):
        _ = ids, skip_special_tokens
        return ["hello###"]


def test_infer_conversation_mode_defaults_to_llava_v0() -> None:
    assert infer_conversation_mode("posterllava/posterllava_v0") == "llava_v0"
    assert infer_conversation_mode("llava-1.5-7b") == "llava_v1"
    assert infer_conversation_mode("anything", override="llava_v1") == "llava_v1"

    try:
        infer_conversation_mode("anything", override="other")
    except ValueError as exc:
        assert "Unsupported conversation mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_tokenizer_image_token_inserts_sentinel() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, TinyTokenizer())
    ids = tokenizer_image_token("a<image>b", tokenizer)

    assert ids.dtype == torch.long
    assert IMAGE_TOKEN_INDEX in ids.tolist()

    try:
        tokenizer_image_token("x", tokenizer, return_tensors=cast(Literal["pt"], "np"))
    except ValueError as exc:
        assert "return_tensors='pt'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_stop_string_criteria_detects_decoded_suffix() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, TinyTokenizer())
    criteria = StopStringCriteria(  # type: ignore[arg-type]
        tokenizer,
        input_length=1,
        stop_strings=("###",),
    )

    assert criteria(torch.tensor([[1, 2, 3]]), torch.zeros(1, 4))
    assert (
        len(
            build_stopping_criteria(
                tokenizer,
                input_ids=torch.ones(1, 1, dtype=torch.long),
            )
        )
        == 1
    )  # type: ignore[arg-type]
