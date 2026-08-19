"""Generation helpers for PosterLLaVA pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, cast

import torch
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

IMAGE_TOKEN: Final[str] = "<image>"
IMAGE_TOKEN_INDEX: Final[int] = -200
DEFAULT_STOP_STRINGS: Final[tuple[str, ...]] = ("###",)


def infer_conversation_mode(
    model_name: str,
    override: str | None = None,
) -> Literal["llava_v0", "llava_v1"]:
    """Infer the LLaVA conversation mode used by a checkpoint.

    Args:
        model_name: Checkpoint id or local model name.
        override: Explicit mode. When provided, it is validated and returned.

    Returns:
        Supported conversation mode.

    Raises:
        ValueError: If the override is unsupported.

    Examples:
        >>> infer_conversation_mode("posterllava/posterllava_v0")
        'llava_v0'
    """
    if override is not None:
        if override not in {"llava_v0", "llava_v1"}:
            raise ValueError(f"Unsupported conversation mode: {override}")

        return cast(Literal["llava_v0", "llava_v1"], override)
    lowered = model_name.lower()
    if "v1" in lowered or "llava-1.5" in lowered:
        return "llava_v1"
    return "llava_v0"


def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: Literal["pt"] = "pt",
) -> Int[torch.Tensor, "tokens"]:
    """Tokenize a prompt while replacing ``<image>`` with LLaVA's image id.

    Args:
        prompt: Prompt text containing zero or more ``<image>`` markers.
        tokenizer: Tokenizer used for surrounding text chunks.
        image_token_index: Sentinel id inserted between text chunks.
        return_tensors: Only ``"pt"`` is supported.

    Returns:
        One-dimensional token id tensor.

    Raises:
        ValueError: If a non-PyTorch return type is requested.
    """
    if return_tensors != "pt":
        raise ValueError("tokenizer_image_token only supports return_tensors='pt'")

    chunks = prompt.split(IMAGE_TOKEN)
    token_ids: list[int] = []
    for idx, chunk in enumerate(chunks):
        encoded = tokenizer(chunk, add_special_tokens=idx == 0)
        chunk_ids = list(cast(Sequence[int], encoded["input_ids"]))
        token_ids.extend(chunk_ids)
        if idx != len(chunks) - 1:
            token_ids.append(image_token_index)
    return torch.tensor(token_ids, dtype=torch.long)


class StopStringCriteria(StoppingCriteria):
    """Stop generation once decoded text contains any configured stop string."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        input_length: int,
        stop_strings: Sequence[str],
    ) -> None:
        """Store tokenizer and decoded suffix matching configuration."""
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.stop_strings = tuple(stop_strings)

    def __call__(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        scores: Float[torch.Tensor, "batch vocab"],
        **kwargs: str | int | float | bool | None,
    ) -> bool:
        """Return whether any batch item has reached a stop string."""
        _ = scores, kwargs
        generated = input_ids[:, self.input_length :]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return any(any(stop in text for stop in self.stop_strings) for text in texts)


def build_stopping_criteria(
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: Int[torch.Tensor, "batch tokens"],
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
) -> StoppingCriteriaList:
    """Build LLaVA-style stop-string criteria.

    Args:
        tokenizer: Tokenizer used for decoding generated suffixes.
        input_ids: Prompt token ids whose length should be ignored.
        stop_strings: Stop strings to detect in generated text.

    Returns:
        Transformers stopping criteria list.
    """
    return StoppingCriteriaList(
        [
            StopStringCriteria(
                tokenizer,
                input_length=input_ids.shape[-1],
                stop_strings=stop_strings,
            )
        ]
    )


__all__ = [
    "DEFAULT_STOP_STRINGS",
    "IMAGE_TOKEN",
    "IMAGE_TOKEN_INDEX",
    "build_stopping_criteria",
    "infer_conversation_mode",
    "tokenizer_image_token",
]
