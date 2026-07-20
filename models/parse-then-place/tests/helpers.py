from __future__ import annotations

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class FakeTokenizer(PreTrainedTokenizerBase):
    model_input_names = ["input_ids", "attention_mask"]

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> BatchEncoding:
        _ = (return_tensors, padding)
        batch = [texts] if isinstance(texts, str) else texts
        input_ids = torch.arange(len(batch) * 2, dtype=torch.long).reshape(
            len(batch), 2
        )
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )

    def batch_decode(  # ty: ignore[invalid-method-override]
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> list[str]:
        _ = (skip_special_tokens, clean_up_tokenization_spaces, kwargs)
        return ["TEXT 0 0 10 20" for _ in range(len(sequences))]
