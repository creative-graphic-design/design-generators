from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPTokenizer,
)


@pytest.mark.vendor_parity
def test_rico_gen_t_logits_match_vendor() -> None:
    root = Path.cwd()
    checkpoint = (
        root
        / ".cache/layoutformerpp/original/ckpts/rico_gen_t/final_checkpoint.pth.tar"
    )
    vocab = root / ".cache/layoutformerpp/original/ckpts/rico_gen_t/vocab.json"
    converted = root / ".cache/layoutformerpp/converted/rico_gen_t"
    vendor_src = root / "vendor/ms-layout-generation/LayoutFormer++/src"
    if (
        not checkpoint.exists()
        or not vocab.exists()
        or not converted.exists()
        or not vendor_src.exists()
    ):
        pytest.skip(
            "LayoutFormer++ original checkpoint, vocab, converted model, or vendor source is absent"
        )

    sys.path.insert(0, str(vendor_src))
    from model.layout_transformer.model import LayoutTransformer  # type: ignore
    from model.layout_transformer.tokenizer import LayoutTransformerTokenizer  # type: ignore

    vendor_tokenizer = LayoutTransformerTokenizer([])
    vendor_tokenizer.from_vocab(str(vocab))
    tokenizer = LayoutFormerPPTokenizer.from_pretrained(converted)
    assert vendor_tokenizer._token2id == tokenizer.get_vocab()

    state = torch.load(checkpoint, map_location="cpu")
    vendor_model = LayoutTransformer(
        vocab_size=len(vendor_tokenizer),
        max_len=150,
        bos_token_id=vendor_tokenizer.bos_token_id,
        pad_token_id=vendor_tokenizer.pad_token_id,
        eos_token_id=vendor_tokenizer.eos_token_id,
        d_model=512,
        num_layers=8,
        nhead=8,
        dropout=0.1,
        d_feedforward=2048,
        share_embedding=True,
    )
    vendor_model.load_state_dict(
        {k.removeprefix("module."): v for k, v in state.items()}, strict=False
    )
    vendor_model.eval()
    model = LayoutFormerPPForConditionalGeneration.from_pretrained(converted)
    model.eval()

    encoded = tokenizer.encode_text(["label_1 label_2"], add_eos=True)
    labels = tokenizer.encode_text(
        ["label_1 0 0 10 10 | label_2 1 1 11 11 |"], add_eos=True
    )["input_ids"]
    with torch.no_grad():
        vendor_logits = vendor_model.compute_loss(
            encoded["input_ids"], ~encoded["attention_mask"].bool(), labels
        )["logits"]
        new_logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
        ).logits
    torch.testing.assert_close(new_logits, vendor_logits, atol=0.0, rtol=0.0)
