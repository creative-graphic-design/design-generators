from pathlib import Path

from layoutformerpp import LayoutFormerPPTokenizer


def test_tokenizer_special_ids_and_roundtrip(tmp_path: Path) -> None:
    tokenizer = LayoutFormerPPTokenizer(tokens=["label_1", "0", "|"])
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2
    encoded = tokenizer.encode_text("label_1 0 0 1 1 |")
    text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    assert "label_1" in text
    tokenizer.save_pretrained(tmp_path)
    loaded = LayoutFormerPPTokenizer.from_pretrained(tmp_path)
    assert loaded.get_vocab() == tokenizer.get_vocab()
