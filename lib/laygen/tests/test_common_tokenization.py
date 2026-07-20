import json

from laygen.common.tokenization import (
    build_token_maps,
    convert_id_to_token,
    convert_token_to_id,
    join_tokens,
    save_json_vocabulary,
    split_whitespace_tokens,
)


def test_build_token_maps_from_token_to_id_vocab(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"<pad>": 0, "label_1": 1}), encoding="utf-8")

    token2id, id2token = build_token_maps(
        vocab_file=vocab,
        tokens=None,
        base_tokens=("<pad>",),
    )

    assert token2id == {"<pad>": 0, "label_1": 1}
    assert id2token == {0: "<pad>", 1: "label_1"}


def test_build_token_maps_from_numeric_id_vocab(tmp_path):
    vocab = tmp_path / "object_pred_id2name.json"
    vocab.write_text(json.dumps({"0": "[PAD]", "4": "person"}), encoding="utf-8")

    token2id, id2token = build_token_maps(
        vocab_file=vocab,
        tokens=None,
        base_tokens=("[PAD]",),
        numeric_id_vocab=True,
    )

    assert token2id == {"[PAD]": 0, "person": 4}
    assert id2token == {0: "[PAD]", 4: "person"}


def test_build_token_maps_from_base_and_extra_tokens():
    token2id, id2token = build_token_maps(
        vocab_file=None,
        tokens=["person", "<pad>", "table"],
        base_tokens=("<pad>", "<unk>"),
    )

    assert token2id == {"<pad>": 0, "<unk>": 1, "person": 2, "table": 3}
    assert id2token == {0: "<pad>", 1: "<unk>", 2: "person", 3: "table"}


def test_whitespace_conversion_join_and_save(tmp_path):
    assert split_whitespace_tokens("  a   b  ") == ["a", "b"]
    assert convert_token_to_id({"a": 7}, "missing", 99) == 99
    assert convert_id_to_token({7: "a"}, 8, "<unk>") == "<unk>"
    assert join_tokens(["a", "b"]) == "a b"

    (path,) = save_json_vocabulary(
        save_directory=tmp_path,
        filename="vocab.json",
        data={"a": 7},
        filename_prefix="demo",
    )

    assert path.endswith("demo-vocab.json")
    assert json.loads((tmp_path / "demo-vocab.json").read_text()) == {"a": 7}
