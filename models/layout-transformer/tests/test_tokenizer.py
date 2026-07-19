import json

from layout_transformer import LayoutTransformerRelationTokenizer


def test_tokenizer_loads_id_to_token_vocab(tmp_path):
    vocab = tmp_path / "object_pred_id2name.json"
    vocab.write_text(
        json.dumps(
            {"0": "[PAD]", "1": "[CLS]", "2": "[SEP]", "3": "[MASK]", "4": "person"}
        )
    )

    tokenizer = LayoutTransformerRelationTokenizer(vocab_file=str(vocab))

    assert tokenizer.convert_ids_to_tokens(4) == "person"
    assert tokenizer.decode_scene_graph_tokens([1, 4, 2]) == [
        "[CLS]",
        "person",
        "[SEP]",
    ]


def test_tokenizer_save_and_load_metadata(tmp_path):
    tokenizer = LayoutTransformerRelationTokenizer(
        tokens=["person", "left of"],
        object_token_ids=[4],
        relation_token_ids=[5],
    )
    tokenizer.save_pretrained(tmp_path)

    loaded = LayoutTransformerRelationTokenizer.from_pretrained(
        tmp_path, local_files_only=True
    )

    assert loaded.object_token_ids == [4]
    assert loaded.relation_token_ids == [5]
    assert loaded.get_vocab()["person"] == 4


def test_tokenizer_rejects_unknown_scene_graph_token():
    tokenizer = LayoutTransformerRelationTokenizer(tokens=["person"])

    try:
        tokenizer.encode_scene_graph_tokens(["missing"])
    except ValueError as exc:
        assert "Unknown scene-graph token" in str(exc)
    else:
        raise AssertionError("expected unknown token error")
