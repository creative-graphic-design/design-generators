from layoutformerpp.serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenT,
    build_default_tokens,
)


def test_full_sequence_parse_with_sep() -> None:
    seq = T5LayoutSequence({1: "label_1", 2: "label_2"}, add_sep_token=True)
    text = seq.build_seq([1, 2], [[0, 1, 2, 3], [4, 5, 6, 7]])
    parsed = seq.parse_seq(text)
    assert parsed is not None
    assert parsed.labels == [1, 2]
    assert parsed.bbox == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_gen_t_and_vocab_tokens() -> None:
    seq = T5LayoutSequenceForGenT({1: "label_1"}, add_sep_token=True)
    assert seq.build_input_seq("gen_ts", [1], [[1, 2, 3, 4]]) == "label_1 3 4"
    tokens = build_default_tokens(("Text",), task="gen_r", grid=2)
    assert "label_1" in tokens
    assert "<sep_labels_relations>" in tokens
