from layoutformerpp.serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenR,
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


def test_sequence_parse_without_sep_and_error_paths() -> None:
    seq = T5LayoutSequence({1: "text"}, add_sep_token=False)

    parsed = seq.parse_seq("text element 1 2 3 4")
    assert parsed is not None
    assert parsed.labels == [0]
    assert parsed.bbox == [[1, 2, 3, 4]]
    assert seq.parse_seq("text 1 2") is None
    assert seq.parse_seq("") is None


def test_gen_t_unknown_tokens_and_relation_serializer() -> None:
    gen_t = T5LayoutSequenceForGenT({1: "label_1"}, add_sep_token=True)
    assert (
        gen_t.build_input_seq("gen_t", [1], [[1, 2, 3, 4]], add_unk_for_label=True)
        == "label_1 <unk> <unk> <unk> <unk>"
    )
    assert (
        gen_t.build_input_seq(
            "gen_ts", [1], [[1, 2, 3, 4]], add_unk_for_label_size=True
        )
        == "label_1 <unk> <unk> 3 4"
    )

    gen_r = T5LayoutSequenceForGenR({1: "label_1", 2: "label_2"}, add_sep_token=True)
    verbose = gen_r.build_input_seq([1, 2], [(2, 1, 1, 0, 3)], add_unk_token=True)
    assert "<sep_ele_rela_ele>" in verbose
    assert "relation_3" in verbose
    compact = gen_r.build_input_seq([1], [(0, 0, 0, 0, 1)], compact=True)
    assert "<sep_ele_rela_ele>" not in compact
    assert "label_0 relation_1 label_0" in compact
