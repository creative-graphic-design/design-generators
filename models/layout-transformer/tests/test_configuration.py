from typing import cast

from layout_transformer import LayoutTransformerConfig


def test_config_round_trip(tmp_path):
    config = LayoutTransformerConfig(
        dataset_name="vg_msdn",
        vocab_size=20,
        obj_classes_size=10,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        id2label={0: "__image__", 1: "person"},
        relation_id2label={1: "left of"},
    )
    config.save_pretrained(tmp_path)

    loaded = LayoutTransformerConfig.from_pretrained(tmp_path)

    assert loaded.dataset_name == "vg_msdn"
    assert loaded.vocab_size == 20
    id2label = cast(dict[int, str], loaded.id2label)
    assert id2label[1] == "person"
    assert loaded.relation_id2label[1] == "left of"
