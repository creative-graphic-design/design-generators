from layout_detr import LayoutDetrConfig
from typing import cast


def test_config_defaults_match_plan():
    config = LayoutDetrConfig()

    assert config.id2label == {
        0: "header",
        1: "pre-header",
        2: "post-header",
        3: "body text",
        4: "disclaimer / footnote",
        5: "button",
        6: "callout",
        7: "logo",
    }
    assert cast(dict[str, int], config.label2id)["button"] == 5
    assert config.max_seq_length == 9
    assert config.max_elements == 9
    assert config.z_dim == 4
    assert config.max_text_length == 256
    assert config.num_bbox_labels == 8


def test_config_round_trip_preserves_metadata(tmp_path):
    config = LayoutDetrConfig(
        original_training_options={"background_size": 256},
        conversion_report={"loaded_key_count": 3},
    )
    config.save_pretrained(tmp_path)

    loaded = LayoutDetrConfig.from_pretrained(tmp_path)

    assert loaded.original_training_options == {"background_size": 256}
    assert loaded.conversion_report == {"loaded_key_count": 3}
