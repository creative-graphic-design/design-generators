from smarttext import SmartTextBackbone, SmartTextConfig, SmartTextRegionMode


def test_config_defaults_match_plan():
    config = SmartTextConfig()

    assert config.id2label == {0: "text"}
    assert config.label2id == {"text": 0}
    assert config.scorer_backbone == SmartTextBackbone.shufflenetv2.value
    assert config.model_type_name == SmartTextRegionMode.RoE.value
    assert config.ratio_list == (1.0, 0.8)
    assert config.num_labels == 1
    assert config.uses_expanded_region
    assert config.scorer_input_channels == 3


def test_config_round_trip(tmp_path):
    config = SmartTextConfig(id2label={"0": "text"}, candi_res=2)
    config.save_pretrained(tmp_path)

    loaded = SmartTextConfig.from_pretrained(tmp_path)

    assert loaded.id2label == {0: "text"}
    assert loaded.candi_res == 2
