from radm import RADMConfig
from radm.configuration_radm import RADM_ORIGINAL_CGL_LABELS, default_id2label


def test_config_defaults_and_label_maps() -> None:
    config = RADMConfig(num_proposals=3, hidden_dim=8, text_feature_dim=4)
    assert config.dataset_name == "cgl"
    assert config.num_labels == 5
    assert config.id2label[0] == "logo"
    assert config.original_id2label[1] == RADM_ORIGINAL_CGL_LABELS[1]
    assert config.label2id["logo"] == 0


def test_default_original_labels() -> None:
    labels = default_id2label("cgl_v2", label_mode="original")
    assert labels[4] == "强调突出子部分文字"
