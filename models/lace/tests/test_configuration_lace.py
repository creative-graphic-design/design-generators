from lace import DATASET_SPECS, default_model_config, get_dataset_spec


def test_dataset_specs_have_expected_dimensions() -> None:
    assert DATASET_SPECS["publaynet"].seq_dim == 10
    assert DATASET_SPECS["rico13"].seq_dim == 18
    assert DATASET_SPECS["rico25"].seq_dim == 30
    assert get_dataset_spec("rico13").pad_label_id == 13
    assert get_dataset_spec("rico25").pad_label_id == 25


def test_default_model_config_matches_dataset() -> None:
    publaynet = default_model_config("publaynet")
    rico13 = default_model_config("rico13")
    assert publaynet["seq_dim"] == 10
    assert publaynet["dim_transformer"] == 1024
    assert rico13["seq_dim"] == 18
    assert rico13["dim_transformer"] == 512
