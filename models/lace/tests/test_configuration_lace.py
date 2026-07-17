import pytest

from lace import (
    DATASET_SPECS,
    LaceDatasetName,
    default_model_config,
    get_dataset_spec,
    normalize_dataset,
)


def test_dataset_specs_have_expected_dimensions() -> None:
    assert DATASET_SPECS[LaceDatasetName.publaynet].seq_dim == 10
    assert DATASET_SPECS[LaceDatasetName.rico13].seq_dim == 18
    assert DATASET_SPECS[LaceDatasetName.rico25].seq_dim == 30
    assert get_dataset_spec("rico13").pad_label_id == 13
    assert get_dataset_spec("rico25").pad_label_id == 25


def test_default_model_config_matches_dataset() -> None:
    publaynet = default_model_config("publaynet")
    rico13 = default_model_config("rico13")
    assert publaynet["seq_dim"] == 10
    assert publaynet["dim_transformer"] == 1024
    assert rico13["seq_dim"] == 18
    assert rico13["dim_transformer"] == 512


def test_dataset_normalization_accepts_aliases_and_rejects_unknown() -> None:
    assert normalize_dataset("publaynet-max25") is LaceDatasetName.publaynet
    assert normalize_dataset(LaceDatasetName.rico25) is LaceDatasetName.rico25
    with pytest.raises(ValueError, match="Unsupported LACE dataset"):
        get_dataset_spec("unknown")
