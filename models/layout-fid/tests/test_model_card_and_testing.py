import numpy as np
import pytest
import torch

from layout_fid import LayoutFIDConfig
from layout_fid.model_card import model_card_metadata
from layout_fid.testing import assert_feature_close, assert_statistics_shape


def test_model_card_metadata_and_testing_helpers():
    cfg = LayoutFIDConfig(
        dataset_name="rico25",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=25,
        num_label_embeddings=26,
        max_length=20,
    )
    metadata = model_card_metadata(cfg, hub_id="creative-graphic-design/example")
    assert metadata["library_name"] == "transformers"
    assert metadata["datasets"] == ["creative-graphic-design/Rico"]
    assert_feature_close(torch.zeros(1, 2), torch.zeros(1, 2))
    assert_statistics_shape(np.zeros(2), np.eye(2))
    with pytest.raises(AssertionError):
        assert_statistics_shape(np.zeros((1, 2)), np.eye(2))
    with pytest.raises(AssertionError):
        assert_statistics_shape(np.zeros(2), np.eye(3))
