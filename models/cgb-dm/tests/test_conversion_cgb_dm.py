import pytest
import torch

from cgb_dm import CGBDMConfig
from cgb_dm.conversion import (
    build_model_from_config,
    build_pipeline_from_checkpoint,
    convert_state_dict,
)


def tiny_config() -> CGBDMConfig:
    return CGBDMConfig(
        max_seq_length=2,
        image_size=(32, 32),
        dim_model=16,
        n_head=2,
        feature_dim=32,
        num_layers=1,
        num_train_timesteps=10,
        ddim_num_steps=1,
    )


def test_convert_state_dict_strips_wrappers():
    state = {"state_dict.model.module.img_encoder.patch.weight": torch.zeros(1)}
    converted = convert_state_dict(state)
    assert "img_encoder.patch.weight" in converted


def test_build_pipeline_from_checkpoint_and_rejects_unmatched(tmp_path):
    config = tiny_config()
    model = build_model_from_config(config)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)

    pipe = build_pipeline_from_checkpoint(checkpoint, config=config)
    assert pipe.processor.max_seq_length == 2

    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {"other.weight": torch.zeros(1)}}, bad)
    with pytest.raises(ValueError, match="missing=.*unexpected"):
        build_pipeline_from_checkpoint(bad, config=config)

    malformed = tmp_path / "malformed.pt"
    torch.save([1, 2, 3], malformed)
    with pytest.raises(TypeError, match="state_dict-like"):
        build_pipeline_from_checkpoint(malformed, config=config)
