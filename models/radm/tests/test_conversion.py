import pytest
import torch

from radm import RADMConfig
from radm.conversion import (
    build_pipeline,
    convert_original_state_dict,
    inspect_checkpoint_payload,
)
from radm.modeling_radm import RADMDenoiser


def test_build_pipeline_from_config() -> None:
    pipe = build_pipeline(RADMConfig(num_proposals=2, hidden_dim=8, text_feature_dim=4))
    assert pipe.radm_config.num_proposals == 2


def test_convert_original_state_dict_supported_prefixes() -> None:
    tensor = torch.zeros(1)
    out = convert_original_state_dict({"model.denoiser.class_head.bias": tensor})
    assert out == {"class_head.bias": tensor}


def test_convert_original_state_dict_maps_detectron2_component_names() -> None:
    tensor = torch.zeros(1)
    out = convert_original_state_dict(
        {
            "backbone.fpn_lateral2.weight": tensor,
            "head.head_series.0.self_attn.in_proj_weight": tensor,
        }
    )
    assert out == {
        "backbone.body.fpn.inner_blocks.0.0.weight": tensor,
        "head.blocks.0.self_attn.in_proj_weight": tensor,
    }


def test_convert_original_state_dict_maps_time_mlp_and_strict_loads() -> None:
    config = RADMConfig(
        num_proposals=2,
        num_classes=4,
        hidden_dim=8,
        text_feature_dim=4,
        max_text_num=4,
        backbone_depth=18,
    )
    source_model = RADMDenoiser(config=config)
    source_state = {
        (
            key if key.startswith("head.time_mlp.") else f"denoiser.{key}"
        ): value.detach().clone()
        for key, value in source_model.state_dict().items()
    }

    converted = convert_original_state_dict(source_state)
    restored_model = RADMDenoiser(config=config)
    restored_model.load_state_dict(converted, strict=True)

    assert converted["head.time_mlp.1.weight"].shape == (32, 8)
    assert converted["head.time_mlp.1.bias"].shape == (32,)
    assert converted["head.time_mlp.3.weight"].shape == (32, 32)
    assert converted["head.time_mlp.3.bias"].shape == (32,)


def test_convert_original_state_dict_rejects_absent_keys() -> None:
    with pytest.raises(RuntimeError, match="No RADM denoiser keys"):
        convert_original_state_dict({"backbone.weight": torch.zeros(1)})


def test_inspect_checkpoint_payload() -> None:
    summary = inspect_checkpoint_payload(
        {"model": {"a": torch.zeros(1)}, "optimizer": {}}
    )
    assert summary["has_model"] is True
    assert summary["state_dict_keys"] == 1
