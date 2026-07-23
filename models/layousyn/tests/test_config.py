import json

from layousyn.configuration_layousyn import LayouSynConfig, resolve_model_shape


def test_vendor_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": "DiT-D2-H64-N4",
                "in_channel": 4,
                "concept_in_channel": 8,
                "y_in_channel": 8,
                "max_in_len": 5,
                "max_y_len": 6,
                "scale": 2.0,
                "noise_schedule": "squaredcos_cap_v2",
                "layout_type": "cxcywh",
                "diffusion_steps": 10,
                "t5_size": "base",
            }
        )
    )
    cfg = LayouSynConfig.from_reference_json(path)
    assert cfg.hidden_size == 64
    assert cfg.depth == 2
    assert cfg.num_heads == 4
    assert cfg.to_reference_dict()["layout_type"] == "cxcywh"


def test_named_model_shape_resolution() -> None:
    assert resolve_model_shape("DiT-S") == {
        "hidden_size": 256,
        "depth": 8,
        "num_heads": 8,
    }
