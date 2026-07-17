import torch

from lace import LaceTransformerModel
from lace.conversion import convert_state_dict
from lace.model_card import write_lace_model_card


def test_convert_state_dict_strips_module_prefix() -> None:
    state = {"module.layer_in.weight": torch.ones(2, 2), "pos_embed": torch.ones(2, 2)}
    converted = convert_state_dict(state)
    assert "layer_in.weight" in converted
    assert "pos_embed" not in converted


def test_vendor_key_families_match_model() -> None:
    model = LaceTransformerModel(
        seq_dim=10,
        max_seq_length=25,
        num_layers=1,
        dim_transformer=32,
        nhead=4,
        dim_feedforward=64,
    )
    keys = set(model.state_dict())
    assert "layer_in.weight" in keys
    assert "layers.0.self_attn.in_proj_weight" in keys
    assert "layers.0.linear1.weight" in keys
    assert "layer_out.weight" in keys


def test_write_lace_model_card_smoke(tmp_path) -> None:
    path = write_lace_model_card(tmp_path, "publaynet")
    text = path.read_text(encoding="utf-8")
    assert path.name == "README.md"
    assert "# Model Card for creative-graphic-design/lace-publaynet" in text
    assert "creative-graphic-design/lace-publaynet" in text
    assert "creative-graphic-design/publaynet" in text
    assert "| publaynet | n/a | n/a | 0 | 0 |" in text
    assert "## Uses" in text
    assert "## Evaluation" in text
    assert "[More Information Needed]" not in text
