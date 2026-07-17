import pytest
import torch

from lace import LaceTransformerModel
from lace.conversion import (
    build_pipeline_from_vendor_checkpoint,
    convert_state_dict,
    load_vendor_state_dict,
)
from lace.model_card import lace_model_card, write_lace_model_card


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


def test_load_vendor_state_dict_accepts_wrapped_state_and_rejects_other(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, checkpoint)
    assert "module.weight" in load_vendor_state_dict(checkpoint)

    bad = tmp_path / "bad.pt"
    torch.save([1, 2, 3], bad)
    with pytest.raises(TypeError, match="state_dict-like"):
        load_vendor_state_dict(bad)


def test_build_pipeline_from_vendor_checkpoint_reports_key_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"unexpected": torch.ones(1)}}, checkpoint)
    with pytest.raises(ValueError, match="State dict mismatch"):
        build_pipeline_from_vendor_checkpoint("publaynet", checkpoint, ddim_num_steps=1)


def test_rico13_model_card_omits_missing_parity_table() -> None:
    text = str(lace_model_card("rico13"))
    assert "creative-graphic-design/lace-rico13" in text
    assert "Rico13 parity metrics are therefore not reported here" in text
