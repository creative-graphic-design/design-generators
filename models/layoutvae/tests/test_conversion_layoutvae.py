import pytest

from layoutvae import LayoutVAEConfig, LayoutVAEModel
from layoutvae.conversion import convert_state_dicts
from layoutvae.model_card import write_layoutvae_model_card


def test_convert_state_dicts_writes_hf_artifacts(tmp_path):
    model = LayoutVAEModel(LayoutVAEConfig())
    output = convert_state_dicts(
        count_state_dict=model.countvae.state_dict(),
        bbox_state_dict=model.bboxvae.state_dict(),
        output_dir=tmp_path,
    )
    assert (output / "config.json").is_file()
    assert (output / "model.safetensors").is_file()
    assert (output / "preprocessor_config.json").is_file()


def test_convert_state_dicts_refuses_non_hf_names(tmp_path):
    (tmp_path / "countvae.h5").write_bytes(b"not a converted artifact")
    model = LayoutVAEModel(LayoutVAEConfig())
    with pytest.raises(ValueError, match="non-HF"):
        convert_state_dicts(
            count_state_dict=model.countvae.state_dict(),
            bbox_state_dict=model.bboxvae.state_dict(),
            output_dir=tmp_path,
        )


def test_model_card_helper_writes_readme(tmp_path):
    path = write_layoutvae_model_card(tmp_path)
    assert path.name == "README.md"
    assert "LayoutVAE PubLayNet" in path.read_text(encoding="utf-8")
