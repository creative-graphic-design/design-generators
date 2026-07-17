import torch

from layout_dm.conversion import (
    remap_denoiser_key,
    split_original_state_dict,
    write_layoutdm_model_card,
)


def test_remap_denoiser_key():
    assert (
        remap_denoiser_key("model.module.transformer.cat_emb.weight")
        == "transformer.cat_emb.weight"
    )


def test_split_original_state_dict_ignores_scheduler_buffers():
    state = {
        "model.module.transformer.cat_emb.weight": torch.zeros(1),
        "model.module.c_log_at": torch.zeros(1),
        "model.module.Lt_history": torch.zeros(1),
    }
    assert split_original_state_dict(state) == {
        "transformer.cat_emb.weight": state["model.module.transformer.cat_emb.weight"]
    }


def test_write_layoutdm_model_card(tmp_path):
    path = write_layoutdm_model_card(tmp_path, "rico25")
    text = path.read_text(encoding="utf-8")

    assert path.name == "README.md"
    assert "license: apache-2.0" in text
    assert "library_name: diffusers" in text
    assert "pipeline_tag: text-to-image" in text
    assert "creative-graphic-design/rico25" in text
    assert "LayoutDMPipeline.from_pretrained" in text
    assert "## Uses" in text
    assert "## Evaluation" in text
    assert "Vendor parity against the original implementation" in text
    assert "Tokenizer exact" in text
    assert "[More Information Needed]" not in text
