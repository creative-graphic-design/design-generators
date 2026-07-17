import torch

from layout_dm.conversion import remap_denoiser_key, split_original_state_dict


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
