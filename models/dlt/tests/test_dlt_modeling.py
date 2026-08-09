import torch

from dlt import DLT


def test_dlt_forward_shapes_and_state_keys() -> None:
    model = DLT(
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
    )
    batch = {
        "box": torch.zeros(2, 3, 4),
        "box_cond": torch.zeros(2, 3, 4),
        "cat": torch.ones(2, 3, dtype=torch.long),
        "mask_box": torch.ones(2, 3, 4, dtype=torch.long),
        "mask_cat": torch.ones(2, 3, dtype=torch.long),
    }
    noisy = {"box": torch.randn(2, 3, 4), "cat": torch.full((2, 3), 6)}
    box, logits = model(batch, noisy, torch.tensor([0, 1]))
    assert box.shape == (2, 3, 4)
    assert logits.shape == (2, 3, 7)
    state_keys = set(model.state_dict())
    assert "cat_emb" in state_keys
    assert "seqTransEncoder.layers.0.self_attn.in_proj_weight" in state_keys
