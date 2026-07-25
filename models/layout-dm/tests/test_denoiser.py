import torch

from layout_dm.modeling_layout_dm import LayoutDMDenoiser


def test_denoiser_forward_tiny():
    model = LayoutDMDenoiser(
        vocab_size=20,
        max_token_length=10,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
    )
    out = model(
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        timesteps=torch.zeros(2, dtype=torch.long),
    )
    assert out.logits.shape == (2, 10, 20)
