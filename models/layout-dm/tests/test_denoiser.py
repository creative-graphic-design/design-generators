import torch
from torch import nn

from layout_dm.modeling_layout_dm import LayoutDMDenoiser, _get_clones


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

    log_probs = model.predict_start_log_probs(
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        timesteps=torch.zeros(2, dtype=torch.long),
    )
    assert log_probs.shape == (2, 10, 20)
    assert torch.allclose(log_probs[:, :, -1], torch.full((2, 10), -70.0))


def test_get_clones_returns_independent_modules():
    clones = _get_clones(nn.Linear(2, 2), 2)
    assert len(clones) == 2
    assert clones[0] is not clones[1]
