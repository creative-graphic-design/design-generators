import torch

from layoutdiffusion import LayoutDiffusionTransformer


def test_transformer_forward_shape() -> None:
    model = LayoutDiffusionTransformer(
        vocab_size=32,
        num_channels=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    out = model(
        input_ids=torch.zeros(2, 7, dtype=torch.long),
        timesteps=torch.zeros(2, dtype=torch.long),
    )
    assert out.logits.shape == (2, 31, 7)


def test_transformer_return_tuple_and_condition_paths() -> None:
    model = LayoutDiffusionTransformer(
        vocab_size=140,
        num_channels=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    input_ids = torch.full((1, 7), 139, dtype=torch.long)
    condition_ids = torch.zeros(1, 7, dtype=torch.long)
    out = model(
        input_ids=input_ids,
        timesteps=torch.zeros(1, dtype=torch.long),
        condition_ids=condition_ids,
        condition_type="label",
        return_dict=False,
    )
    assert out[0].shape == (1, 139, 7)
    completion = model(
        input_ids=input_ids,
        timesteps=torch.zeros(1, dtype=torch.long),
        condition_ids=condition_ids,
        condition_type="completion",
    )
    assert completion.logits.shape == (1, 139, 7)
