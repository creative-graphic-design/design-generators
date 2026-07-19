import torch

from layout_transformer import (
    LayoutTransformerConfig,
    LayoutTransformerForLayoutGeneration,
)


def test_model_forward_shapes():
    config = LayoutTransformerConfig(
        vocab_size=16,
        obj_classes_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_sequence_length=6,
    )
    model = LayoutTransformerForLayoutGeneration(config)
    input_token = torch.tensor([[1, 4, 5, 6, 2, 0]])
    input_obj_id = torch.tensor([[0, 1, 0, 2, 0, 0]])
    segment_label = torch.tensor([[0, 1, 1, 1, 1, 0]])
    token_type = torch.tensor([[0, 1, 2, 3, 0, 0]])
    src_mask = input_token.ne(0).unsqueeze(1)

    output = model(
        input_token=input_token,
        input_obj_id=input_obj_id,
        segment_label=segment_label,
        token_type=token_type,
        src_mask=src_mask,
    )

    assert output.vocab_logits.shape == (1, 6, 16)
    assert output.obj_id_logits.shape == (1, 6, 8)
    assert output.token_type_logits.shape == (1, 6, 4)
    assert output.coarse_box.shape == (1, 6, 4)
