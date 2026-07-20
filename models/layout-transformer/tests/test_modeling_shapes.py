import torch

from layout_transformer import (
    LayoutTransformerConfig,
    LayoutTransformerForLayoutGeneration,
)


def build_model(refine=False):
    config = LayoutTransformerConfig(
        vocab_size=16,
        obj_classes_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_sequence_length=6,
        refine=refine,
    )
    return LayoutTransformerForLayoutGeneration(config)


def build_inputs():
    input_token = torch.tensor([[1, 4, 5, 6, 2, 0]])
    input_obj_id = torch.tensor([[0, 1, 0, 2, 0, 0]])
    segment_label = torch.tensor([[0, 1, 1, 1, 1, 0]])
    token_type = torch.tensor([[0, 1, 2, 3, 0, 0]])
    src_mask = input_token.ne(0).unsqueeze(1)
    return input_token, input_obj_id, segment_label, token_type, src_mask


def test_model_forward_shapes():
    model = build_model()
    input_token, input_obj_id, segment_label, token_type, src_mask = build_inputs()

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


def test_model_forward_refine_tuple_and_hidden_states():
    model = build_model(refine=True)
    input_token, input_obj_id, segment_label, token_type, _ = build_inputs()
    bbox = torch.zeros(1, 6, 4)

    output = model(
        input_token=input_token,
        input_obj_id=input_obj_id,
        segment_label=segment_label,
        token_type=token_type,
        src_mask=input_token.ne(0),
        bbox=bbox,
        output_hidden_states=True,
    )
    output_tuple = model(
        input_token=input_token,
        input_obj_id=input_obj_id,
        segment_label=segment_label,
        token_type=token_type,
        bbox=bbox,
        return_dict=False,
    )

    assert output.refine_box.shape == (1, 6, 4)
    assert output.hidden_states.shape == (1, 6, 32)
    assert isinstance(output_tuple, tuple)


def test_model_generate_boxes_uses_private_pipeline_path():
    model = build_model()
    input_token, input_obj_id, segment_label, token_type, src_mask = build_inputs()

    output = model._generate_boxes(
        input_token=input_token,
        input_obj_id=input_obj_id,
        segment_label=segment_label,
        token_type=token_type,
        src_mask=src_mask,
    )

    assert output.coarse_box.shape == (1, 6, 4)


def test_model_forward_rejects_mismatched_input_shapes():
    model = build_model()
    input_token, input_obj_id, segment_label, token_type, src_mask = build_inputs()

    try:
        model(
            input_token=input_token[:, :5],
            input_obj_id=input_obj_id,
            segment_label=segment_label,
            token_type=token_type,
            src_mask=src_mask,
        )
    except ValueError as exc:
        assert "must have the same shape" in str(exc)
    else:
        raise AssertionError("expected shape mismatch error")
