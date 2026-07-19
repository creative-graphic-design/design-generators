import torch
import pytest

from ralf import RalfConfig, RalfForConditionalLayoutGeneration


def tiny_config() -> RalfConfig:
    return RalfConfig(
        max_seq_length=2,
        num_bin=8,
        decoder_d_model=16,
        decoder_layers=1,
        num_attention_heads=4,
        d_model=16,
    )


def test_forward_logits_shape() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    input_ids = torch.tensor([[config.bos_token_id, 0, config.pad_token_id]])
    attention_mask = input_ids.ne(config.pad_token_id)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 3, config.vocab_size)


def test_generate_sequences_respects_generator() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    input_ids = torch.tensor([[config.bos_token_id]])
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    first = model._generate_sequences(input_ids, generator=generator_a, max_length=3)
    second = model._generate_sequences(input_ids, generator=generator_b, max_length=3)

    assert torch.equal(first, second)


def test_forward_loss_tuple_and_image_context_branches() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    input_ids = torch.tensor(
        [
            [config.bos_token_id, 0, config.pad_token_id],
            [config.bos_token_id, 1, config.pad_token_id],
        ]
    )
    labels = input_ids.clone()
    pixel_values = torch.ones(1, 3, 8, 8)

    output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        labels=labels,
        return_dict=False,
    )

    assert len(output) == 2
    assert output[0].ndim == 0
    assert output[1].shape == (2, 3, config.vocab_size)


def test_forward_requires_input_ids_for_non_vendor_model() -> None:
    model = RalfForConditionalLayoutGeneration(tiny_config())

    with pytest.raises(ValueError, match="input_ids is required"):
        model()


def test_vendor_task_aliases() -> None:
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_vendor_task("unconditional")
        == "uncond"
    )
    assert RalfForConditionalLayoutGeneration._canonical_to_vendor_task("label") == "c"
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_vendor_task("label_size")
        == "cwh"
    )
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_vendor_task("completion")
        == "partial"
    )
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_vendor_task("relation")
        == "relation"
    )
