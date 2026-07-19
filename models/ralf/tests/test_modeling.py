import torch

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
