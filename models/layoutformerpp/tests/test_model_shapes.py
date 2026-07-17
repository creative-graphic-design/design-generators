import torch

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPProcessor,
)


def test_tiny_model_forward_and_generation() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="publaynet", task="ugen")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        dataset="publaynet",
        task="ugen",
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    encoded = processor(condition_type="unconditional")
    labels = torch.tensor([[5, 6, 7, 8, 9, 3, 1]])
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=labels,
    )
    assert outputs.logits.shape[:2] == labels.shape
    generated = model.generate_sequences(
        encoded["input_ids"], encoded["attention_mask"], max_length=3
    )
    assert generated.shape == (1, 3)
