import torch
import pytest

from layoutformerpp import (
    ConditionType,
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


def test_model_task_embeddings_constraints_and_errors() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="ugen")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        max_position_embeddings=20,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
        add_task_embedding=True,
        add_task_prompt_token_in_model=True,
        num_task_prompt_token=2,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    encoded = processor(condition_type="unconditional")
    task_ids = torch.zeros(encoded["input_ids"].size(0), dtype=torch.long)

    with pytest.raises(ValueError, match="task prompt"):
        model.encode(encoded["input_ids"], ~encoded["attention_mask"].bool())
    with pytest.raises(ValueError, match="decoder_input_ids or labels"):
        model(encoded["input_ids"], encoded["attention_mask"], task_ids=task_ids)

    labels = torch.tensor([[5, 6, 1]])
    outputs = model(
        encoded["input_ids"],
        encoded["attention_mask"],
        labels=labels,
        task_ids=task_ids,
        return_dict=False,
    )
    assert len(outputs) == 2

    def only_eos(
        batch_idx: int, step: int, current: torch.Tensor
    ) -> tuple[list[int], int | None]:
        assert batch_idx == 0
        assert step == current.numel()
        return [model.eos_token_id], None

    constrained = model.generate_sequences(
        encoded["input_ids"],
        encoded["attention_mask"],
        max_length=3,
        generation_constraint_fn=only_eos,
        task_ids=task_ids,
    )
    assert constrained.tolist() == [[model.eos_token_id]]

    plain_config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        max_position_embeddings=20,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    plain_model = LayoutFormerPPForConditionalGeneration(plain_config)
    sampled = plain_model.generate_layout(
        processor=processor,
        condition_type=ConditionType.unconditional,
        batch_size=1,
        max_length=1,
        do_sample=True,
        top_k=1,
    )
    assert sampled.sequences.shape == (1, 1)

    with pytest.raises(ValueError, match="processor is required"):
        model.generate_layout()
