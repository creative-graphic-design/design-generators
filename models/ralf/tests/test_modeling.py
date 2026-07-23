from collections.abc import Mapping
from typing import cast

import pytest
import torch

from ralf import RalfConfig, RalfForConditionalLayoutGeneration, RalfProcessor


def tiny_config() -> RalfConfig:
    return RalfConfig(
        max_seq_length=2,
        num_bin=8,
        decoder_d_model=16,
        decoder_layers=1,
        num_attention_heads=4,
        d_model=16,
    )


def _stub_encode_memory(model: RalfForConditionalLayoutGeneration) -> None:
    def encode(
        inputs: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        image = cast(torch.Tensor, inputs["image"])
        return {
            "memory": torch.zeros(
                image.size(0),
                1,
                model.config.decoder_d_model,
                device=image.device,
            )
        }

    model._encode_into_memory = encode  # ty: ignore[invalid-assignment]


def test_forward_logits_shape() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    _stub_encode_memory(model)
    input_ids = torch.tensor([[config.bos_token_id, 0, config.pad_token_id]])
    attention_mask = input_ids.ne(config.pad_token_id)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 3, config.vocab_size)


def test_generate_sequences_respects_generator() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    _stub_encode_memory(model)
    input_ids = torch.tensor([[config.bos_token_id]])
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    first = model._generate_sequences(input_ids, generator=generator_a, max_length=3)
    second = model._generate_sequences(input_ids, generator=generator_b, max_length=3)

    assert torch.equal(first, second)


def test_forward_loss_tuple_and_image_context_branches() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    _stub_encode_memory(model)
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


def test_task_name_aliases() -> None:
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_task_name("unconditional")
        == "uncond"
    )
    assert RalfForConditionalLayoutGeneration._canonical_to_task_name("label") == "c"
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_task_name("label_size")
        == "cwh"
    )
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_task_name("completion")
        == "partial"
    )
    assert (
        RalfForConditionalLayoutGeneration._canonical_to_task_name("relation")
        == "relation"
    )


def test_model_accepts_label_checkpoint_task() -> None:
    config = tiny_config()
    config.task = "label"

    model = RalfForConditionalLayoutGeneration(config)

    assert model.auxilary_task == "c"


def test_model_accepts_relation_checkpoint_task() -> None:
    config = tiny_config()
    config.task = "relation"

    model = RalfForConditionalLayoutGeneration(config)

    assert model.auxilary_task == "relation"


def test_prepare_inputs_expands_explicit_retrieved_rgb_and_global_task() -> None:
    from ralf import RalfRetrievedBatch

    config = tiny_config()
    config.global_task_embedding = True
    config.use_flag_embedding = False
    model = RalfForConditionalLayoutGeneration(config)
    retrieved = RalfRetrievedBatch(
        image=torch.zeros(1, config.top_k, 3, 1, 1),
        saliency=torch.zeros(1, config.top_k, 1, 1, 1),
        bbox=torch.zeros(1, config.top_k, config.max_seq_length, 4),
        labels=torch.zeros(1, config.top_k, config.max_seq_length, dtype=torch.long),
        mask=torch.ones(1, config.top_k, config.max_seq_length, dtype=torch.bool),
    )

    prepared = model._prepare_unconditional_inputs(
        pixel_values=torch.zeros(1, 3, 8, 8),
        saliency=None,
        retrieved=retrieved,
        batch_size=1,
    )

    retrieved_dict = cast(dict[str, torch.Tensor], prepared["retrieved"])
    seq_layout_const = cast(torch.Tensor, prepared["seq_layout_const"])
    assert retrieved_dict["image"].shape[2] == 4
    assert seq_layout_const.shape[0] == 1


def test_generate_sequences_accepts_label_constraints() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    _stub_encode_memory(model)
    processor = RalfProcessor.from_config(config)
    encoded = processor(
        condition_type="label",
        labels=[[0, 1]],
        bbox=torch.full((1, 2, 4), 0.5),
        mask=torch.tensor([[True, True]]),
    )

    output = model._generate_sequences(
        encoded["input_ids"],
        condition_type="label",
        constraint_input_ids=encoded["input_ids"],
        constraint_mask=encoded["attention_mask"],
        constraint_element_mask=encoded["constraint_mask"],
        max_length=3,
        top_k=5,
        generator=torch.Generator().manual_seed(5),
    )

    assert output.shape[0] == 1


def test_generate_sequences_accepts_remaining_condition_constraints() -> None:
    config = tiny_config()
    model = RalfForConditionalLayoutGeneration(config)
    _stub_encode_memory(model)
    processor = RalfProcessor.from_config(config)
    encoded = processor(
        labels=[[0, 1]],
        bbox=torch.full((1, 2, 4), 0.5),
        mask=torch.tensor([[True, False]]),
    )

    for condition_type in [
        "label_size",
        "completion",
        "refinement",
        "relation",
        "content_image",
    ]:
        output = model._generate_sequences(
            encoded["input_ids"],
            condition_type=condition_type,
            constraint_input_ids=encoded["input_ids"],
            constraint_mask=encoded["attention_mask"],
            constraint_element_mask=encoded["constraint_mask"],
            max_length=2,
            top_k=5,
            generator=torch.Generator().manual_seed(6),
        )

        assert output.shape[0] == 1
