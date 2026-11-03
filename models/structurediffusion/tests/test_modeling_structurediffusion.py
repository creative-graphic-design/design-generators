from __future__ import annotations

import pytest
import torch
from diffusers.models.attention_processor import Attention
from jaxtyping import Float

from structurediffusion.modeling_structurediffusion import (
    StructuredAttentionProcessor,
    StructuredConditioning,
)


def _attention() -> Attention:
    return Attention(query_dim=4, cross_attention_dim=4, heads=2, dim_head=2).eval()


def _key_projection(
    attention: Attention,
    context: Float[torch.Tensor, "batch tokens dim"],
) -> Float[torch.Tensor, "batch tokens inner_dim"]:
    if attention.to_k is None:
        raise AssertionError("The attention layer has no key projection.")
    return attention.to_k(context)


def _value_projection(
    attention: Attention,
    context: Float[torch.Tensor, "batch tokens dim"],
) -> Float[torch.Tensor, "batch tokens inner_dim"]:
    if attention.to_v is None:
        raise AssertionError("The attention layer has no value projection.")
    return attention.to_v(context)


def _conditioning(
    keys: tuple[Float[torch.Tensor, "batch tokens dim"], ...],
    values: tuple[Float[torch.Tensor, "batch tokens dim"], ...],
    do_cfg: bool = False,
) -> StructuredConditioning:
    return StructuredConditioning(
        keys=keys,
        values=values,
        key_validity=tuple(
            torch.ones(keys[0].shape[0], dtype=torch.bool) for _ in keys
        ),
        value_validity=tuple(
            torch.ones(keys[0].shape[0], dtype=torch.bool) for _ in values
        ),
        do_classifier_free_guidance=do_cfg,
    )


def test_head_layout_round_trip() -> None:
    values = torch.randn((2, 5, 8))
    heads = StructuredAttentionProcessor._to_heads(values, heads=2)
    restored = StructuredAttentionProcessor._from_heads(heads, heads=2)
    assert torch.equal(values, restored)


def test_unstructured_path_matches_diffusers_default() -> None:
    attention = _attention()
    hidden = torch.randn((1, 4, 4))
    context = torch.randn((1, 5, 4))
    expected = attention.processor(attention, hidden, context)
    actual = StructuredAttentionProcessor("cross", False)(
        attention,
        hidden,
        context,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_attention_supports_spatial_and_group_normalization_with_residuals() -> None:
    class SpatialNorm(torch.nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            temb: torch.Tensor | None,
        ) -> torch.Tensor:
            del temb
            return hidden_states

    attention = _attention()
    object.__setattr__(attention, "spatial_norm", SpatialNorm())
    attention.group_norm = torch.nn.GroupNorm(num_groups=2, num_channels=4)
    object.__setattr__(attention, "norm_q", torch.nn.Identity())
    object.__setattr__(attention, "norm_k", torch.nn.Identity())
    attention.residual_connection = True
    hidden_states = torch.randn((1, 4, 2, 2))
    context = torch.randn((1, 5, 4))

    output = StructuredAttentionProcessor("cross", False)(
        attention,
        hidden_states,
        context,
        temb=torch.randn((1, 4)),
    )

    assert output.shape == hidden_states.shape


def test_processor_rejects_attention_without_query_projection() -> None:
    attention = _attention()
    object.__setattr__(attention, "to_q", None)

    with pytest.raises(ValueError, match="query, key, value, and output projections"):
        StructuredAttentionProcessor("cross", False)(
            attention,
            torch.randn((1, 3, 4)),
            torch.randn((1, 5, 4)),
        )


def test_unstructured_path_uses_hidden_states_when_context_is_missing() -> None:
    attention = _attention()
    hidden_states = torch.randn((1, 4, 4))

    output = StructuredAttentionProcessor("self", False)(
        attention,
        hidden_states,
    )

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("mask_rank", ["2d", "4d", "3d_batch", "3d_heads"])
@pytest.mark.parametrize("mask_kind", ["bool", "float"])
def test_attention_probability_masks_cover_broadcast_shapes(
    mask_rank: str,
    mask_kind: str,
) -> None:
    attention = _attention()
    query = torch.zeros((2, 3, 2))
    key = torch.zeros((2, 5, 2))
    if mask_kind == "bool":
        mask = torch.ones((1, 5), dtype=torch.bool)
        mask[:, -1] = False
    else:
        mask = torch.zeros((1, 5))
        mask[:, -1] = -100

    if mask_rank == "4d":
        mask = mask[:, None, None, :].expand(1, 1, 3, 5)
    elif mask_rank == "3d_batch":
        mask = mask[:, None, :]
    elif mask_rank == "3d_heads":
        mask = mask.expand(2, 5)[:, None, :]

    probabilities = StructuredAttentionProcessor._attention_probabilities(
        attention,
        query,
        key,
        mask,
    )

    assert probabilities.shape == (2, 3, 5)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones((2, 3)))
    assert torch.all(probabilities[..., -1] < 1e-6)


def test_one_full_prompt_key_averages_full_and_phrase_values() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    query = torch.randn((1, 3, 4))
    full_context = torch.randn((1, 5, 4))
    phrase_context = torch.randn((1, 5, 4))
    processor.conditioning = _conditioning(
        keys=(full_context,),
        values=(full_context, phrase_context),
    )

    if attention.to_q is None:
        raise AssertionError("The attention layer has no query projection.")
    projected_query = attention.to_q(query)
    actual = processor._structured_attention(attention, projected_query, None)
    query_heads = processor._to_heads(projected_query, attention.heads)
    key_heads = processor._to_heads(
        _key_projection(attention, full_context), attention.heads
    )
    value_heads = [
        processor._to_heads(_value_projection(attention, context), attention.heads)
        for context in (full_context, phrase_context)
    ]
    probabilities = processor._attention_probabilities(
        attention,
        query_heads,
        key_heads,
        None,
    )
    expected = processor._from_heads(
        (
            torch.bmm(probabilities, value_heads[0])
            + torch.bmm(probabilities, value_heads[1])
        )
        / 2,
        attention.heads,
    )
    assert torch.allclose(actual, expected)


def test_multiple_keys_with_one_conjunction_value_average_attention_maps() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    query = torch.randn((1, 3, 4))
    first_context = torch.randn((1, 5, 4))
    second_context = torch.randn((1, 5, 4))
    combined_context = torch.randn((1, 5, 4))
    processor.conditioning = _conditioning(
        keys=(first_context, second_context),
        values=(combined_context,),
    )

    if attention.to_q is None:
        raise AssertionError("The attention layer has no query projection.")
    projected_query = attention.to_q(query)
    actual = processor._structured_attention(attention, projected_query, None)
    query_heads = processor._to_heads(projected_query, attention.heads)
    attention_maps = [
        processor._attention_probabilities(
            attention,
            query_heads,
            processor._to_heads(_key_projection(attention, context), attention.heads),
            None,
        )
        for context in (first_context, second_context)
    ]
    value_heads = processor._to_heads(
        _value_projection(attention, combined_context),
        attention.heads,
    )
    expected = processor._from_heads(
        (
            torch.bmm(attention_maps[0], value_heads)
            + torch.bmm(attention_maps[1], value_heads)
        )
        / 2,
        attention.heads,
    )
    assert torch.allclose(actual, expected)


def test_cfg_separates_unconditional_and_structured_branches() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    keys = torch.randn((1, 5, 4))
    phrase = torch.randn((1, 5, 4))
    unconditional = torch.randn((1, 5, 4))
    processor.conditioning = _conditioning((keys,), (keys, phrase), do_cfg=True)

    output = processor(
        attention,
        torch.randn((2, 3, 4)),
        torch.cat((unconditional, keys)),
        attention_mask=torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, False]]
        ),
    )

    assert output.shape == (2, 3, 4)


def test_cfg_rejects_mismatched_query_batch() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    context = torch.randn((1, 5, 4))
    processor.conditioning = _conditioning((context,), (context,), do_cfg=True)
    with pytest.raises(ValueError, match="matching unconditional and conditional"):
        processor(attention, torch.randn((3, 3, 4)), torch.randn((3, 5, 4)))


def test_structured_path_rejects_mismatched_query_batch() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    context = torch.randn((1, 5, 4))
    processor.conditioning = _conditioning((context,), (context,))

    with pytest.raises(ValueError, match="conditioning and query batches must match"):
        processor(attention, torch.randn((2, 3, 4)), torch.randn((2, 5, 4)))


def test_attention_maps_keep_spatial_axes_and_first_twenty_tokens() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", True)
    context = torch.randn((1, 25, 4))
    processor.conditioning = _conditioning((context,), (context,))

    output = processor(attention, torch.randn((1, 4, 2, 2)), context)

    assert output.shape == (1, 4, 2, 2)
    assert processor.attention_maps[0].shape == (1, 2, 2, 2, 20)


def test_attention_maps_infer_square_spatial_shape_from_sequence() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", True)
    context = torch.randn((1, 25, 4))
    processor.conditioning = _conditioning((context,), (context,))

    processor(attention, torch.randn((1, 4, 4)), context)

    assert processor.attention_maps[0].shape == (1, 2, 2, 2, 20)


def test_attention_maps_skip_non_square_query_sequences() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", True)
    context = torch.randn((1, 25, 4))

    processor(attention, torch.randn((1, 3, 4)), context)

    assert processor.attention_maps == []


def test_ragged_phrase_groups_ignore_padded_contexts() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    query = torch.randn((2, 3, 4))
    full = torch.randn((2, 5, 4))
    phrase = torch.randn((2, 5, 4))
    padded = torch.randn((2, 5, 4))
    processor.conditioning = StructuredConditioning(
        keys=(full,),
        values=(full, phrase, padded),
        key_validity=(torch.ones((2,), dtype=torch.bool),),
        value_validity=(
            torch.ones((2,), dtype=torch.bool),
            torch.ones((2,), dtype=torch.bool),
            torch.tensor([True, False]),
        ),
        do_classifier_free_guidance=False,
    )

    if attention.to_q is None:
        raise AssertionError("The attention layer has no query projection.")
    projected_query = attention.to_q(query)
    actual = processor._structured_attention(attention, projected_query, None)
    query_heads = processor._to_heads(projected_query, attention.heads)
    key_heads = processor._to_heads(_key_projection(attention, full), attention.heads)
    probabilities = processor._attention_probabilities(
        attention,
        query_heads,
        key_heads,
        None,
    )
    value_heads = [
        processor._to_heads(_value_projection(attention, context), attention.heads)
        for context in (full, phrase, padded)
    ]
    per_value = torch.stack([torch.bmm(probabilities, value) for value in value_heads])
    expected = torch.stack(
        (
            per_value[:, : attention.heads].mean(dim=0),
            per_value[:2, attention.heads :].mean(dim=0),
        )
    )
    expected = processor._from_heads(expected.reshape(4, 3, 2), attention.heads)

    assert torch.allclose(actual, expected)


def test_structured_attention_rejects_two_nonmatching_lists() -> None:
    attention = _attention()
    processor = StructuredAttentionProcessor("cross", False)
    contexts = tuple(torch.randn((1, 5, 4)) for _ in range(2))
    processor.conditioning = _conditioning(
        contexts, (contexts[0], contexts[1], contexts[0])
    )
    with pytest.raises(ValueError, match="matching lists or a singleton side"):
        processor._structured_attention(attention, torch.randn((1, 3, 4)), None)
