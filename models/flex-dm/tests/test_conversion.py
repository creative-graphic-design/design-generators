"""Flex-DM conversion helper tests."""

import numpy as np
import torch

from flex_dm.conversion import (
    conversion_report,
    convert_dense_bias,
    convert_dense_kernel,
    convert_embedding,
    convert_layer_norm_gamma_beta,
    map_tensor_by_rule,
)


def test_dense_transpose_and_basic_converters() -> None:
    """TensorFlow Dense kernels transpose into PyTorch layout."""
    kernel = np.arange(6, dtype=np.float32).reshape(2, 3)
    bias = np.arange(3, dtype=np.float32)
    embedding = np.arange(4, dtype=np.float32).reshape(2, 2)

    assert torch.equal(convert_dense_kernel(kernel), torch.from_numpy(kernel.T))
    assert torch.equal(convert_dense_bias(bias), torch.from_numpy(bias))
    assert torch.equal(convert_embedding(embedding), torch.from_numpy(embedding))
    gamma, beta = convert_layer_norm_gamma_beta(bias, bias + 1)
    assert torch.equal(gamma, torch.from_numpy(bias))
    assert torch.equal(beta, torch.from_numpy(bias + 1))


def test_conversion_report_missing_and_unexpected() -> None:
    """Conversion reports strict missing and unexpected key sets."""
    report = conversion_report(
        converted={"a": torch.ones(2)},
        target_keys={"a", "b"},
        source_keys={"x", "y"},
        consumed_source_keys={"x"},
    )

    assert report.matched_tensor_count == 1
    assert report.matched_parameter_count == 2
    assert report.missing_target_keys == ("b",)
    assert report.unexpected_source_keys == ("y",)


def test_map_tensor_by_rule_branches() -> None:
    """Semantic mapping handles dense, norm, embedding, and ignored tensors."""
    value = np.ones((2, 3), dtype=np.float32)

    assert (
        map_tensor_by_rule("model/encoder/input_layer/type/embeddings:0", value)[0]
        == "encoder.input_embeddings.field_type.weight"
    )
    assert (
        map_tensor_by_rule(
            "model/encoder/input_layer/type_special/embeddings:0", value
        )[0]
        == "encoder.special_embeddings.field_type.weight"
    )
    assert (
        map_tensor_by_rule("model/encoder/input_layer/left/kernel:0", value)[0]
        == "encoder.input_projections.field_left.weight"
    )
    assert (
        map_tensor_by_rule("model/encoder/input_layer/left/bias:0", value)[0]
        == "encoder.input_projections.field_left.bias"
    )
    assert (
        map_tensor_by_rule(
            "model/blocks/seq2seq/seq2seq_0/attn/dense_query/kernel:0", value
        )[0]
        == "blocks.0.attention.q_proj.weight"
    )
    assert (
        map_tensor_by_rule(
            "model/blocks/seq2seq/seq2seq_1/attn/dense_key/bias:0", value
        )[0]
        == "blocks.1.attention.k_proj.bias"
    )
    assert (
        map_tensor_by_rule("model/blocks/seq2seq/seq2seq_2/norm1/gamma:0", value)[0]
        == "blocks.2.norm1.weight"
    )
    assert (
        map_tensor_by_rule("model/blocks/seq2seq/seq2seq_3/norm2/beta:0", value)[0]
        == "blocks.3.norm2.bias"
    )
    assert (
        map_tensor_by_rule("model/decoder/decoders/type/kernel:0", value)[0]
        == "decoder.heads.field_type.weight"
    )
    assert (
        map_tensor_by_rule("model/decoder/decoders/type/bias:0", value)[0]
        == "decoder.heads.field_type.bias"
    )
    assert (
        map_tensor_by_rule("optimizer/model/encoder/input_layer/type:0", value) is None
    )
    assert (
        map_tensor_by_rule("model/encoder/input_layer/type/.OPTIMIZER_SLOT/m:0", value)
        is None
    )
    assert map_tensor_by_rule("model/encoder/input_layer/type/other:0", value) is None
