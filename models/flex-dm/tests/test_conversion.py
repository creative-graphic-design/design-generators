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

    assert map_tensor_by_rule("a/b/kernel:0", value)[0] == "a.b.weight"
    assert map_tensor_by_rule("a/b/bias:0", value)[0] == "a.b.bias"
    assert map_tensor_by_rule("a/b/beta:0", value)[0] == "a.b.bias"
    assert map_tensor_by_rule("a/b/gamma:0", value)[0] == "a.b.weight"
    assert map_tensor_by_rule("a/b/embeddings:0", value)[0] == "a.b.weight"
    assert map_tensor_by_rule("a/b/other:0", value) is None
