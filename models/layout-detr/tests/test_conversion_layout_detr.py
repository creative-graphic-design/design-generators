import torch
import pytest

from layout_detr import LayoutDetrConfig, LayoutDetrForConditionalGeneration
from layout_detr.vendor_state import (
    _LegacyTokenizerHelper,
    _LegacyTrie,
    _find_pruneable_heads_and_indices,
    _install_transformers_vendor_compat,
    _patch_legacy_unpickler,
    build_conversion_report,
    remap_generator_key,
    strict_load_converted_state,
    temporary_sys_path,
)


def test_remap_generator_key_and_report_counts():
    assert remap_generator_key("module.emb_label.weight") == "emb_label.weight"
    source = {"a": torch.zeros(1), "b": torch.zeros(2)}
    target = {"a": torch.zeros(1), "c": torch.zeros(1)}
    report = build_conversion_report(
        source,
        target,
        source,
        custom_op_import_required=False,
    )

    assert report["source_key_count"] == 2
    assert report["target_key_count"] == 2
    assert report["loaded_key_count"] == 1
    assert report["missing_keys"] == ["c"]
    assert report["unexpected_keys"] == ["b"]


def test_strict_load_refuses_missing_keys():
    model = LayoutDetrForConditionalGeneration(
        LayoutDetrConfig(
            hidden_dim=16, bert_f_dim=16, max_text_length=4, text_vocab_size=64
        )
    )
    with pytest.raises(RuntimeError):
        strict_load_converted_state(model, {})


def test_temporary_sys_path_adds_and_removes_path(tmp_path):
    import sys

    raw = str(tmp_path)
    assert raw not in sys.path
    with temporary_sys_path(tmp_path):
        assert sys.path[0] == raw
    assert raw not in sys.path


def test_transformers_vendor_compat_shims_are_installed():
    import transformers.modeling_utils as modeling_utils

    _install_transformers_vendor_compat()

    assert hasattr(modeling_utils, "apply_chunking_to_forward")
    assert hasattr(modeling_utils, "prune_linear_layer")
    assert hasattr(modeling_utils, "find_pruneable_heads_and_indices")

    heads, index = _find_pruneable_heads_and_indices({0}, 2, 3, set())
    assert heads == {0}
    assert index.tolist() == [3, 4, 5]


def test_legacy_unpickler_patch_resolves_removed_tokenizer_helpers():
    class FakeUnpickler:
        _layout_detr_compat_patched = False

        def find_class(self, module, name):
            return (module, name)

    class FakeLegacy:
        _LegacyUnpickler = FakeUnpickler

    _patch_legacy_unpickler(FakeLegacy)

    assert FakeUnpickler._layout_detr_compat_patched is True
    unpickler = FakeUnpickler()
    assert (
        unpickler.find_class("transformers.tokenization_utils", "Trie") is _LegacyTrie
    )
    assert (
        unpickler.find_class(
            "transformers.models.bert.tokenization_bert", "BasicTokenizer"
        )
        is _LegacyTokenizerHelper
    )
    assert unpickler.find_class("pkg", "Name") == ("pkg", "Name")
