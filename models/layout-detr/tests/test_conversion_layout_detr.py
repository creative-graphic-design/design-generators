import torch
import pytest

from layout_detr import LayoutDetrConfig, LayoutDetrForConditionalGeneration
from layout_detr.vendor_state import (
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
