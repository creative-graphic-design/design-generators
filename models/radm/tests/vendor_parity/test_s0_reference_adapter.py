from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from reference_adapter import (
    RADMReferenceAdapter,
    ReferenceUnavailable,
    _legacy_pillow_compat,
    _runtime_text_encoding,
)


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def test_s0_reference_adapter_is_lazy_and_source_selected() -> None:
    source = (
        Path(__file__).with_name("reference_adapter.py").read_text(encoding="utf-8")
    )
    assert "import detectron2" not in source
    assert 'importlib.import_module("detectron2.config")' in source
    assert source.index('importlib.import_module("train_net")') < source.index(
        "detectron2_modeling.build_model(config)"
    )
    if importlib.util.find_spec("detectron2") is None:
        with pytest.raises(ReferenceUnavailable, match="optional Detectron2"):
            RADMReferenceAdapter(
                vendor_root=Path("vendor/radm")
            ).build_initialized_state()
    else:
        adapter = RADMReferenceAdapter(vendor_root=Path("vendor/radm"))
        assert adapter.vendor_root == Path("vendor/radm")


def test_s0_reference_adapter_bridges_detectron2_pillow_symbol() -> None:
    had_linear = hasattr(Image, "LINEAR")
    original_linear = getattr(Image, "LINEAR", None)

    with _legacy_pillow_compat():
        assert getattr(Image, "LINEAR", None) == Image.Resampling.BILINEAR

    if had_linear:
        assert getattr(Image, "LINEAR", None) == original_linear
    else:
        assert not hasattr(Image, "LINEAR")


def test_s0_reference_adapter_derives_text_encoding_from_mapper() -> None:
    """Derive feature width, padding length, and fallback semantics at runtime."""

    class Mapper:
        text_feature_dir = ".cache/radm/nonexistent-text-features"

        @staticmethod
        def load_text(
            text_name: str, max_text_num: int = 17
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            assert text_name.startswith("__radm_s0_missing_")
            features = torch.zeros(max_text_num, 9)
            valid_mask = torch.zeros(max_text_num, 1, dtype=torch.bool)
            return {"feats": features}, valid_mask

    encoding = _runtime_text_encoding(Mapper())

    assert encoding.feature_dim == 9
    assert encoding.max_text_num == 17
    assert encoding.mask_semantics == "true_valid_false_padding"
    assert encoding.missing_fallback == "zero_features_all_padding"
