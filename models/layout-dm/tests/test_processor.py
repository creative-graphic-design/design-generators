import pytest
import torch

from laygen.common.bbox import BoxFormat
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def test_processor_accepts_pixel_ltrb():
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(
            LayoutDMConfig(dataset_name="publaynet", bbox_quantization="linear")
        )
    )
    out = processor(
        bbox=[[[10, 20, 30, 60]]],
        labels=[[1]],
        box_format=BoxFormat.ltrb,
        normalized=False,
        canvas_size=(100, 100),
    )
    assert out["input_ids"].shape == (1, 125)
    assert out["mask"].dtype == torch.bool


def test_processor_accepts_1d_mask_ltwh_and_save_load(tmp_path):
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(
            LayoutDMConfig(
                dataset_name="publaynet",
                bbox_quantization="linear",
                max_seq_length=1,
            )
        )
    )
    out = processor(
        bbox=[[0.0, 0.0, 0.5, 0.5]],
        labels=[1],
        mask=[True],
        box_format="ltwh",
    )
    assert out["input_ids"].shape == (1, 5)

    processor.save_pretrained(str(tmp_path))
    loaded = LayoutDMProcessor.from_pretrained(str(tmp_path))
    assert loaded.tokenizer.config.max_seq_length == 1


def test_processor_rejects_unsupported_return_tensors_and_missing_canvas():
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    )
    with pytest.raises(ValueError, match="return_tensors"):
        processor(bbox=[[[0.0, 0.0, 1.0, 1.0]]], labels=[[0]], return_tensors="np")
    with pytest.raises(ValueError, match="canvas_size is required"):
        processor(
            bbox=[[[0.0, 0.0, 10.0, 10.0]]],
            labels=[[0]],
            normalized=False,
        )
