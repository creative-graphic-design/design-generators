import torch
import pytest

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


def test_processor_handles_1d_inputs_and_ltwh():
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(
            LayoutDMConfig(
                dataset_name="publaynet",
                bbox_quantization="linear",
                max_seq_length=2,
            )
        )
    )

    out = processor(
        bbox=[[0.4, 0.3, 0.2, 0.2]],
        labels=[1],
        mask=[True],
        box_format=BoxFormat.ltwh,
    )

    assert out["input_ids"].shape == (1, 10)
    assert out["attention_mask"].sum().item() == 5


def test_processor_rejects_unsupported_return_tensors():
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    )

    with pytest.raises(ValueError, match="return_tensors='pt'"):
        processor(
            bbox=[[[0.5, 0.5, 0.2, 0.2]]],
            labels=[[1]],
            return_tensors="np",
        )


def test_processor_requires_canvas_for_pixel_boxes():
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    )

    with pytest.raises(ValueError, match="canvas_size"):
        processor(
            bbox=[[[10, 20, 30, 60]]],
            labels=[[1]],
            box_format="ltrb",
            normalized=False,
        )


def test_processor_save_load_roundtrip(tmp_path):
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    )

    processor.save_pretrained(str(tmp_path))
    loaded = LayoutDMProcessor.from_pretrained(str(tmp_path))

    assert loaded.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
