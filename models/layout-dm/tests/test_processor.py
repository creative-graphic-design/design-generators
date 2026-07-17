import torch
from typing import Literal, cast
from transformers import ProcessorMixin

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


def test_processor_accepts_unbatched_ltwh_and_validates_options(tmp_path):
    processor = LayoutDMProcessor(
        LayoutDMTokenizer(
            LayoutDMConfig(
                dataset_name="publaynet",
                bbox_quantization="linear",
                max_seq_length=2,
                num_bin_bboxes=4,
            )
        )
    )
    out = processor(
        bbox=[[0.4, 0.4, 0.2, 0.2]],
        labels=[1],
        mask=[True],
        box_format="ltwh",
    )
    assert out["input_ids"].shape == (1, 10)

    try:
        processor(
            bbox=[[[0.0, 0.0, 1.0, 1.0]]],
            labels=[[0]],
            return_tensors=cast(Literal["pt"], "np"),
        )
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("unsupported return_tensors should fail")

    try:
        processor(
            bbox=[[[0.0, 0.0, 1.0, 1.0]]],
            labels=[[0]],
            normalized=False,
        )
    except ValueError as exc:
        assert "canvas_size is required" in str(exc)
    else:
        raise AssertionError("missing canvas_size should fail")

    processor.save_pretrained(str(tmp_path))
    assert isinstance(processor, ProcessorMixin)
    assert (tmp_path / "processor_config.json").exists()
    assert isinstance(
        LayoutDMProcessor.from_pretrained(str(tmp_path)), LayoutDMProcessor
    )
