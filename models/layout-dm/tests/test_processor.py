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
