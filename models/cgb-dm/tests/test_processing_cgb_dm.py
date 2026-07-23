import torch
from PIL import Image

from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from cgb_dm import CGBDMProcessor


def test_processor_image_and_saliency_merge():
    processor = CGBDMProcessor(image_size=(32, 32), max_seq_length=2)
    encoded = processor(
        torch.zeros(3, 32, 32),
        saliency_isnet=torch.zeros(32, 32),
        saliency_basnet=torch.ones(32, 32),
    )

    assert encoded["pixel_values"].shape == (1, 4, 32, 32)
    assert encoded["saliency_box"].shape == (1, 1, 4)


def test_layout_encode_decode_filters_invalid():
    processor = CGBDMProcessor(max_seq_length=3)
    encoded = processor.encode_layout(
        bbox=[[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]],
        labels=[[0, 3]],
        mask=[[True, True]],
    )
    assert encoded["layout"].shape == (1, 3, 8)

    output = processor.decode(encoded["layout"])
    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.shape == (1, 3, 4)
    assert output.mask.tolist() == [[False, False, False]]


def test_processor_error_and_path_inputs(tmp_path):
    processor = CGBDMProcessor(image_size=(16, 16), max_seq_length=1)
    image_path = tmp_path / "image.png"
    sal_path = tmp_path / "sal.png"
    Image.new("RGB", (8, 8)).save(image_path)
    Image.new("L", (8, 8), color=255).save(sal_path)

    encoded = processor(image_path, saliency=sal_path)
    assert encoded["pixel_values"].shape == (1, 4, 16, 16)

    try:
        processor(None)
    except ValueError as exc:
        assert "images or pixel_values" in str(exc)
    else:
        raise AssertionError("expected missing image error")

    try:
        processor(torch.zeros(2, 2, 2))
    except ValueError as exc:
        assert "RGB tensor" in str(exc)
    else:
        raise AssertionError("expected RGB shape error")


def test_processor_output_type_error_and_pixel_boxes():
    processor = CGBDMProcessor(max_seq_length=1)
    encoded = processor.encode_layout(
        bbox=[[[0, 0, 10, 10]]],
        labels=[[1]],
        box_format="ltrb",
        normalized=False,
        canvas_size=(100, 100),
    )
    assert encoded["bbox"].shape == (1, 1, 4)
    try:
        processor.decode(encoded["layout"], output_type="bad")
    except ValueError as exc:
        assert "Unsupported output_type" in str(exc)
    else:
        raise AssertionError("expected bad output type error")
