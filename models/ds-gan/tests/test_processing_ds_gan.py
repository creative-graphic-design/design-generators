import numpy as np
import torch
from PIL import Image
from typing import cast

from laygen.modeling_outputs import LayoutGenerationOutput

from ds_gan import DSGANProcessor


def test_processor_merges_saliency_and_resizes():
    processor = DSGANProcessor(image_size=(8, 6))
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    saliency_a = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
    saliency_b = Image.fromarray(np.full((4, 4), 128, dtype=np.uint8))

    encoded = processor(
        image,
        saliency_pfpnet=saliency_a,
        saliency_basnet=saliency_b,
    )

    assert encoded["pixel_values"].shape == (1, 4, 8, 6)
    assert float(encoded["pixel_values"][:, 3].max()) > 0.49


def test_processor_handles_tensor_batches_and_direct_saliency():
    processor = DSGANProcessor(image_size=(8, 6))
    images = torch.full((2, 4, 4, 3), 255.0)
    saliency = torch.zeros(2, 1, 4, 4)

    encoded = processor(images, saliency=saliency)

    assert encoded["pixel_values"].shape == (2, 4, 8, 6)


def test_processor_rejects_bad_return_tensor_and_saliency_batch():
    processor = DSGANProcessor(image_size=(8, 6))
    image = torch.zeros(3, 4, 4)

    try:
        processor(image, return_tensors="np")  # ty: ignore[invalid-argument-type]
    except ValueError as exc:
        assert "return_tensors='pt'" in str(exc)
    else:
        raise AssertionError("expected return_tensors to raise")

    try:
        processor([image, image], saliency=[torch.zeros(4, 4)])
    except ValueError as exc:
        assert "batch size must match" in str(exc)
    else:
        raise AssertionError("expected saliency batch mismatch to raise")


def test_decode_maps_vendor_no_object_to_mask():
    processor = DSGANProcessor()
    class_probs = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [1.2, -0.1, 0.3, 0.4]]])

    output = processor.decode(class_probs=class_probs, bbox=bbox)

    assert isinstance(output, LayoutGenerationOutput)
    assert output.mask.tolist() == [[False, True]]
    assert output.labels.tolist() == [[0, 1]]
    assert output.bbox.max() <= 1
    assert output.id2label == {0: "text", 1: "logo", 2: "underlay"}


def test_decode_dict_scores_and_bad_output_type():
    processor = DSGANProcessor()
    class_probs = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    bbox = torch.zeros(1, 1, 4)
    scores = torch.ones(1, 1)

    decoded = processor.decode(
        class_probs=class_probs,
        bbox=bbox,
        output_type="dict",
        scores=scores,
        intermediates={"ok": True},
    )

    decoded = cast(dict[str, object], decoded)
    assert cast(torch.Tensor, decoded["scores"]).tolist() == [[1.0]]
    assert decoded["intermediates"] == {"ok": True}
    try:
        processor.decode(
            class_probs=class_probs,
            bbox=bbox,
            output_type="bad",  # ty: ignore[invalid-argument-type]
        )
    except ValueError as exc:
        assert "Unsupported output_type" in str(exc)
    else:
        raise AssertionError("expected output_type to raise")


def test_encode_layout_filters_padding_to_vendor_zero_class():
    processor = DSGANProcessor()

    encoded = processor.encode_layout(
        bbox=[[[10, 20, 110, 220]]],
        labels=[[2]],
        mask=[[False]],
        box_format="ltrb",
        normalized=False,
        canvas_size=(200, 400),
        max_elem=2,
    )

    assert encoded["layout"].shape == (1, 2, 2, 4)
    assert encoded["layout"][0, 0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert encoded["mask"].tolist() == [[False, False]]


def test_encode_layout_errors_and_normalized_ltrb():
    processor = DSGANProcessor()

    encoded = processor.encode_layout(
        bbox=[[[0.0, 0.0, 1.0, 1.0]]],
        labels=[[0]],
        box_format="ltrb",
        normalized=True,
        max_elem=1,
    )
    assert encoded["bbox"].tolist() == [[[0.5, 0.5, 1.0, 1.0]]]

    try:
        processor.encode_layout(
            bbox=[[[0, 0, 10, 10]]],
            labels=[[0]],
            normalized=False,
        )
    except ValueError as exc:
        assert "canvas_size is required" in str(exc)
    else:
        raise AssertionError("expected missing canvas_size to raise")

    try:
        processor.pad(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            max_elem=1,
        )
    except ValueError as exc:
        assert "supports at most" in str(exc)
    else:
        raise AssertionError("expected too many elements to raise")
