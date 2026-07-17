import torch

from lace import LaceProcessor


def test_encode_maps_public_bbox_to_vendor_range() -> None:
    processor = LaceProcessor.from_dataset("publaynet")
    bbox = torch.tensor([[[0.0, 0.5, 1.0, 0.25]]])
    labels = torch.tensor([[2]])
    encoded = processor.encode(bbox, labels)
    assert encoded.shape == (1, 25, 10)
    assert torch.equal(encoded[0, 0, :6], torch.tensor([0, 0, 1, 0, 0, 0]))
    assert torch.allclose(encoded[0, 0, 6:], torch.tensor([-1.0, 0.0, 1.0, -0.5]))
    assert encoded[0, 1, processor.pad_label_id] == 1


def test_decode_encode_roundtrip_uses_mask_for_padding() -> None:
    processor = LaceProcessor.from_dataset("rico13")
    bbox = torch.tensor([[[0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]]])
    labels = torch.tensor([[4, 0]])
    mask = torch.tensor([[True, False]])
    encoded = processor.encode(bbox, labels, mask)
    decoded = processor.decode(encoded)
    assert torch.allclose(decoded.bbox[0, 0], bbox[0, 0])
    assert decoded.labels[0, 0].item() == 4
    assert decoded.labels[0, 1].item() == processor.pad_label_id
    assert decoded.mask[0, 0].item() is True
    assert decoded.mask[0, 1].item() is False


def test_processor_converts_ltrb_input() -> None:
    processor = LaceProcessor.from_dataset("publaynet")
    out = processor(
        bbox=[[[0.0, 0.0, 1.0, 1.0]]],
        labels=[[1]],
        box_format="ltrb",
    )
    assert torch.allclose(out["bbox"][0, 0], torch.tensor([0.5, 0.5, 1.0, 1.0]))
