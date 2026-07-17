import torch

from layout_flow import LayoutFlowConfig, LayoutFlowProcessor


def test_analog_bits_round_trip_for_layout_flow_vocabularies() -> None:
    for dataset in ("rico25", "publaynet"):
        processor = LayoutFlowProcessor(LayoutFlowConfig(dataset_name=dataset))
        labels = torch.arange(processor.config.num_labels).unsqueeze(0)
        assert torch.equal(
            processor.decode_labels(processor.encode_labels(labels)), labels
        )


def test_analog_bit_threshold_matches_vendor_rule() -> None:
    processor = LayoutFlowProcessor(LayoutFlowConfig(dataset_name="publaynet"))
    bits = torch.tensor([[[0.49, 0.50, 0.49]]])
    assert processor.decode_labels(bits).item() == 2
