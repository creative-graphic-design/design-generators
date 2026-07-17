import torch

from layout_flow import LayoutFlowConfig, LayoutFlowProcessor


def test_processor_converts_ltwh_to_xywh_and_pads() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=3)
    )
    out = processor(
        bbox=[[[0.1, 0.2, 0.4, 0.6]]],
        labels=[[2]],
        mask=[[True]],
        box_format="ltwh",
    )
    assert out["bbox"].shape == (1, 3, 4)
    assert torch.allclose(out["bbox"][0, 0], torch.tensor([0.3, 0.5, 0.4, 0.6]))
    assert out["mask"].tolist() == [[True, False, False]]


def test_condition_masks_match_layout_flow_semantics() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=2)
    )
    mask = torch.tensor([[True, True]])
    assert processor.make_condition_mask("unconditional", mask=mask).sum().item() == 14
    label = processor.make_condition_mask("c", mask=mask)
    assert label[:, :, :4].all()
    assert not label[:, :, 4:].any()
    size = processor.make_condition_mask("cwh", mask=mask)
    assert size[:, :, :2].all()
    assert not size[:, :, 2:].any()
