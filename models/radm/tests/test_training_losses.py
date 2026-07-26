import torch

from radm.training.losses import dynamic_k_match, radm_losses


def test_dynamic_k_match_selects_targets() -> None:
    logits = torch.tensor([[4.0, -1.0], [-1.0, 4.0], [0.0, 0.0]])
    boxes = torch.tensor(
        [[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0], [0.4, 0.4, 0.6, 0.6]]
    )
    targets = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]])
    labels = torch.tensor([0, 1])
    query, target = dynamic_k_match(
        logits=logits,
        boxes_xyxy=boxes,
        target_boxes_xyxy=targets,
        target_labels=labels,
        ota_k=2,
    )
    assert query.numel() >= 2
    assert set(target.tolist()) == {0, 1}


def test_radm_losses_are_finite() -> None:
    out = radm_losses(
        logits=torch.randn(1, 4, 3),
        boxes_xyxy=torch.rand(1, 4, 4).sort(dim=-1).values,
        target_boxes_xyxy=torch.tensor([[[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.7, 0.7]]]),
        target_labels=torch.tensor([[0, 2]]),
        target_mask=torch.tensor([[True, True]]),
        class_weight=5.0,
        bbox_weight=1.0,
        giou_weight=1.0,
        alpha=0.25,
        gamma=2.0,
        ota_k=2,
    )
    assert torch.isfinite(out.train_loss)
