import torch

from canvas_vae.metrics import (
    bleu1,
    component_grid,
    field_histograms,
    histogram_scores,
    layout_scores,
    reconstruction_scores,
)


def test_bleu1_precision_and_brevity():
    target = torch.tensor([[1, 2, 2, 0]])
    target_mask = torch.tensor([[True, True, True, False]])
    predicted = torch.tensor([[2, 3]])
    predicted_mask = torch.tensor([[True, True]])
    score = bleu1(target, target_mask, predicted, predicted_mask, 4)
    expected = torch.exp(torch.tensor(1 - 3 / 2)) * torch.sqrt(torch.tensor(0.5))
    torch.testing.assert_close(score, expected.reshape(1))


def test_reconstruction_scores_total_is_field_mean():
    sizes = {
        name: 4
        for name in (
            "left",
            "top",
            "width",
            "height",
            "clickable",
            "component",
            "icon",
            "text_button",
        )
    }
    ids = torch.randint(0, 4, (2, 3, 8), generator=torch.Generator().manual_seed(0))
    mask = torch.ones(2, 3, dtype=torch.bool)
    scores = reconstruction_scores(ids, mask, ids, mask, sizes)
    assert torch.equal(scores["total"], torch.ones(2))
    assert len(scores) == 9


def test_component_grid_skips_empty_spans_and_clips():
    elements = torch.tensor(
        [
            [0, 0, 0, 2, 0, 5, 0, 0],
            [1, 1, 9, 9, 0, 3, 0, 0],
        ]
    )
    grid = component_grid(elements, grid_size=4, background_id=1)
    assert grid.tolist() == [[1, 1, 1, 1], [1, 3, 3, 3], [1, 3, 3, 3], [1, 3, 3, 3]]


def test_layout_scores_perfect_and_partial():
    target = torch.tensor([[[0, 0, 1, 1, 0, 2, 0, 0]]])
    mask = torch.tensor([[True]])
    perfect = layout_scores(
        target, mask, target, mask, grid_size=4, num_labels=3, background_id=1
    )
    assert perfect["layout_acc"].tolist() == [1.0]
    torch.testing.assert_close(perfect["layout_miou"], torch.tensor([1.0]))
    shifted = target.clone()
    shifted[..., 0] = 2
    partial = layout_scores(
        target, mask, shifted, mask, grid_size=4, num_labels=3, background_id=1
    )
    assert partial["layout_acc"].item() == 8 / 16


def test_histograms_and_scores():
    sizes = {
        name: 3
        for name in (
            "left",
            "top",
            "width",
            "height",
            "clickable",
            "component",
            "icon",
            "text_button",
        )
    }
    ids = torch.tensor([[[1] * 8, [2] * 8], [[0] * 8, [9] * 8]])
    hist = field_histograms(torch.tensor([2, 1]), ids, sizes, max_length=3)
    assert hist["length"].tolist() == [0.5, 0.5, 0.0]
    torch.testing.assert_close(hist["left"], torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    scores = histogram_scores(hist, hist)
    assert scores["total"] == 1.0 and len(scores) == 10
