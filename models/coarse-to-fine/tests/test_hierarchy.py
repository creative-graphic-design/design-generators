import torch

from coarse_to_fine.hierarchy import build_cut_hierarchy


def test_build_cut_hierarchy_groups_two_row_layout():
    bbox = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.3, 0.0, 0.2, 0.2],
            [0.0, 0.5, 0.2, 0.2],
        ]
    )
    labels = torch.tensor([1, 2, 1])

    enc = build_cut_hierarchy(
        bbox,
        labels,
        num_labels=2,
        discrete_x_grid=128,
        discrete_y_grid=128,
    )

    assert enc.group_bounding_box.shape == (2, 4)
    assert [item.tolist() for item in enc.grouped_labels] == [[1, 2], [1]]
    torch.testing.assert_close(
        enc.label_in_one_group, torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    )
