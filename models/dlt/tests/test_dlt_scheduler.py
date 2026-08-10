import tempfile

import torch

from dlt import DLTJointDiffusionScheduler


def test_mapping_cont2disc_matches_original_boundaries() -> None:
    mapping = DLTJointDiffusionScheduler.mapping_cont2disc(100, 10)
    assert mapping[0] == 0
    assert mapping[9] == 0
    assert mapping[10] == 1
    assert mapping[99] == 9


def test_transition_matrices_and_round_trip() -> None:
    scheduler = DLTJointDiffusionScheduler(
        discrete_features_names=[("cat", 7)],
        num_discrete_steps=[3],
        num_train_timesteps=6,
    )
    matrices = scheduler.transition_matrices["cat"]
    assert len(matrices) == 3
    assert matrices[0].shape == (7, 7)
    assert torch.allclose(matrices[0][-1], torch.tensor([0, 0, 0, 0, 0, 0, 1.0]))
    with tempfile.TemporaryDirectory() as tmp:
        scheduler.save_pretrained(tmp)
        loaded = DLTJointDiffusionScheduler.from_pretrained(tmp)
    assert loaded.cont2disc["cat"] == scheduler.cont2disc["cat"]
    assert torch.allclose(
        loaded.transition_matrices["cat"][1], scheduler.transition_matrices["cat"][1]
    )


def test_joint_step_shapes() -> None:
    scheduler = DLTJointDiffusionScheduler(
        num_train_timesteps=4, num_discrete_steps=[2]
    )
    sample = torch.randn(2, 3, 4)
    box_out = torch.randn_like(sample)
    cat_out = {"cat": torch.randn(2, 3, 7)}
    out, cat = scheduler.step_jointly(
        box_out,
        cat_out,
        torch.tensor([1, 1]),
        sample,
        generator=torch.Generator().manual_seed(0),
    )
    assert out.prev_sample.shape == sample.shape
    assert cat["cat"].shape == (2, 3)
