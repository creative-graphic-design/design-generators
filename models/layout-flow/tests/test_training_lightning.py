from typing import cast

import pytest
import torch

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")

from layout_flow import LayoutFlowConfig
from layout_flow.training.cli import LayoutFlowLightningCLI, main
from layout_flow.training.config import LayoutFlowSeedMode
from layout_flow.training.lightning_module import LayoutFlowTrainingModule
from layout_flow.training.parity import (
    compare_layout_flow_optimizer_step,
    compare_layout_flow_step,
    trace_layout_flow_step,
)
from layout_flow.training.seed import apply_layout_flow_seed_mode
from traingen_parity.trace import build_step_trace


pytestmark = pytest.mark.training


def tiny_batch() -> dict[str, torch.Tensor]:
    bbox = torch.rand(4, 3, 4)
    labels = torch.tensor(
        [[1, 2, 3], [1, 0, 0], [2, 3, 0], [4, 5, 1]], dtype=torch.long
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    return {
        "bbox": bbox * mask.unsqueeze(-1),
        "type": labels * mask.long(),
        "mask": mask,
        "length": mask.sum(dim=1).long(),
    }


def test_random4_condition_mask_reproduces_vendor_quarters() -> None:
    module = LayoutFlowTrainingModule(
        config=LayoutFlowConfig(
            dataset_name="publaynet",
            max_length=3,
            latent_dim=8,
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            num_layers=1,
        )
    )
    torch.manual_seed(0)
    cond = module.random4_condition_mask(torch.tensor([3, 2, 2, 3]), 3)
    assert cond.shape == (4, 3, 7)
    assert torch.equal(cond[1, :, 4:], torch.zeros(3, 3, dtype=torch.int))
    assert torch.equal(cond[2, :, 2:], torch.zeros(3, 5, dtype=torch.int))
    assert torch.equal(cond[3], torch.ones(3, 7, dtype=torch.int))


def test_training_step_records_required_trace_points() -> None:
    module = LayoutFlowTrainingModule(
        config=LayoutFlowConfig(
            dataset_name="publaynet",
            max_length=3,
            latent_dim=8,
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            num_layers=1,
        ),
        scheduler=None,
    )
    loss = module.training_step(tiny_batch(), 0)
    assert loss.ndim == 0
    for key in ["cond_mask", "x0", "x1", "t", "xt", "ut", "vt", "train_loss"]:
        assert key in module.latest_step_trace


def test_optimizer_scheduler_and_parity_helpers() -> None:
    module = LayoutFlowTrainingModule(
        config=LayoutFlowConfig(
            dataset_name="publaynet",
            max_length=3,
            latent_dim=8,
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            num_layers=1,
        )
    )
    optimizers = module.configure_optimizers()
    assert isinstance(optimizers, dict)
    optimizer_config = cast(dict[str, object], optimizers)
    lr_scheduler = cast(dict[str, object], optimizer_config["lr_scheduler"])
    assert lr_scheduler["monitor"] == "FID_Layout"
    torch.manual_seed(3)
    trace = trace_layout_flow_step(module, tiny_batch())
    assert compare_layout_flow_step(trace, trace).passed
    assert compare_layout_flow_optimizer_step(
        {"x": torch.ones(1)}, {"x": torch.ones(1)}
    ).passed
    empty_reference = build_step_trace("reference", {"x": torch.ones(1)})
    empty_target = build_step_trace("target", {"x": torch.zeros(1)})
    assert not compare_layout_flow_step(empty_reference, empty_target).passed


def test_seed_modes_and_package_lazy_exports() -> None:
    assert LayoutFlowSeedMode("default") is LayoutFlowSeedMode.default
    apply_layout_flow_seed_mode("default", seed=1)
    apply_layout_flow_seed_mode("deterministic", seed=1)


def test_lightning_cli_help_entrypoint() -> None:
    assert LayoutFlowLightningCLI.__name__ == "LayoutFlowLightningCLI"
    with pytest.raises(SystemExit):
        main(["--help"])
