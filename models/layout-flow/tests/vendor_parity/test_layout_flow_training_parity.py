from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import NamedTuple, Protocol, cast

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.common.vendor import vendor_root
from layout_flow import LayoutFlowConfig
from layout_flow.training.lightning_module import LayoutFlowTrainingModule
from layout_flow.training.parity import (
    compare_layout_flow_optimizer_step,
    compare_layout_flow_step,
)
from traingen_parity.trace import build_step_trace


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT = Path(__file__).resolve().parents[4]


class VendorTrainingModule(Protocol):
    """Structural type for the vendor LayoutFlow LightningModule."""

    model: torch.nn.Module

    def eval(self) -> VendorTrainingModule:
        """Switch to eval mode."""

    def get_cond_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the vendor conditioning mask."""

    def get_start_end(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return vendor x0 and x1 tensors."""

    def sample_t(self, x0: torch.Tensor) -> torch.Tensor:
        """Sample vendor timesteps."""

    def sample_xt(
        self,
        batch: dict[str, torch.Tensor],
        x0: torch.Tensor,
        x1: torch.Tensor,
        cond_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return vendor xt and ut tensors."""

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one vendor training step."""

    def __call__(
        self, xt: torch.Tensor, cond_mask: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Predict vendor vector fields."""


class TrainingParityFixture(NamedTuple):
    vendor: VendorTrainingModule
    target: LayoutFlowTrainingModule
    batch: dict[str, torch.Tensor]


def _vendor_classes() -> tuple[
    type[torch.nn.Module], type[torch.nn.Module], type[torch.nn.Module]
]:
    try:
        vendor_dir = vendor_root(
            "layout-flow",
            marker=Path("src/models/LayoutFlow.py"),
            repo_root=ROOT,
        )
    except FileNotFoundError as exc:
        skip_or_fail_vendor_parity(
            str(exc),
            missing_paths=[ROOT / "vendor" / "layout-flow"],
            regeneration_hint="run `git submodule update --init vendor/layout-flow`",
        )
    sys.path.insert(0, str(vendor_dir))
    from src.models.LayoutFlow import LayoutFlow as VendorLayoutFlow
    from src.models.backbone.layoutdm_backbone import LayoutDMBackbone as VendorBackbone
    from src.utils.distribution_sampler import DistributionSampler

    return VendorLayoutFlow, VendorBackbone, DistributionSampler


def _build_fixture(device: torch.device) -> TrainingParityFixture:
    vendor_layout_flow, vendor_backbone, distribution_sampler = _vendor_classes()
    torch.manual_seed(123)
    vendor = cast(
        VendorTrainingModule,
        vendor_layout_flow(
            vendor_backbone(
                latent_dim=8,
                tr_enc_only=True,
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                num_layers=1,
                dropout=0.0,
                use_pos_enc=False,
                num_cat=6,
                attr_encoding="AnalogBit",
                seq_type="stacked",
            ),
            distribution_sampler(
                distribution="gaussian",
                sample_padding=False,
                out_dim=7,
            ),
            optimizer=functools.partial(torch.optim.AdamW, lr=0.0005),
            scheduler=None,
            loss_fcn="mse",
            fid_calc_every_n=0,
            num_cat=6,
            cond="random4",
            attr_encoding="AnalogBit",
            add_loss="geom_l1_loss",
            add_loss_weight=0.2,
        ).to(device),
    )
    config = LayoutFlowConfig(
        dataset_name="publaynet",
        max_length=3,
        latent_dim=8,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
        dropout=0.0,
    )
    target = LayoutFlowTrainingModule(config=config, scheduler=None).to(device)
    target.model.backbone.load_state_dict(vendor.model.state_dict(), strict=True)
    batch = {
        "bbox": torch.rand(4, 3, 4, device=device),
        "type": torch.tensor(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], device=device
        ),
        "mask": torch.ones(4, 3, 1, dtype=torch.bool, device=device),
        "length": torch.full((4,), 3, dtype=torch.long, device=device),
    }
    vendor.eval()
    target.eval()
    return TrainingParityFixture(vendor=vendor, target=target, batch=batch)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _vendor_trace(
    vendor: VendorTrainingModule, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    cond_mask = vendor.get_cond_mask(batch)
    x0, x1 = vendor.get_start_end(batch)
    timestep = vendor.sample_t(x0)
    xt, ut = vendor.sample_xt(batch, x0, x1, cond_mask, timestep)
    vt = vendor(xt, cond_mask, timestep.squeeze(-1))
    flow_loss = F.mse_loss(cond_mask * vt, cond_mask * ut)
    geom_l1_loss = F.l1_loss(
        cond_mask[..., :4] * vt[..., :4], cond_mask[..., :4] * ut[..., :4]
    )
    train_loss = flow_loss + 0.2 * geom_l1_loss
    return {
        "bbox": batch["bbox"].detach(),
        "type": batch["type"].detach(),
        "mask": batch["mask"].squeeze(-1).detach(),
        "length": batch["length"].detach(),
        "cond_mask": cond_mask.detach(),
        "x0": x0.detach(),
        "x1": x1.detach(),
        "t": timestep.detach(),
        "xt": xt.detach(),
        "ut": ut.detach(),
        "vt": vt.detach(),
        "flow_loss": flow_loss.detach(),
        "geom_l1_loss": geom_l1_loss.detach(),
        "train_loss": train_loss.detach(),
    }


def test_s0_training_static_state_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    report = compare_layout_flow_optimizer_step(
        fixture.vendor.model.state_dict(),
        fixture.target.model.backbone.state_dict(),
    )
    assert report.passed
    assert fixture.target.condition_policy == "random4"
    assert fixture.target.geom_l1_weight == 0.2


def test_s1_fixed_batch_pre_optimizer_trace_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    torch.manual_seed(999)
    vendor_trace = build_step_trace(
        "vendor", _vendor_trace(fixture.vendor, fixture.batch)
    )
    torch.manual_seed(999)
    target_loss = fixture.target.training_step(fixture.batch, 0)
    assert torch.equal(
        target_loss.detach(), fixture.target.latest_step_trace["train_loss"]
    )
    target_trace = build_step_trace("target", fixture.target.latest_step_trace)
    report = compare_layout_flow_step(vendor_trace, target_trace)
    assert report.passed, report


def test_s2_one_optimizer_step_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    vendor_optimizer = torch.optim.AdamW(
        fixture.vendor.model.parameters(), lr=0.0005, betas=(0.9, 0.98)
    )
    target_optimizer = torch.optim.AdamW(
        fixture.target.model.parameters(), lr=0.0005, betas=(0.9, 0.98)
    )
    vendor_optimizer.zero_grad()
    target_optimizer.zero_grad()
    torch.manual_seed(999)
    vendor_loss = fixture.vendor.training_step(fixture.batch, 0)
    vendor_loss.backward()
    vendor_optimizer.step()
    torch.manual_seed(999)
    target_loss = fixture.target.training_step(fixture.batch, 0)
    target_loss.backward()
    target_optimizer.step()
    report = compare_layout_flow_optimizer_step(
        fixture.vendor.model.state_dict(),
        fixture.target.model.backbone.state_dict(),
    )
    assert report.passed, report
