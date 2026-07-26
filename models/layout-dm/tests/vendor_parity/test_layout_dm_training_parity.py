from pathlib import Path

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.training.datamodule import LayoutDMDataModule
from layout_dm.training.lightning_module import LayoutDMTrainingModule
from layout_dm.training.parity import trace_layout_dm_step

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def _vendor_root() -> Path:
    return Path(__file__).parents[4] / "vendor" / "layout-dm"


def _require_vendor_assets() -> None:
    root = _vendor_root()
    required = [
        root / "src" / "trainer" / "trainer" / "main.py",
        root / "src" / "trainer" / "trainer" / "models" / "layoutdm.py",
        root
        / "src"
        / "trainer"
        / "trainer"
        / "models"
        / "categorical_diffusion"
        / "vanilla.py",
    ]
    if not all(path.exists() for path in required):
        skip_or_fail_vendor_parity(
            "LayoutDM training parity requires the vendor submodule and local assets",
            missing_paths=required,
            regeneration_hint="git submodule update --init vendor/layout-dm",
        )


def _config() -> LayoutDMConfig:
    return LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=4,
        num_bin_bboxes=8,
        bbox_quantization="linear",
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        num_timesteps=4,
    )


def _batch() -> dict[str, torch.Tensor]:
    datamodule = LayoutDMDataModule(
        dataset_name="publaynet",
        config=_config(),
        batch_size=2,
        num_workers=0,
        synthetic_size=2,
    )
    datamodule.setup("fit")
    return next(iter(datamodule.train_dataloader()))


def test_s0_static_training_state_topology_guard() -> None:
    _require_vendor_assets()
    module = LayoutDMTrainingModule(config=_config())
    param_count = sum(parameter.numel() for parameter in module.model.parameters())
    assert param_count > 0
    assert module.lt_history.shape == (module.num_timesteps,)
    assert module.lt_count.shape == (module.num_timesteps,)
    assert all(key.startswith("transformer.") for key in module.model.state_dict())


def test_s1_package_fixed_batch_trace_points_are_available() -> None:
    _require_vendor_assets()
    module = LayoutDMTrainingModule(config=_config(), time_sampler="uniform")
    torch.manual_seed(42975)
    trace = trace_layout_dm_step(module, _batch())
    for key in ("t", "xt", "log_model_prob", "kl", "kl_loss", "train_loss"):
        assert key in trace.tensors
        assert torch.isfinite(trace.tensors[key].float()).all()


def test_s2_one_optimizer_step_produces_gradients() -> None:
    _require_vendor_assets()
    module = LayoutDMTrainingModule(
        config=_config(), scheduler=None, time_sampler="uniform"
    )
    optimizer = torch.optim.AdamW(module.optim_groups(), lr=1e-4)
    loss = module.training_step(_batch(), 0)
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    optimizer.step()
    optimizer.zero_grad()
