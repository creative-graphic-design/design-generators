import os
import sys
from pathlib import Path
from functools import partial
from typing import Any, cast  # noqa: TID251 - vendor modules are dynamic imports.

import pytest
import torch
from torch import nn

from cgb_dm.modeling_cgb_dm import CGBDMModelOutput
from cgb_dm.training import CGBDMTrainingModule
from cgb_dm.training.parity import (
    CGBDMStepTraceAdapter,
    build_reference_encoded_dataset,
)


@pytest.mark.vendor_parity
def test_s0_s2_adapter_is_gated():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    adapter = CGBDMStepTraceAdapter()
    batch = (
        torch.zeros(1, 4, 32, 32),
        torch.zeros(1, 2, 8),
        torch.zeros(1, 1, 4),
    )
    comparable = adapter.comparable_batch(batch)
    assert set(comparable) == {"pixel_values", "layout", "saliency_box"}


@pytest.mark.vendor_parity
def test_s0_real_vendor_loader_matches_manifest_replay():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    data_root = Path(
        os.environ.get("CGB_DM_DATA_ROOT", ".cache/cgb-dm/datasets/pku/split")
    )
    manifest = Path(
        os.environ.get(
            "CGB_DM_VENDOR_ORDER_MANIFEST",
            ".cache/cgb-dm/reference/pku_posterlayout_train_manifest.json",
        )
    )
    vendor_root = Path(os.environ.get("CGB_DM_VENDOR_ROOT", "vendor/layout-dit"))
    if not data_root.exists():
        raise AssertionError(f"CGB-DM data root is missing: {data_root}")
    if not manifest.exists():
        raise AssertionError(f"CGB-DM source order manifest is missing: {manifest}")
    if not vendor_root.exists():
        raise AssertionError(f"CGB-DM vendor root is missing: {vendor_root}")

    vendor_batch = _load_vendor_rows(vendor_root, data_root, count=3)
    replay = build_reference_encoded_dataset(data_root, manifest=manifest)
    assert len(replay) >= len(vendor_batch)
    for index, (image, layout, saliency_box) in enumerate(vendor_batch):
        row = replay[index]
        torch.testing.assert_close(row["pixel_values"], image, rtol=0.0, atol=0.0)
        torch.testing.assert_close(row["layout"], layout, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            row["saliency_box"], saliency_box, rtol=0.0, atol=0.0
        )


@pytest.mark.vendor_parity
@pytest.mark.training
def test_s1_real_vendor_training_trace_matches():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    vendor_root = Path(os.environ.get("CGB_DM_VENDOR_ROOT", "vendor/layout-dit"))
    data_root = Path(
        os.environ.get("CGB_DM_DATA_ROOT", ".cache/cgb-dm/datasets/pku/split")
    )
    if not vendor_root.exists():
        raise AssertionError(f"CGB-DM vendor root is missing: {vendor_root}")
    if not data_root.exists():
        raise AssertionError(f"CGB-DM data root is missing: {data_root}")

    _set_deterministic_cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image, layout, saliency_box = _load_vendor_rows(vendor_root, data_root, count=1)[0]
    batch = (
        image.unsqueeze(0).to(device),
        layout.unsqueeze(0).to(device),
        saliency_box.unsqueeze(0).to(device),
    )
    vendor_diffusion, ours_module = _paired_training_models(vendor_root, device)

    torch.manual_seed(23)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(23)
    vendor_trace = _run_vendor_trace(vendor_diffusion, batch, cond="uncond")

    torch.manual_seed(23)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(23)
    image, layout, saliency_box = batch
    ours_module.training_step(
        {
            "pixel_values": image.clone(),
            "layout": layout.clone(),
            "saliency_box": saliency_box.clone(),
        },
        0,
    )
    ours = ours_module.latest_step_trace

    for key in (
        "t",
        "noise",
        "fix_mask",
        "noisy_layout",
        "predicted_epsilon",
        "cgb_weight",
        "loss",
    ):
        _assert_trace_close(key, ours[key], vendor_trace[key])


@pytest.mark.vendor_parity
@pytest.mark.training
def test_s2_real_vendor_optimizer_step_matches():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    vendor_root = Path(os.environ.get("CGB_DM_VENDOR_ROOT", "vendor/layout-dit"))
    data_root = Path(
        os.environ.get("CGB_DM_DATA_ROOT", ".cache/cgb-dm/datasets/pku/split")
    )
    if not vendor_root.exists():
        raise AssertionError(f"CGB-DM vendor root is missing: {vendor_root}")
    if not data_root.exists():
        raise AssertionError(f"CGB-DM data root is missing: {data_root}")

    _set_deterministic_cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image, layout, saliency_box = _load_vendor_rows(vendor_root, data_root, count=1)[0]
    batch = (
        image.unsqueeze(0).to(device),
        layout.unsqueeze(0).to(device),
        saliency_box.unsqueeze(0).to(device),
    )
    vendor_diffusion, ours_module = _paired_training_models(vendor_root, device)

    vendor_optimizer = torch.optim.Adam(
        vendor_diffusion.model.parameters(),
        lr=1.0e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        amsgrad=False,
        eps=1.0e-8,
    )
    ours_optimizer = torch.optim.Adam(
        ours_module.model.parameters(),
        lr=1.0e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        amsgrad=False,
        eps=1.0e-8,
    )

    vendor_optimizer.zero_grad()
    torch.manual_seed(29)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(29)
    vendor_trace = _run_vendor_trace(vendor_diffusion, batch, cond="uncond")
    vendor_trace["loss"].reshape(()).backward()
    torch.nn.utils.clip_grad_norm_(vendor_diffusion.model.parameters(), 1.0)

    ours_optimizer.zero_grad()
    torch.manual_seed(29)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(29)
    image, layout, saliency_box = batch
    ours_loss = ours_module.training_step(
        {
            "pixel_values": image.clone(),
            "layout": layout.clone(),
            "saliency_box": saliency_box.clone(),
        },
        0,
    )
    ours_loss.backward()
    torch.nn.utils.clip_grad_norm_(ours_module.model.parameters(), 1.0)

    for vendor_param, ours_param in zip(
        vendor_diffusion.model.parameters(),
        ours_module.model.parameters(),
        strict=True,
    ):
        assert (vendor_param.grad is None) == (ours_param.grad is None)
        if vendor_param.grad is None:
            continue
        torch.testing.assert_close(
            ours_param.grad, vendor_param.grad, rtol=1e-5, atol=1e-9
        )

    vendor_optimizer.step()
    ours_optimizer.step()

    for vendor_param, ours_param in zip(
        vendor_diffusion.model.parameters(),
        ours_module.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(ours_param, vendor_param, rtol=2e-3, atol=5e-7)
        assert (vendor_param in vendor_optimizer.state) == (
            ours_param in ours_optimizer.state
        )
        if vendor_param in vendor_optimizer.state:
            _assert_adam_state_close(
                ours_optimizer.state[ours_param], vendor_optimizer.state[vendor_param]
            )


def _load_vendor_rows(
    vendor_root: Path, data_root: Path, *, count: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    vendor_root = vendor_root.resolve()
    data_root = data_root.resolve()
    sys.path.insert(0, str(vendor_root))
    cwd = Path.cwd()
    try:
        os.chdir(vendor_root)
        from data_process.dataloader import train_dataset
        from utils.util import load_config

        cfg = load_config("configs/pku.yaml")
        cfg.paths.train.inp_dir = str(data_root / "train/inpaint")
        cfg.paths.train.sal_dir = str(data_root / "train/saliency")
        cfg.paths.train.sal_sub_dir = str(data_root / "train/saliency_sub")
        cfg.paths.train.annotated_dir = str(data_root / "csv/train.csv")
        cfg.paths.train.salbox_dir = str(data_root / "csv/train_sal.csv")
        dataset = train_dataset(cfg)
        return [dataset[index] for index in range(min(count, len(dataset)))]
    finally:
        os.chdir(cwd)
        sys.path = [entry for entry in sys.path if entry != str(vendor_root)]


def _set_deterministic_cuda() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _assert_trace_close(key: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if key in {"t", "fix_mask"}:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


class _VendorOutputAdapter(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        self.latest_cgb_weight: torch.Tensor | None = None
        self.model.cgbwp.register_forward_hook(self._record_cgb_weight)

    def _record_cgb_weight(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self.latest_cgb_weight = output

    def forward(
        self,
        sample: torch.Tensor,
        image: torch.Tensor,
        saliency_box: torch.Tensor,
        timestep: torch.Tensor,
    ) -> CGBDMModelOutput:
        sample = self.model(sample, image, saliency_box, timestep)
        return CGBDMModelOutput(sample=sample, cgb_weight=self.latest_cgb_weight)


def _paired_training_models(
    vendor_root: Path, device: torch.device
) -> tuple[Any, CGBDMTrainingModule]:
    sys.path.insert(0, str(vendor_root.resolve()))
    try:
        from cgbdm.diffusion import Diffusion

        torch.manual_seed(11)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(11)
        vendor_diffusion = Diffusion(
            num_timesteps=1000,
            ddim_num_steps=100,
            n_head=8,
            dim_model=512,
            feature_dim=1024,
            seq_dim=8,
            num_layers=4,
            device=str(device),
            max_elem=16,
        )
        torch.manual_seed(11)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(11)
        ours_diffusion = Diffusion(
            num_timesteps=1000,
            ddim_num_steps=100,
            n_head=8,
            dim_model=512,
            feature_dim=1024,
            seq_dim=8,
            num_layers=4,
            device=str(device),
            max_elem=16,
        )
    finally:
        sys.path = [entry for entry in sys.path if entry != str(vendor_root.resolve())]

    ours_module = CGBDMTrainingModule(
        config={
            "num_labels": 4,
            "max_seq_length": 16,
            "image_size": (384, 256),
            "dim_model": 512,
            "n_head": 8,
            "feature_dim": 1024,
            "num_layers": 4,
            "num_train_timesteps": 1000,
            "ddim_num_steps": 100,
        },
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        model=cast(Any, _VendorOutputAdapter(ours_diffusion.model)),
        condition_type="content_image",
    ).to(device)
    vendor_diffusion.model.train()
    ours_module.train()
    return vendor_diffusion, ours_module


def _run_vendor_trace(
    diffusion_model: Any,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    cond: str,
) -> dict[str, torch.Tensor]:
    image, layout, saliency_box = batch
    cgb_weight: torch.Tensor | None = None

    def record_cgb_weight(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        nonlocal cgb_weight
        cgb_weight = output

    hook = diffusion_model.model.cgbwp.register_forward_hook(record_cgb_weight)
    try:
        timesteps = diffusion_model.sample_t(
            [layout.shape[0]], t_max=diffusion_model.num_timesteps - 1
        )
        predicted, noise = diffusion_model.forward_t(
            layout, image, saliency_box, t=timesteps, cond=cond
        )
    finally:
        hook.remove()
    fix_mask = torch.zeros_like(layout, dtype=torch.bool)
    noisy = diffusion_model.alphas_bar_sqrt[timesteps].reshape(-1, 1, 1) * layout
    noisy = (
        noisy
        + diffusion_model.one_minus_alphas_bar_sqrt[timesteps].reshape(-1, 1, 1) * noise
    )
    loss = torch.nn.functional.mse_loss(noise, predicted).reshape(1)
    assert cgb_weight is not None
    return {
        "t": timesteps.detach(),
        "noise": noise.detach(),
        "fix_mask": fix_mask.detach(),
        "noisy_layout": noisy.detach(),
        "predicted_epsilon": predicted.detach(),
        "cgb_weight": cgb_weight.detach(),
        "loss": loss,
    }


def _assert_adam_state_close(
    ours_state: dict[str, torch.Tensor], vendor_state: dict[str, torch.Tensor]
) -> None:
    assert set(ours_state) == set(vendor_state)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            ours_state[key], vendor_state[key], rtol=2e-3, atol=5e-7
        )
