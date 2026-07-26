from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Protocol, cast

import pytest
import torch

pytest.importorskip("lightning")
pytest.importorskip("traingen_parity")

from laygen.common.discrete import index_to_log_onehot, log_onehot_to_index
from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.common.vendor import vendor_root
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer
from layout_dm.training.dataset import LayoutDMDataset
from layout_dm.training.lightning_module import LayoutDMTrainingModule
from layout_dm.training.parity import (
    compare_layout_dm_optimizer_step,
    compare_layout_dm_step,
)
from traingen_parity.trace import build_step_trace

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT = Path(__file__).resolve().parents[4]


class VendorTokenizer(Protocol):
    N_total: int
    max_token_length: int

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode structured layout tensors into a flattened token sequence."""


class VendorDiffusion(Protocol):
    transformer: torch.nn.Module
    num_classes: int
    num_timesteps: int
    Lt_history: torch.Tensor
    Lt_count: torch.Tensor
    auxiliary_loss_weight: float
    adaptive_auxiliary_loss: bool
    mask_weight: list[float]

    def sample_time(
        self, b: int, device: torch.device, method: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample diffusion timesteps and probabilities."""

    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample the noised sequence."""

    def predict_start(self, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the clean sequence log probabilities."""

    def q_posterior(
        self, log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute posterior transition log probabilities."""

    def multinomial_kl(
        self, log_prob1: torch.Tensor, log_prob2: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence over categorical log probabilities."""


class VendorLayoutDM(Protocol):
    model: torch.nn.Module
    tokenizer: VendorTokenizer

    def eval(self) -> VendorLayoutDM:
        """Switch to evaluation mode."""

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> object:
        """Run one training step if implemented by the source framework."""


class TrainingParityFixture(NamedTuple):
    vendor: VendorLayoutDM
    vendor_model: VendorDiffusion
    target: LayoutDMTrainingModule
    batch: dict[str, torch.Tensor]
    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor


def _vendor_classes() -> tuple[type[torch.nn.Module], type[object]]:
    try:
        vendor_dir = vendor_root(
            "layout-dm",
            marker=Path("src/trainer/trainer/models/layoutdm.py"),
            repo_root=ROOT,
        )
    except FileNotFoundError as exc:
        skip_or_fail_vendor_parity(
            str(exc),
            missing_paths=[ROOT / "vendor" / "layout-dm"],
            regeneration_hint="run `git submodule update --init vendor/layout-dm`",
        )
    sys.path.insert(0, str(vendor_dir / "src" / "trainer"))
    from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from trainer.models.layoutdm import LayoutDM

    return LayoutDM, LayoutSequenceTokenizer


def _config() -> LayoutDMConfig:
    return LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=4,
        num_bin_bboxes=8,
        bbox_quantization="linear",
        hidden_size=29,
        num_attention_heads=1,
        num_hidden_layers=1,
        intermediate_size=29,
        num_timesteps=4,
        q_type="vanilla",
    )


def _vendor_tokenizer(vendor_tokenizer_cls: type[object]) -> VendorTokenizer:
    omegaconf = pytest.importorskip("omegaconf")
    data_cfg = omegaconf.OmegaConf.create(
        {
            "pad_until_max": True,
            "shared_bbox_vocab": "x-y-w-h",
            "bbox_quantization": "linear",
            "var_order": "c-x-y-w-h",
            "special_tokens": ["pad", "mask"],
            "num_bin_bboxes": 8,
        }
    )
    dataset_cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "trainer.datasets.publaynet.PubLayNetDataset",
            "max_seq_length": 4,
        }
    )
    return cast(VendorTokenizer, vendor_tokenizer_cls(data_cfg, dataset_cfg))


def _build_fixture(device: torch.device) -> TrainingParityFixture:
    omegaconf = pytest.importorskip("omegaconf")
    vendor_layout_dm_cls, vendor_tokenizer_cls = _vendor_classes()
    vendor_tokenizer = _vendor_tokenizer(vendor_tokenizer_cls)
    backbone_cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "trainer.models.transformer_utils.TransformerEncoder",
            "encoder_layer": {
                "_target_": "trainer.models.transformer_utils.Block",
                "d_model": 32,
                "nhead": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "batch_first": True,
                "norm_first": True,
                "diffusion_step": 100,
                "timestep_type": "adalayernorm",
            },
            "num_layers": 1,
        }
    )
    torch.manual_seed(123)
    vendor = cast(
        VendorLayoutDM,
        vendor_layout_dm_cls(
            backbone_cfg=backbone_cfg,
            tokenizer=vendor_tokenizer,
            transformer_type="flattened",
            pos_emb="elem_attr",
            num_timesteps=4,
            auxiliary_loss_weight=0.1,
            q_type="vanilla",
            seq_type="poset",
        ).to(device),
    )
    cfg = _config()
    target = LayoutDMTrainingModule(config=cfg, time_sampler="uniform", scheduler=None)
    target.to(device)
    vendor_model = cast(VendorDiffusion, vendor.model.module)
    target.model.transformer.load_state_dict(
        vendor_model.transformer.state_dict(), strict=True
    )
    cast(torch.Tensor, target.lt_history).copy_(vendor_model.Lt_history)
    cast(torch.Tensor, target.lt_count).copy_(vendor_model.Lt_count)
    bbox = torch.tensor(
        [
            [
                [0.125, 0.25, 0.375, 0.5],
                [0.5, 0.625, 0.25, 0.125],
                [0.875, 0.125, 0.25, 0.375],
            ],
            [
                [0.25, 0.125, 0.5, 0.25],
                [0.625, 0.75, 0.125, 0.125],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor([[1, 2, 3], [2, 3, 0]], dtype=torch.long, device=device)
    mask = torch.tensor(
        [[True, True, True], [True, True, False]], dtype=torch.bool, device=device
    )
    batch = {
        "input_ids": vendor_tokenizer.encode(
            {"bbox": bbox.cpu(), "label": labels.cpu(), "mask": mask.cpu()}
        )["seq"].to(device),
    }
    vendor.eval()
    target.eval()
    return TrainingParityFixture(
        vendor=vendor,
        vendor_model=vendor_model,
        target=target,
        batch=batch,
        bbox=bbox,
        labels=labels,
        mask=mask,
    )


def _device() -> torch.device:
    return torch.device("cpu")


def _mean_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1)


def _log_categorical(log_x_start: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    return (log_x_start.exp() * log_prob).sum(dim=1)


def _vendor_trace(
    vendor: VendorDiffusion, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    x_start = batch["input_ids"].long()
    batch_size = x_start.size(0)
    t, pt = vendor.sample_time(batch_size, x_start.device, "importance")
    log_x_start = index_to_log_onehot(x_start, vendor.num_classes)
    log_xt = vendor.q_sample(log_x_start=log_x_start, t=t)
    xt = log_onehot_to_index(log_xt)
    log_x0_recon = vendor.predict_start(log_xt, t=t)
    log_model_prob = vendor.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)
    log_true_prob = vendor.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
    kl = vendor.multinomial_kl(log_true_prob, log_model_prob)
    mask_region = (xt == vendor.num_classes - 1).float()
    mask_weight = (
        mask_region * vendor.mask_weight[0]
        + (1.0 - mask_region) * vendor.mask_weight[1]
    )
    kl = _mean_except_batch(kl * mask_weight)
    decoder_nll = _mean_except_batch(-_log_categorical(log_x_start, log_model_prob))
    at_zero = (t == 0).float()
    kl_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl
    lt2 = kl_loss.pow(2)
    lt2_prev = vendor.Lt_history.gather(dim=0, index=t)
    vendor.Lt_history.scatter_(
        dim=0, index=t, src=(0.1 * lt2 + 0.9 * lt2_prev).detach()
    )
    vendor.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(lt2))
    losses = {"kl_loss": (kl_loss / pt).mean()}
    if vendor.auxiliary_loss_weight != 0:
        kl_aux = vendor.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
        kl_aux = _mean_except_batch(kl_aux * mask_weight)
        kl_aux_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl_aux
        weight = (1 - t / vendor.num_timesteps) + 1.0
        if not vendor.adaptive_auxiliary_loss:
            weight = torch.ones_like(weight)
        losses["aux_loss"] = (
            weight * vendor.auxiliary_loss_weight * kl_aux_loss / pt
        ).mean()
    train_loss = torch.stack(tuple(losses.values())).sum()
    return {
        "t": t.detach(),
        "pt": pt.detach(),
        "xt": xt.detach(),
        "log_model_prob": log_model_prob.detach(),
        "kl": kl.detach(),
        "decoder_nll": decoder_nll.detach(),
        "kl_loss": kl_loss.detach(),
        **{key: value.detach() for key, value in losses.items()},
        "train_loss": train_loss.detach(),
    }


def test_s0_training_static_state_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    report = compare_layout_dm_optimizer_step(
        fixture.vendor_model.transformer.state_dict(),
        fixture.target.model.transformer.state_dict(),
    )
    assert report.passed, report
    assert fixture.vendor_model.num_classes == fixture.target.num_classes
    assert fixture.vendor_model.num_timesteps == fixture.target.num_timesteps
    assert fixture.vendor.tokenizer.N_total == fixture.target.tokenizer.vocab_size
    assert (
        fixture.vendor.tokenizer.max_token_length
        == fixture.target.tokenizer.model_max_length
    )


def test_s1_fixed_batch_pre_optimizer_trace_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    torch.manual_seed(999)
    vendor_trace = build_step_trace(
        "vendor", _vendor_trace(fixture.vendor_model, fixture.batch)
    )
    torch.manual_seed(999)
    target_loss = fixture.target.training_step(fixture.batch, 0)
    assert torch.equal(
        target_loss.detach(), fixture.target.latest_step_trace["train_loss"]
    )
    target_trace = build_step_trace("target", fixture.target.latest_step_trace)
    report = compare_layout_dm_step(vendor_trace, target_trace)
    assert report.passed, report


def test_s2_one_optimizer_step_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    vendor_optimizer = torch.optim.AdamW(
        fixture.vendor_model.transformer.parameters(), lr=5e-4, betas=(0.9, 0.98)
    )
    target_optimizer = torch.optim.AdamW(
        fixture.target.model.transformer.parameters(), lr=5e-4, betas=(0.9, 0.98)
    )
    vendor_optimizer.zero_grad()
    target_optimizer.zero_grad()
    torch.manual_seed(999)
    vendor_loss = torch.stack(
        tuple(fixture.vendor.model(fixture.batch["input_ids"])[1].values())
    ).sum()
    vendor_loss.backward()
    vendor_optimizer.step()
    torch.manual_seed(999)
    target_loss = fixture.target.training_step(fixture.batch, 0)
    target_loss.backward()
    target_optimizer.step()
    report = compare_layout_dm_optimizer_step(
        fixture.vendor_model.transformer.state_dict(),
        fixture.target.model.transformer.state_dict(),
    )
    assert report.passed, report


def test_s4_loader_tokenizer_output_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    vendor_encoded = fixture.vendor.tokenizer.encode(
        {
            "bbox": fixture.bbox.cpu(),
            "label": fixture.labels.cpu(),
            "mask": fixture.mask.cpu(),
        }
    )
    processor = LayoutDMProcessor(LayoutDMTokenizer(_config()))
    target_encoded = processor(
        bbox=fixture.bbox.cpu(),
        labels=fixture.labels.cpu(),
        mask=fixture.mask.cpu(),
    )
    assert torch.equal(vendor_encoded["seq"], target_encoded["input_ids"])
    assert torch.equal(vendor_encoded["mask"], target_encoded["mask"])
    assert torch.equal(vendor_encoded["mask"], target_encoded["attention_mask"])

    dataset = LayoutDMDataset.__new__(LayoutDMDataset)
    dataset.dataset_name = "publaynet"
    dataset.split = "train"
    dataset.config = _config()
    dataset.tokenizer = LayoutDMTokenizer(dataset.config)
    dataset.processor = LayoutDMProcessor(dataset.tokenizer)
    dataset.max_seq_length = dataset.config.max_seq_length
    dataset.box_format = "xywh"
    dataset.normalized = True
    dataset.label2id = {}
    loader_encoded = dataset._encode_sample(
        {
            "id": "layout-dm-s4-fixture",
            "bbox": fixture.bbox[0].cpu().tolist(),
            "labels": fixture.labels[0].cpu().tolist(),
        }
    )
    assert torch.equal(loader_encoded["input_ids"], vendor_encoded["seq"][0])
    assert torch.equal(loader_encoded["mask"], vendor_encoded["mask"][0])
