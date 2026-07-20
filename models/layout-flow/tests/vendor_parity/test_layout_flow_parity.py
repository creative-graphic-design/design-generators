from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from laygen.common.testing import (
    load_torch_checkpoint_state_dict,
    strip_torch_state_dict_prefix,
    vendor_backbone_kwargs,
)
from laygen.common.vendor import vendor_root
from layout_flow import LayoutFlowConfig, LayoutFlowTransformerModel
from layout_flow.conversion import convert_lightning_state_dict


ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_DIR = ROOT / ".cache" / "layout-flow" / "original" / "checkpoints"


def _load_vendor_backbone():
    vendor_dir = vendor_root(
        "layout-flow",
        marker=Path("src/models/backbone/layoutdm_backbone.py"),
        repo_root=ROOT,
    )
    sys.path.insert(0, str(vendor_dir))
    from src.models.backbone.layoutdm_backbone import LayoutDMBackbone

    return LayoutDMBackbone


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("dataset", "checkpoint_name"),
    [
        ("publaynet", "checkpoint_PubLayNet_LayoutFlow.ckpt"),
        ("rico25", "checkpoint_RICO_LayoutFlow.ckpt"),
    ],
)
def test_converted_vector_field_matches_vendor(
    dataset: str, checkpoint_name: str
) -> None:
    checkpoint = CHECKPOINT_DIR / checkpoint_name
    if not checkpoint.exists():
        pytest.skip(f"missing checkpoint: {checkpoint}")
    LayoutDMBackbone = _load_vendor_backbone()
    config = LayoutFlowConfig(dataset_name=dataset)
    state_dict = load_torch_checkpoint_state_dict(
        checkpoint,
        state_dict_key="state_dict",
        map_location="cpu",
        weights_only=False,
    )
    vendor = LayoutDMBackbone(
        **vendor_backbone_kwargs(
            config,
            (
                "latent_dim",
                "tr_enc_only",
                "d_model",
                "nhead",
                "dim_feedforward",
                "num_layers",
                "dropout",
                "use_pos_enc",
                "num_cat",
                "attr_encoding",
                "seq_type",
            ),
            aliases={"num_cat": "num_labels"},
        )
    )
    vendor.load_state_dict(
        strip_torch_state_dict_prefix(
            state_dict,
            strip_prefix="model.",
            include_prefix="model.",
        )
    )
    converted = LayoutFlowTransformerModel(
        num_labels=config.num_labels,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    converted.load_state_dict(convert_lightning_state_dict(state_dict))
    vendor.eval()
    converted.eval()
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(
        2, 4, config.sample_dim, generator=generator, dtype=torch.float32
    )
    cond_mask = torch.randint(
        0, 2, (2, 4, config.sample_dim), generator=generator, dtype=torch.long
    )
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32)
    with torch.no_grad():
        expected = vendor(sample[:, :, :4], sample[:, :, 4:], cond_mask, timestep)
        actual = converted(sample=sample, timestep=timestep, cond_mask=cond_mask).sample
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.abs().clamp_min(1e-8)).max().item()
    assert max_abs == pytest.approx(0.0, abs=1e-6)
    assert max_rel == pytest.approx(0.0, abs=1e-5)
