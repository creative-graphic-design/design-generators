from typing import cast

import torch
import pytest

from layout_flow.configuration_layout_flow import SeqType
from layout_flow import LayoutFlowTransformerModel
from layout_flow.modeling_layout_flow import (
    LayoutDMBackbone,
    LayoutFlowBlock,
    _get_activation_fn,
)


def test_model_output_shape_for_tiny_config() -> None:
    model = LayoutFlowTransformerModel(
        num_labels=6,
        latent_dim=8,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
    )
    sample = torch.randn(2, 3, 7)
    cond_mask = torch.ones(2, 3, 7, dtype=torch.long)
    out = model(sample=sample, timestep=torch.zeros(2), cond_mask=cond_mask)
    assert out.sample.shape == sample.shape
    tuple_out = model(
        sample=sample,
        timestep=torch.tensor(0.0),
        cond_mask=cond_mask,
        return_dict=False,
    )
    assert tuple_out[0].shape == sample.shape


def test_backbone_variants_and_activation_errors() -> None:
    assert _get_activation_fn("gelu2")(torch.ones(1)).shape == (1,)
    with pytest.raises(ValueError, match="Unsupported activation"):
        _get_activation_fn("bad")
    with pytest.raises(ValueError, match="prenorm"):
        LayoutFlowBlock(d_model=8, nhead=2, norm_first=False)

    geom = torch.rand(1, 2, 4)
    cond = torch.ones(1, 2, 7, dtype=torch.long)
    timestep = torch.zeros(1)
    for seq_type in ["seq_cond", "seq"]:
        model = LayoutDMBackbone(
            latent_dim=4,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            num_layers=1,
            num_cat=6,
            attr_encoding="AnalogBit",
            seq_type=seq_type,
            use_pos_enc=True,
        )
        attr = torch.rand(1, 2, 3)
        assert model(geom, attr, cond, timestep).shape == (1, 2, 7)

    discrete = LayoutDMBackbone(
        latent_dim=4,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_layers=1,
        num_cat=6,
        attr_encoding="discrete",
        seq_type="stacked",
        tr_enc_only=False,
    )
    geom_discrete = torch.rand(2, 2, 4)
    cond_discrete = torch.ones(2, 2, 5, dtype=torch.long)
    timestep_discrete = torch.zeros(2)
    assert discrete(
        geom_discrete,
        torch.ones(2, 2, 1).long(),
        cond_discrete,
        timestep_discrete,
    ).shape == (2, 2, 5)
    bad = LayoutDMBackbone(
        latent_dim=4,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_layers=1,
        num_cat=6,
        seq_type=cast(SeqType, "bad"),
    )
    with pytest.raises(ValueError, match="Unsupported seq_type"):
        bad(geom, torch.rand(1, 2, 1), cond, timestep)
