import tempfile
from typing import cast

import pytest
import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from dlt import DLTConfig, DLTPipeline, build_pipeline
from dlt.pipeline_dlt import normalize_condition_type


def tiny_pipeline() -> DLTPipeline:
    config = DLTConfig(
        dataset_name="publaynet",
        max_num_comp=3,
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
        num_cont_timesteps=3,
        num_discrete_steps=2,
    )
    return build_pipeline(config)


def test_pipeline_unconditional_seed_reproducible() -> None:
    pipe = tiny_pipeline()
    out1 = cast(
        LayoutGenerationOutput,
        pipe(batch_size=1, num_elements=2, seed=0, num_inference_steps=2),
    )
    out2 = cast(
        LayoutGenerationOutput,
        pipe(batch_size=1, num_elements=2, seed=0, num_inference_steps=2),
    )
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.equal(out1.labels, out2.labels)
    assert torch.allclose(out1.bbox, out2.bbox)


def test_pipeline_public_condition_modes() -> None:
    pipe = tiny_pipeline()
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    labels = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, True]])
    for condition_type in ["label", "label_size", "whole_box", "loc"]:
        out = cast(
            LayoutGenerationOutput,
            pipe(
                condition_type=condition_type,
                bbox=bbox,
                labels=labels,
                mask=mask,
                seed=1,
                num_inference_steps=2,
                output_type="dataclass",
            ),
        )
        assert_layout_output_schema(out, batch_size=1)
    assert str(normalize_condition_type("all")) == "unconditional"


def test_pipeline_dict_output_errors_and_round_trip() -> None:
    pipe = tiny_pipeline()
    out = pipe(
        batch_size=1,
        num_elements=1,
        seed=0,
        num_inference_steps=1,
        output_type="dict",
        return_intermediates=True,
    )
    assert type(out) is dict
    assert out["trajectory"][0].shape == (1, 3, 4)
    with pytest.raises(ValueError, match="bbox and labels are required"):
        pipe(condition_type="label", num_inference_steps=1)
    with pytest.raises(ValueError, match="Unsupported DLT condition_type"):
        pipe(condition_type="completion", num_inference_steps=1)
    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = DLTPipeline.from_pretrained(tmp)
        loaded_out = cast(
            LayoutGenerationOutput,
            loaded(batch_size=1, num_elements=1, seed=2, num_inference_steps=1),
        )
    assert_layout_output_schema(loaded_out, batch_size=1)
