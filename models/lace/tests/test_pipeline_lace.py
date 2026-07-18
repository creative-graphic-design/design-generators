import tempfile
from typing import cast

import pytest
import torch

from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from lace import (
    ConditionType,
    LacePipeline,
    LaceProcessor,
    LaceScheduler,
    LaceTransformerModel,
    PipelineOutputType,
    normalize_condition_type,
    normalize_output_type,
)


def _tiny_pipe() -> LacePipeline:
    model = LaceTransformerModel(
        seq_dim=10,
        max_seq_length=5,
        num_layers=1,
        dim_transformer=32,
        nhead=4,
        dim_feedforward=64,
        diffusion_step=1000,
    )
    return LacePipeline(
        model=model,
        scheduler=LaceScheduler(ddim_num_steps=2),
        processor=LaceProcessor(
            dataset="publaynet",
            labels=["text", "title", "list", "table", "figure"],
            max_seq_length=5,
        ),
    )


def test_pipeline_unconditional_seed_reproducible() -> None:
    pipe = _tiny_pipe()
    first = cast(
        LayoutGenerationOutput, pipe(batch_size=2, seed=123, num_inference_steps=2)
    )
    second = cast(
        LayoutGenerationOutput, pipe(batch_size=2, seed=123, num_inference_steps=2)
    )
    assert first.bbox.shape == (2, 5, 4)
    assert first.labels.shape == (2, 5)
    assert first.mask.shape == (2, 5)
    assert torch.allclose(first.bbox, second.bbox)
    assert torch.equal(first.labels, second.labels)
    assert torch.all((0 <= first.bbox) & (first.bbox <= 1))


def test_pipeline_condition_modes_return_shapes() -> None:
    pipe = _tiny_pipe()
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    labels = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, False]])
    for condition_type in ["label", "label_size", "completion", "refinement"]:
        out = cast(
            LayoutGenerationOutput,
            pipe(
                condition_type=condition_type,
                bbox=bbox,
                labels=labels,
                mask=mask,
                seed=7,
                num_inference_steps=2,
            ),
        )
        assert out.bbox.shape == (1, 5, 4)
        assert out.labels.shape == (1, 5)


def test_pipeline_enum_modes_dict_output_and_errors() -> None:
    pipe = _tiny_pipe()
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    labels = torch.tensor([[1]])
    out = pipe(
        condition_type=ConditionType.label_size,
        bbox=bbox,
        labels=labels,
        seed=4,
        num_inference_steps=1,
        output_type=PipelineOutputType.dict,
        return_intermediates=True,
    )
    assert type(out) is dict
    assert out["bbox"].shape == (1, 5, 4)
    assert out["trajectory"][0].shape == (1, 5, 10)
    assert normalize_condition_type("c") is ConditionType.label
    assert normalize_output_type("dataclass") is PipelineOutputType.dataclass

    with pytest.raises(ValueError, match="bbox and labels are required"):
        pipe(condition_type=ConditionType.label, num_inference_steps=1)
    with pytest.raises(ValueError, match="Unsupported LACE condition_type"):
        normalize_condition_type("bad")
    with pytest.raises(ValueError, match="Unsupported output_type"):
        normalize_output_type("bad")


def test_pipeline_save_pretrained_smoke() -> None:
    pipe = _tiny_pipe()
    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = LacePipeline.from_pretrained(tmp)
    out = cast(
        LayoutGenerationOutput, loaded(batch_size=1, seed=1, num_inference_steps=1)
    )
    assert out.bbox.shape == (1, 5, 4)
