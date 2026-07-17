import tempfile

import torch

from lace import LacePipeline, LaceProcessor, LaceScheduler, LaceTransformerModel


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
    first = pipe(batch_size=2, seed=123, num_inference_steps=2)
    second = pipe(batch_size=2, seed=123, num_inference_steps=2)
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
        out = pipe(
            condition_type=condition_type,
            bbox=bbox,
            labels=labels,
            mask=mask,
            seed=7,
            num_inference_steps=2,
        )
        assert out.bbox.shape == (1, 5, 4)
        assert out.labels.shape == (1, 5)


def test_pipeline_save_pretrained_smoke() -> None:
    pipe = _tiny_pipe()
    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = LacePipeline.from_pretrained(tmp)
    out = loaded(batch_size=1, seed=1, num_inference_steps=1)
    assert out.bbox.shape == (1, 5, 4)
