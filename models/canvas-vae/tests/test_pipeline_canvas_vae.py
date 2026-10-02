import pytest
import torch
from laygen.common.testing import assert_generator_reproducible, assert_layout_output_schema

from canvas_vae import CanvasVAEModel, CanvasVAEPipeline
from canvas_vae.pipeline_canvas_vae import bins_to_ltwh


@pytest.fixture
def pipe(config):
    return CanvasVAEPipeline(model=CanvasVAEModel(config))


def test_unconditional_output_schema(pipe):
    out = pipe(batch_size=3, seed=0, return_intermediates=True)
    assert_layout_output_schema(out, batch_size=3)
    assert out.id2label == pipe.model.config.id2label
    assert set(out.intermediates) == {"clickable", "icon", "text_button", "latents"}
    assert not out.labels[~out.mask].any()


def test_generator_reproducible_and_wins_over_seed(pipe):
    assert_generator_reproducible(lambda generator: pipe(batch_size=2, generator=generator))
    first = pipe(batch_size=2, seed=1, generator=torch.Generator().manual_seed(5), return_intermediates=True)
    second = pipe(batch_size=2, seed=2, generator=torch.Generator().manual_seed(5), return_intermediates=True)
    assert torch.equal(first.intermediates["latents"], second.intermediates["latents"])


def test_dict_output(pipe):
    out = pipe(batch_size=1, seed=0, output_type="dict")
    assert isinstance(out, dict) and {"bbox", "labels", "mask", "id2label"} <= set(out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"condition_type": "label"},
        {"labels": [[1]]},
        {"num_elements": 2},
        {"box_format": "ltrb"},
        {"normalized": False},
        {"num_inference_steps": 2},
    ],
)
def test_unsupported_requests_raise(pipe, kwargs):
    with pytest.raises(NotImplementedError):
        pipe(**kwargs)


def test_save_and_load(tmp_path, pipe):
    pipe.save_pretrained(tmp_path)
    loaded = CanvasVAEPipeline.from_pretrained(tmp_path)
    expected = pipe(batch_size=2, seed=3)
    actual = loaded(batch_size=2, seed=3)
    assert torch.equal(actual.bbox, expected.bbox) and torch.equal(actual.labels, expected.labels)


def test_bins_to_ltwh():
    assert bins_to_ltwh(torch.tensor([0, 63]), 64).tolist() == [0.0, 1.0]
