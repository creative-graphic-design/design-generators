import os
from pathlib import Path
import sys

import pytest
from torch import nn
import torch

from layoutvae import LayoutVAEModel
from layoutvae.conversion import load_original_modules, load_original_state_dicts


pytestmark = pytest.mark.vendor_parity


def _required_path(env_name):
    value = os.environ.get(env_name)
    if value:
        path = Path(value)
        if path.exists():
            return path
    if os.environ.get("PARITY_REQUIRE") == "1":
        raise AssertionError(f"{env_name} must point to an existing path")
    pytest.skip(f"{env_name} is not set")


def _source_root():
    candidates = [
        Path(value)
        for value in [
            os.environ.get("LAYOUTVAE_SOURCE_ROOT"),
            "vendor/layout-generation-baselines/LayoutVAE",
        ]
        if value
    ]
    for path in candidates:
        if path.exists():
            return path
    if os.environ.get("PARITY_REQUIRE") == "1":
        raise AssertionError("LAYOUTVAE_SOURCE_ROOT must point to an existing path")
    pytest.skip("LayoutVAE source root is not available")


class _FixedLatents(nn.Module):
    def __init__(self, latents):
        super().__init__()
        self.latents: torch.Tensor
        self.register_buffer("latents", latents)
        self.index = 0

    def forward(self, _inputs):
        value = self.latents[:, self.index, :]
        self.index += 1
        return value


class _FixedBboxNoise(nn.Module):
    def __init__(self, noise):
        super().__init__()
        self.noise: torch.Tensor
        self.register_buffer("noise", noise)
        self.index = 0

    def forward(self, inputs):
        value = inputs + self.noise[:, self.index, :]
        self.index += 1
        return value


class _FixedPoisson:
    def __init__(self, samples):
        self.samples = samples
        self.index = 0

    def __call__(self, _rate):
        return self

    def sample(self):
        value = self.samples[:, self.index].view(-1, 1)
        self.index += 1
        return value


def _fixed_inputs():
    label_set = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    )
    count_latents = torch.linspace(-0.75, 0.75, steps=2 * 6 * 32).reshape(2, 6, 32)
    count_samples = torch.tensor(
        [[0.0, 2.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 3.0, 4.0, 0.0, 0.0]]
    )
    bbox_latents = torch.linspace(-0.5, 0.5, steps=2 * 9 * 32).reshape(2, 9, 32)
    bbox_noise = torch.linspace(0.0, 0.019, steps=2 * 9 * 4).reshape(2, 9, 4)
    return label_set, count_latents, count_samples, bbox_latents, bbox_noise


def _run_original_count(count_module, label_set, count_latents, count_samples):
    count_module.eval()
    count_module.rep = _FixedLatents(count_latents)
    module = sys.modules[count_module.__class__.__module__]
    original_poisson = getattr(module, "Poisson")
    setattr(module, "Poisson", _FixedPoisson(count_samples))
    try:
        with torch.no_grad():
            return count_module(label_set, isTrain=False)
    finally:
        setattr(module, "Poisson", original_poisson)


def _run_original_bbox(
    bbox_module, class_counts, class_labels, bbox_latents, bbox_noise
):
    bbox_module.eval()
    bbox_module.rep = _FixedLatents(bbox_latents)
    bbox_module.rep_mul = _FixedBboxNoise(bbox_noise)
    with torch.no_grad():
        output = bbox_module([class_counts, class_labels], isTrain=False)
    return output.permute(2, 0, 1)


def test_reference_metadata_exists():
    reference_dir = _required_path("LAYOUTVAE_REFERENCE_DIR")
    assert (reference_dir / "metadata.json").is_file()
    assert (reference_dir / "fixture.pt").is_file()
    assert (reference_dir / "fixed_forward.pt").is_file()


def test_converted_checkpoint_exists():
    converted_dir = _required_path("LAYOUTVAE_CONVERTED_DIR")
    assert (converted_dir / "config.json").is_file()
    assert (converted_dir / "model.safetensors").is_file()
    assert (converted_dir / "preprocessor_config.json").is_file()


def test_converted_state_dict_matches_original_tensors():
    converted_dir = _required_path("LAYOUTVAE_CONVERTED_DIR")
    count_state_dict, bbox_state_dict = load_original_state_dicts(_source_root())
    model = LayoutVAEModel.from_pretrained(converted_dir)
    for key, expected in count_state_dict.items():
        torch.testing.assert_close(model.countvae.state_dict()[key], expected)
    for key, expected in bbox_state_dict.items():
        torch.testing.assert_close(model.bboxvae.state_dict()[key], expected)


def test_fixed_latent_forward_matches_original_modules():
    converted_dir = _required_path("LAYOUTVAE_CONVERTED_DIR")
    source_root = _source_root()
    original_count, original_bbox = load_original_modules(source_root)
    model = LayoutVAEModel.from_pretrained(converted_dir)
    model.eval()
    label_set, count_latents, count_samples, bbox_latents, bbox_noise = _fixed_inputs()

    original_counts = _run_original_count(
        original_count, label_set, count_latents, count_samples
    )
    converted_counts = model.countvae(
        label_set,
        latents=count_latents,
        count_samples=count_samples,
    )
    torch.testing.assert_close(converted_counts, original_counts, atol=0.0, rtol=0.0)

    class_counts = model._normalize_counts(converted_counts)
    class_labels = model._labels_from_counts(class_counts)
    original_boxes = _run_original_bbox(
        original_bbox, class_counts, class_labels, bbox_latents, bbox_noise
    )
    converted_boxes = model.bboxvae(
        class_counts,
        class_labels,
        latents=bbox_latents,
        output_noise=bbox_noise,
    )
    torch.testing.assert_close(converted_boxes, original_boxes, atol=1e-6, rtol=1e-6)

    converted_output = model(
        label_set,
        count_latents=count_latents,
        count_samples=count_samples,
        bbox_latents=bbox_latents,
        bbox_noise=bbox_noise,
        return_dict=True,
    )
    torch.testing.assert_close(
        converted_output.raw_ltwh, original_boxes, atol=1e-6, rtol=1e-6
    )
