import pytest
import torch

from canvas_vae import CanvasVAEModel
from canvas_vae.modeling_canvas_vae import CanvasVAEBatchNorm, length_mask


def batch(config, sizes=(3, 1)):
    generator = torch.Generator().manual_seed(0)
    width = max(sizes)
    columns = [
        torch.randint(0, size, (len(sizes), width, 1), generator=generator)
        for size in config.field_sizes.values()
    ]
    return torch.tensor(sizes), torch.cat(columns, dim=-1)


def test_training_forward_has_losses(config):
    model = CanvasVAEModel(config).train()
    num_elements, element_ids = batch(config)
    output = model(num_elements, element_ids, posterior_noise=torch.zeros(2, 16))
    assert output.loss is not None and output.loss.ndim == 0
    assert set(output.reconstruction_losses) == {"length", *config.field_sizes}
    assert output.mask.tolist() == [[True, True, True], [True, False, False]]
    torch.testing.assert_close(output.latents, output.z_mean)
    output.loss.backward()


def test_training_samples_noise_without_injection(config):
    model = CanvasVAEModel(config).train()
    output = model(*batch(config))
    assert not torch.equal(output.latents, output.z_mean)


def test_eval_forward_uses_mean_and_predicted_length(config):
    model = CanvasVAEModel(config).eval()
    output = model(*batch(config))
    assert output.loss is None
    torch.testing.assert_close(output.latents, output.z_mean)
    predicted = output.length_logits.argmax(-1) + 1
    assert output.mask.sum(-1).tolist() == predicted.tolist()


def test_latent_decoding_and_errors(config):
    model = CanvasVAEModel(config).eval()
    output = model(latents=torch.zeros(3, 16))
    assert output.element_logits["component"].shape[:2] == output.mask.shape
    with pytest.raises(ValueError):
        model()


def test_multiple_blocks_pool_only_last(make_config):
    model = CanvasVAEModel(make_config(num_blocks=2))
    assert [block.pooling for block in model.encoder.blocks] == [False, True]


def test_initialization_matches_keras_defaults(config):
    model = CanvasVAEModel(config)
    assert model.encoder.length_embedding.weight.abs().max() <= 0.05
    assert torch.count_nonzero(model.decoder.length_head.bias) == 0
    bound = (6 / (16 + 32)) ** 0.5
    assert model.encoder.blocks[0].mlp[0].weight.abs().max() <= bound
    assert torch.equal(model.encoder.norm.running_var, torch.ones(16))


def test_batch_norm_updates_population_statistics():
    norm = CanvasVAEBatchNorm(2, eps=1e-3, momentum=0.9)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    output = norm(inputs)
    torch.testing.assert_close(norm.running_mean, torch.tensor([0.2, 0.4]))
    torch.testing.assert_close(norm.running_var, torch.tensor([0.9 + 0.1, 0.9 + 0.4]))
    expected = (inputs - inputs.mean(0)) / torch.sqrt(
        inputs.var(0, unbiased=False) + 1e-3
    )
    torch.testing.assert_close(output, expected)
    norm.eval()
    torch.testing.assert_close(
        norm(inputs), (inputs - norm.running_mean) / torch.sqrt(norm.running_var + 1e-3)
    )


def test_length_mask():
    assert length_mask(torch.tensor([1]), 3).tolist() == [[True, True, False]]


def test_save_load_round_trip(tmp_path, config):
    model = CanvasVAEModel(config).eval()
    model.save_pretrained(tmp_path)
    loaded = CanvasVAEModel.from_pretrained(tmp_path).eval()
    latents = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(
        loaded(latents=latents).length_logits, model(latents=latents).length_logits
    )
    torch.testing.assert_close(
        loaded.encoder.norm.running_var, model.encoder.norm.running_var
    )
