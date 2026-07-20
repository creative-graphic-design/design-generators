import torch

from ds_gan import DSGANConfig, DSGANModel, random_initial_layout


def tiny_config() -> DSGANConfig:
    return DSGANConfig(
        backbone="resnet18",
        max_elem=4,
        hidden_size=32,
        num_layers=2,
        image_size=(64, 64),
        backbone_feature_size=16,
    )


def test_random_initial_layout_shape_and_generator_precedence():
    generator = torch.Generator().manual_seed(7)
    first = random_initial_layout(2, 4, generator=generator, seed=123)
    generator = torch.Generator().manual_seed(7)
    second = random_initial_layout(2, 4, generator=generator, seed=999)

    assert first.shape == (2, 4, 2, 4)
    assert torch.equal(first, second)
    assert torch.allclose(first[:, :, 0].sum(dim=-1), torch.ones(2, 4))


def test_model_forward_returns_vendor_shapes():
    config = tiny_config()
    model = DSGANModel(config).eval()
    pixel_values = torch.zeros(1, 4, 64, 64)
    layout = random_initial_layout(1, config.max_elem, seed=0)

    with torch.no_grad():
        output = model(pixel_values=pixel_values, layout=layout)
        tuple_output = model(
            pixel_values=pixel_values, layout=layout, return_dict=False
        )

    assert output.class_probs.shape == (1, 4, 4)
    assert tuple_output[0].shape == (1, 4, 4)
    assert output.bbox.shape == (1, 4, 4)
    assert torch.allclose(output.class_probs.sum(dim=-1), torch.ones(1, 4))
    assert bool(torch.all((output.bbox >= 0) & (output.bbox <= 1)))


def test_model_rejects_bad_layout_shape():
    model = DSGANModel(tiny_config())
    pixel_values = torch.zeros(1, 4, 64, 64)

    try:
        model(pixel_values=pixel_values, layout=torch.zeros(1, 3, 2, 4))
    except ValueError as exc:
        assert "layout must have shape" in str(exc)
    else:
        raise AssertionError("expected bad layout shape to raise")


def test_model_rejects_bad_pixel_values_shape_and_backbone():
    model = DSGANModel(tiny_config())

    try:
        model(pixel_values=torch.zeros(1, 3, 64, 64), layout=torch.zeros(1, 4, 2, 4))
    except ValueError as exc:
        assert "pixel_values must have shape" in str(exc)
    else:
        raise AssertionError("expected bad pixel shape to raise")

    bad_config = tiny_config()
    bad_config.backbone = "unknown"
    try:
        DSGANModel(bad_config)
    except ValueError as exc:
        assert "Unsupported DS-GAN backbone" in str(exc)
    else:
        raise AssertionError("expected bad backbone to raise")


def test_random_initial_layout_numpy_and_uniform_modes():
    numpy_layout = random_initial_layout(
        1,
        3,
        seed=3,
        weighted_classes=True,
        use_numpy_classes=True,
    )
    uniform_layout = random_initial_layout(1, 3, seed=3, weighted_classes=False)

    assert numpy_layout.shape == (1, 3, 2, 4)
    assert uniform_layout.shape == (1, 3, 2, 4)
