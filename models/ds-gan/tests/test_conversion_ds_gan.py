import torch
from types import SimpleNamespace

from ds_gan.conversion import config_from_vendor_args, convert_vendor_state_dict


def test_convert_vendor_state_dict_strips_data_parallel_prefixes():
    converted = convert_vendor_state_dict(
        {
            "module.fc1.weight": torch.zeros(1),
            "generator.fc2.bias": torch.ones(1),
        }
    )

    assert set(converted) == {"fc1.weight", "fc2.bias"}


def test_config_from_vendor_args_uses_defaults():
    config = config_from_vendor_args({"max_elem": "8", "backbone": "resnet18"})

    assert config.max_elem == 8
    assert config.hidden_size == 64
    assert config.backbone == "resnet18"


def test_config_from_vendor_namespace_and_bad_int():
    args = SimpleNamespace(
        max_elem=4,
        hidden_size=32,
        num_layers=2,
        output_size=8,
        in_channels=8,
        out_channels=32,
        backbone="resnet18",
    )

    assert config_from_vendor_args(args).num_layers == 2

    try:
        config_from_vendor_args({"max_elem": object()})
    except ValueError:
        pass
    else:
        raise AssertionError("expected bad integer value to raise")
