import sys
import types

from traingen.lightning.cli import lightning_cli_class, main


def test_lightning_cli_class_imports_lightning_lazily(monkeypatch) -> None:
    class FakeLightningCLI:
        pass

    lightning_module = types.ModuleType("lightning")
    pytorch_module = types.ModuleType("lightning.pytorch")
    cli_module = types.ModuleType("lightning.pytorch.cli")
    setattr(cli_module, "LightningCLI", FakeLightningCLI)
    setattr(pytorch_module, "cli", cli_module)
    setattr(lightning_module, "pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "lightning", lightning_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.cli", cli_module)

    assert lightning_cli_class() is FakeLightningCLI


def test_main_uses_yaml_class_path_mode(monkeypatch) -> None:
    calls = []

    class FakeLightningCLI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    lightning_module = types.ModuleType("lightning")
    pytorch_module = types.ModuleType("lightning.pytorch")
    cli_module = types.ModuleType("lightning.pytorch.cli")
    setattr(cli_module, "LightningCLI", FakeLightningCLI)
    setattr(pytorch_module, "cli", cli_module)
    setattr(lightning_module, "pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "lightning", lightning_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.cli", cli_module)

    assert isinstance(main(["fit", "--config", "config.yaml"]), FakeLightningCLI)
    assert calls == [
        {
            "model_class": None,
            "datamodule_class": None,
            "subclass_mode_model": True,
            "subclass_mode_data": True,
            "args": ["fit", "--config", "config.yaml"],
        }
    ]
