import sys
import types

from traingen.lightning.cli import lightning_cli_class


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
