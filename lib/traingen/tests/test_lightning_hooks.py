import sys
import types

import torch

from traingen.lightning.cli import lightning_cli_class
from traingen.lightning.hooks import grad_norms, learning_rates


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


def test_lightning_hooks_report_lr_and_grad_norms() -> None:
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    output = module(torch.ones(1, 2)).sum()
    output.backward()
    assert learning_rates(optimizer) == (0.1,)
    norms = grad_norms(module)
    assert set(norms) == {"weight", "bias"}
