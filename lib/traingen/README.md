# traingen

![package](https://img.shields.io/static/v1?label=package&message=traingen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![extras](https://img.shields.io/static/v1?label=extras&message=lightning&color=informational&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)

`traingen` contains shared training utilities for design-generator packages in the train-ourselves lane. It keeps reusable Lightning inspection helpers and CLI integration points out of individual model packages while leaving model-specific data modules, losses, and training loops in `models/*`.

Use `traingen` for generic training infrastructure such as optimizer diagnostics and optional LightningCLI helpers. Use `traingen-parity` for deterministic trace capture and reference/target comparison reports.

## Install

```bash
uv sync --package traingen
uv sync --package traingen --extra lightning
```

## API Entry Points

Inspect optimizer learning rates and parameter gradient norms from a Lightning module or plain PyTorch module.

```bash
uv run --package traingen python
```

```python
import torch
from traingen.lightning import grad_norms, learning_rates

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
loss = model(torch.ones(1, 2)).sum()
loss.backward()

print(learning_rates(optimizer))
print(sorted(grad_norms(model)))
```

Create package-local LightningCLI subclasses without importing Lightning at top-level package import time.

```python
from traingen.lightning.cli import lightning_cli_class


class MyLightningCLI(lightning_cli_class()):
    pass
```

## Scope

- Keep reusable training utilities here only when more than one train-ourselves package can use them.
- Keep dataset-specific modules, model losses, condition policies, and checkpoint conversion in each model package.
- Keep deterministic comparison logic in `traingen-parity` so regular training helpers do not depend on parity-only behavior.

## Pointers

- [Documentation site](https://creative-graphic-design.github.io/design-generators/)
- [API reference](https://creative-graphic-design.github.io/design-generators/api/libraries/traingen/)
