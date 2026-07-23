# traingen

![package](https://img.shields.io/static/v1?label=package&message=traingen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![extras](https://img.shields.io/static/v1?label=extras&message=lightning&color=informational&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)

`traingen` contains shared training utilities for [design-generators](https://github.com/creative-graphic-design/design-generators) packages in the train-ourselves lane. It keeps reusable [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) CLI integration points out of individual model packages while leaving model-specific data modules, losses, and training loops in `models/*`.

Use `traingen` for generic training infrastructure such as optional [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) helpers. Use [`traingen-parity`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/traingen-parity) for deterministic trace capture and reference/target comparison reports.

## Install

```bash
uv sync --package traingen
uv sync --package traingen --extra lightning
```

Install from outside the workspace with pip's direct-reference subdirectory form:

```bash
pip install "traingen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/traingen"
pip install "traingen[lightning] @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/traingen"
```

## API Entry Points

Run a model-agnostic LightningCLI from a training YAML that declares
`model.class_path` and `data.class_path`.

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/smoke.yaml
```

Create package-local LightningCLI subclasses only when a model package needs
custom CLI arguments or defaults that cannot live in YAML.

```bash
uv run --package traingen python
```

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
