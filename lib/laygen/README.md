# laygen

![package](https://img.shields.io/static/v1?label=package&message=laygen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square)
![core](https://img.shields.io/static/v1?label=core&message=torch--free&color=success&style=flat-square)
![extras](https://img.shields.io/static/v1?label=extras&message=agents%20%7C%20diffusion%20%7C%20torch&color=informational&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square)](https://creative-graphic-design.github.io/design-generators/)

`laygen` contains shared layout-generation schemas and utilities for the workspace. The core package is intentionally torch-free; tensor-backed helpers, agent integrations, and [Diffusers](https://huggingface.co/docs/diffusers/index) adapters are available through extras declared in `lib/laygen/pyproject.toml`.

## Install

```bash
uv sync --package laygen
uv sync --package laygen --extra torch
uv sync --package laygen --extra diffusion
uv sync --package laygen --extra agents
```

## Core APIs

Use schema and normalization helpers without importing [PyTorch](https://pytorch.org/docs/stable/index.html) or Diffusers.

```bash
uv run --package laygen python - <<'PY'
from laygen.common import normalize_condition_type
from laygen.common.labels import id2label_for_dataset

print(normalize_condition_type("gen_t"))
print(id2label_for_dataset("publaynet"))
PY
```

Install the `torch` extra when constructing tensor-backed `LayoutGenerationOutput` objects, and install the `diffusion` extra when using Diffusers pipeline output helpers or continuous scheduler adapters.

## Design Rules

- Shared public outputs expose `bbox`, `labels`, `mask`, and `id2label`; optional trace data belongs in `intermediates`.
- [Transformers](https://huggingface.co/docs/transformers/index)-side pipelines subclass `laygen.pipelines.LayoutGenerationPipeline`.
- Public boxes are normalized center `xywh` in `[0, 1]`, and `mask=True` means a valid element.
- Move code into `laygen` only after at least two model packages need it, or when a shared public contract is required before the second consumer lands.

## References

- [issue #2](https://github.com/creative-graphic-design/design-generators/issues/2) tracks the umbrella model split and dataset policy.
- [issue #64](https://github.com/creative-graphic-design/design-generators/issues/64) defines the shared library structure.
- [issue #81](https://github.com/creative-graphic-design/design-generators/issues/81) defines the `laygen.nn` and `laygen.schedulers` shared-module restructuring.
