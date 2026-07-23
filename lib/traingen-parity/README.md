# traingen-parity

![package](https://img.shields.io/static/v1?label=package&message=traingen%2Dparity&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![runtime](https://img.shields.io/static/v1?label=runtime&message=torch&color=informational&style=flat-square)
![extras](https://img.shields.io/static/v1?label=extras&message=lightning&color=informational&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)

`traingen-parity` contains deterministic training-parity primitives for comparing two training implementations in [design-generators](https://github.com/creative-graphic-design/design-generators). It records named tensors from one training step, captures and restores RNG state, applies deterministic runtime settings, and reports tensor differences for step traces, optimizer states, and dataloader streams.

Keep package-specific trace-point selection in the model package. Use this library for shared comparison mechanics and reproducibility controls.

## Install

```bash
uv sync --package traingen-parity
uv sync --package traingen-parity --extra lightning
```

Install from outside the workspace with pip's direct-reference subdirectory form. Install [`traingen`](https://github.com/creative-graphic-design/design-generators/tree/main/lib/traingen) first when using the `lightning` extra, because `traingen-parity[lightning]` depends on `traingen[lightning]`.

```bash
pip install "traingen[lightning] @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/traingen"
pip install "traingen-parity @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/traingen-parity"
pip install "traingen-parity[lightning] @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/traingen-parity"
```

## API Entry Points

Compare exact tensor traces for a single training step.

```bash
uv run --package traingen-parity python
```

```python
import torch
from traingen_parity import build_step_trace, compare_step_trace

reference = build_step_trace("reference", {"loss": torch.tensor(0.0)})
target = build_step_trace("target", {"loss": torch.tensor(0.0)})

print(compare_step_trace(reference, target).passed)
```

Capture deterministic summaries and RNG state for reproducible parity hooks.

```python
import torch
from traingen_parity import (
    DeterminismConfig,
    apply_determinism,
    capture_rng_state,
    summarize_tensor,
)

apply_determinism(DeterminismConfig(seed=1, deterministic_algorithms=False))
state = capture_rng_state()
summary = summarize_tensor(torch.ones(2))

print(bool(state.torch_cpu.numel()))
print(summary.shape)
```

Compare optimizer parameters after one update.

```python
import torch
from traingen_parity import compare_optimizer_step

reference_state = {"weight": torch.ones(2)}
target_state = {"weight": torch.ones(2)}

print(compare_optimizer_step(reference_state, target_state).passed)
```

## Scope

- Keep comparison dataclasses, tensor summaries, RNG helpers, and deterministic runtime controls here.
- Keep model-specific fixture construction, trace-point names, and parity pytest markers in `models/*`.
- Keep routine Lightning hooks and CLI integration in `traingen`.

## Pointers

- [Documentation site](https://creative-graphic-design.github.io/design-generators/)
- [API reference](https://creative-graphic-design.github.io/design-generators/api/libraries/traingen-parity/)
