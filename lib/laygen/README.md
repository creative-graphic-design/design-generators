# laygen

![package](https://img.shields.io/static/v1?label=package&message=laygen&color=blue&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![core](https://img.shields.io/static/v1?label=core&message=torch--free&color=success&style=flat-square)
![extras](https://img.shields.io/static/v1?label=extras&message=agents+%7C+diffusion+%7C+torch&color=informational&style=flat-square)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)

`laygen` contains shared layout-generation schemas and utilities used by this repository's model packages. The core package is intentionally torch-free; tensor-backed helpers, agent integrations, and 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index) adapters are available through extras declared in `lib/laygen/pyproject.toml`.

Model-specific tokenizers and generation logic stay in each model package. Shared pipeline loading rules, output schemas, bbox helpers, schedulers, and model-card helpers live here.

API reference pages are generated on the documentation site: <https://creative-graphic-design.github.io/design-generators/>.

## Install

```bash
uv sync --package laygen
uv sync --package laygen --extra torch
uv sync --package laygen --extra diffusion
uv sync --package laygen --extra agents
```

## Core APIs

Use schema and normalization helpers without importing [PyTorch](https://pytorch.org/docs/stable/index.html) or `diffusers`.

```bash
uv run --package laygen python
```

```python
from laygen.common import normalize_condition_type
from laygen.common.labels import id2label_for_dataset

print(normalize_condition_type("gen_t"))
print(id2label_for_dataset("publaynet"))
```

Use `laygen.modeling_outputs.LayoutGenerationOutput` for [`transformers`](https://huggingface.co/docs/transformers/index)-style code paths and schema-only utilities.

```bash
uv run --package laygen python
```

```python
import torch
from laygen.modeling_outputs import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 2, 4),
    labels=torch.zeros(1, 2, dtype=torch.long),
    mask=torch.tensor([[True, False]]),
    id2label={0: "text"},
)
print(out["bbox"].shape)
```

Use `laygen.pipelines.pipeline_output.LayoutGenerationOutput` inside `diffusers` pipeline packages.

```bash
uv run --package laygen python
```

```python
import torch
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 1, 4),
    labels=torch.zeros(1, 1, dtype=torch.long),
    mask=torch.ones(1, 1, dtype=torch.bool),
    id2label={0: "text"},
)
print(out.to_tuple()[0].shape)
```

Subclass `laygen.pipelines.LayoutGenerationPipeline` for `transformers`-side pipeline packages that load a root config plus model or processor subfolders.

```python
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec


class MyPipeline(LayoutGenerationPipeline):
    component_specs = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            subfolder="model",
            loader=load_model,
        )
    }
```

Build shared continuous diffusion schedules through the CompVis latent-diffusion-style adapter. Common schedules use `diffusers` under the hood while preserving vendor aliases such as `quad`.

```python
from laygen.schedulers.continuous import get_beta_schedule, get_ddim_timesteps

betas = get_beta_schedule("quad", num_timesteps=8)
timesteps = get_ddim_timesteps("uniform", 4, 100)
print(betas.shape, timesteps.tolist())
```

Convert boxes through normalized center `xywh` when crossing package boundaries.

```bash
uv run --package laygen python
```

```python
import torch
from laygen.common.bbox import denormalize_boxes, normalize_boxes

pixels = torch.tensor([[[10.0, 20.0, 40.0, 60.0]]])
boxes = normalize_boxes(pixels, canvas_size=(100, 100), box_format="ltrb")
print(denormalize_boxes(boxes, canvas_size=(100, 100), box_format="ltrb"))
```

Read dataset labels from the shared label registry.

```bash
uv run --package laygen python
```

```python
from laygen.common.labels import id2label_for_dataset, label2id_for_dataset

print(id2label_for_dataset("publaynet"))
print(label2id_for_dataset("rico25")["Text"])
```

Normalize public and vendor condition names through the shared condition enum.

```bash
uv run --package laygen python
```

```python
from laygen.common import normalize_condition_type

print(normalize_condition_type("gen_t"))
print(normalize_condition_type("gen_r"))
```

Generate a LayoutDM model card when writing converted checkpoint directories.

```bash
uv run --package laygen python
```

```python
from laygen.common.model_card import layoutdm_model_card

card = layoutdm_model_card(dataset="rico25")
print(card.data.to_dict()["datasets"])
```

Example output:

```text
['creative-graphic-design/rico25']
```

Install the `torch` extra when constructing tensor-backed `LayoutGenerationOutput` objects, and install the `diffusion` extra when using `diffusers` pipeline output helpers or continuous scheduler adapters.

## Design Rules

- Shared public outputs expose `bbox`, `labels`, `mask`, and `id2label`; optional trace data belongs in `intermediates`.
- [`transformers`](https://huggingface.co/docs/transformers/index)-side pipelines subclass `laygen.pipelines.LayoutGenerationPipeline`.
- Public boxes are normalized center `xywh` in `[0, 1]`, and `mask=True` means a valid element.
- Move code into `laygen` only after at least two model packages need it, or when a shared public contract is required before the second consumer lands.
- Keep `laygen.modeling_outputs.LayoutGenerationOutput` and `laygen.pipelines.pipeline_output.LayoutGenerationOutput` field names, order, and defaults aligned when adding or renaming output fields.
- Do not add arbitrary `extras` dictionaries to output objects. Put debug, trajectory, or model-specific data in the `intermediates` field.
- Shared schema tests should use duck typing through `laygen.common.testing` so they work for both output variants.
- `laygen.pipelines.LayoutGenerationPipeline` owns the shared `transformers`-side pipeline contract for config/subfolder loading, saving, device/dtype movement, and generator-over-seed handling. Model packages own their task-specific `__call__` orchestration.
- `laygen.modeling_outputs` and `laygen.pipelines.pipeline_output` own output schemas; `laygen.common` owns bbox, conditions, labels, testing, serialization, visualization, and model-card helpers; neural-network blocks live in `laygen.nn`, and scheduler adapters live in `laygen.schedulers`.
- `diffusers`-backed helpers stay behind the `diffusion` extra unless they specifically target `diffusers` pipeline outputs.

## References

- [issue #2](https://github.com/creative-graphic-design/design-generators/issues/2) tracks the umbrella model split and dataset policy.
- [issue #64](https://github.com/creative-graphic-design/design-generators/issues/64) defines the `lib/laygen`, `laygen.common.*`, `laygen.modeling_outputs`, `laygen.pipelines.*`, and `lib/posgen` migration layout.
- [issue #81](https://github.com/creative-graphic-design/design-generators/issues/81) defines the `laygen.nn` and `laygen.schedulers` shared-module restructuring.
