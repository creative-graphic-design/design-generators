# laygen

Shared layout-generation schemas and utilities for the design-generators workspace.

`laygen` holds code that is useful to more than one layout model. Model-specific checkpoint loading, tokenizers, schedulers, and pipelines stay in each model package.

API reference pages are generated on the documentation site: <https://creative-graphic-design.github.io/design-generators/>.

## Usage

Use `laygen.modeling_outputs.LayoutGenerationOutput` for Transformers-style code paths and schema-only utilities. Use `laygen.pipelines.pipeline_output.LayoutGenerationOutput` inside Diffusers pipeline packages.

```bash
uv run --package laygen python - <<'PY'
import torch
from laygen.modeling_outputs import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 2, 4),
    labels=torch.zeros(1, 2, dtype=torch.long),
    mask=torch.tensor([[True, False]]),
    id2label={0: "text"},
)
print(out["bbox"].shape)
PY
```

Use `laygen.pipelines.pipeline_output.LayoutGenerationOutput` inside Diffusers pipeline packages.

```bash
uv run --package laygen python - <<'PY'
import torch
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 1, 4),
    labels=torch.zeros(1, 1, dtype=torch.long),
    mask=torch.ones(1, 1, dtype=torch.bool),
    id2label={0: "text"},
)
print(out.to_tuple()[0].shape)
PY
```

Build shared continuous diffusion schedules through the CompVis latent-diffusion-style adapter. Common schedules use Diffusers under the hood while preserving vendor aliases such as `quad`.

```python
from laygen.schedulers.continuous import get_beta_schedule, get_ddim_timesteps

betas = get_beta_schedule("quad", num_timesteps=8)
timesteps = get_ddim_timesteps("uniform", 4, 100)
print(betas.shape, timesteps.tolist())
```

Convert boxes through normalized center `xywh` when crossing package boundaries.

```bash
uv run --package laygen python - <<'PY'
import torch
from laygen.common.bbox import denormalize_boxes, normalize_boxes

pixels = torch.tensor([[[10.0, 20.0, 40.0, 60.0]]])
boxes = normalize_boxes(pixels, canvas_size=(100, 100), box_format="ltrb")
print(denormalize_boxes(boxes, canvas_size=(100, 100), box_format="ltrb"))
PY
```

Read dataset labels from the shared label registry.

```bash
uv run --package laygen python - <<'PY'
from laygen.common.labels import id2label_for_dataset, label2id_for_dataset

print(id2label_for_dataset("publaynet"))
print(label2id_for_dataset("rico25")["Text"])
PY
```

Normalize public and vendor condition names through the shared condition enum.

```bash
uv run --package laygen python - <<'PY'
from laygen.common import normalize_condition_type

print(normalize_condition_type("gen_t"))
print(normalize_condition_type("gen_r"))
PY
```

Generate a LayoutDM model card when writing converted checkpoint directories.

```bash
uv run --package laygen python - <<'PY'
from laygen.common.model_card import layoutdm_model_card

card = layoutdm_model_card(dataset="rico25")
print(card.data.to_dict()["datasets"])
PY
```

Example output:

```text
['creative-graphic-design/rico25']
```

## Design Rules

- Keep `laygen.modeling_outputs.LayoutGenerationOutput` and `laygen.pipelines.pipeline_output.LayoutGenerationOutput` field names, order, and defaults aligned when adding or renaming output fields.
- Do not add arbitrary `extras` dictionaries to output objects. Put debug, trajectory, or model-specific data in the `intermediates` field.
- Shared schema tests should use duck typing through `laygen.common.testing` so they work for both output variants.
- Move code into `laygen` only after at least two model packages need it, or when a shared public contract is required before the second consumer lands.
- `laygen.modeling_outputs` and `laygen.pipelines.pipeline_output` own output schemas; `laygen.common` owns bbox, conditions, labels, testing, serialization, visualization, and model-card helpers; neural-network blocks live in `laygen.nn`, and scheduler adapters live in `laygen.schedulers`.
- Diffusers is a normal laygen dependency. Keep schema-only helpers independent of Diffusers imports unless they specifically target Diffusers pipeline outputs.

## References

- [Issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2) tracks the original layout-generation model split.
- [Issue #64 (shared library structure)](https://github.com/creative-graphic-design/design-generators/issues/64) defines the `lib/laygen`, `laygen.common.*`, `laygen.modeling_outputs`, `laygen.pipelines.*`, and `lib/posgen` migration layout.
- [Issue #81 (laygen shared modules)](https://github.com/creative-graphic-design/design-generators/issues/81) defines the `laygen.nn` and `laygen.schedulers` shared-module restructuring.
