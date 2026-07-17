# laygen

Shared layout-generation schemas and utilities for the design-generators workspace.

`laygen` holds code that is useful to more than one layout model. Model-specific checkpoint loading, tokenizers, schedulers, and pipelines stay in each model package.

## Module Map

| Module | Purpose |
| --- | --- |
| `laygen.common.outputs` | Canonical layout output on `transformers.utils.ModelOutput`. |
| `laygen.common.outputs_diffusers` | Diffusers pipeline output on `diffusers.utils.BaseOutput`. Requires the `diffusers` extra. |
| `laygen.common.bbox` | Normalized layout box conversion, clamping, discretization, and continuization helpers. |
| `laygen.common.labels` | Dataset label vocabularies and `id2label` / `label2id` maps. |
| `laygen.common.discrete` | Discrete diffusion tensor helpers and categorical sampling utilities. |
| `laygen.common.visualization` | Lightweight layout rendering helpers for quick inspection. |
| `laygen.common.testing` | Duck-typed schema assertions shared by model tests. |
| `laygen.common.model_card` | Model-card builders used by converted layout model packages. |

## Usage

Use `laygen.common.outputs.LayoutGenerationOutput` for Transformers-style code paths and shared utilities that should not import Diffusers.

```bash
uv run --package laygen python - <<'PY'
import torch
from laygen.common.outputs import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 2, 4),
    labels=torch.zeros(1, 2, dtype=torch.long),
    mask=torch.tensor([[True, False]]),
    id2label={0: "text"},
)
print(out["bbox"].shape)
PY
```

Use `laygen.common.outputs_diffusers.LayoutGenerationOutput` only inside Diffusers pipeline packages.

```bash
uv run --package laygen --extra diffusers python - <<'PY'
import torch
from laygen.common.outputs_diffusers import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.zeros(1, 1, 4),
    labels=torch.zeros(1, 1, dtype=torch.long),
    mask=torch.ones(1, 1, dtype=torch.bool),
    id2label={0: "text"},
)
print(out.to_tuple()[0].shape)
PY
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

- `outputs` and `outputs_diffusers` share one field definition from `laygen.common._output_spec`; add or rename output fields there first.
- Do not add arbitrary `extras` dictionaries to output objects. Put debug, trajectory, or model-specific data in the `intermediates` field.
- Shared schema tests should use duck typing through `laygen.common.testing` so they work for both output variants.
- Move code into `laygen` only after at least two model packages need it, or when a shared public contract is required before the second consumer lands.
- `laygen` may provide optional Diffusers integration, but the default import path must stay usable without importing Diffusers.

## References

- #2 tracks the original layout-generation model split.
- #64 defines the `lib/laygen`, `laygen.common.*`, and `lib/posgen` migration layout.
