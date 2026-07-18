# laygen

Shared layout-generation schemas and utilities for the design-generators workspace.

`laygen` holds code that is useful to more than one layout model. Model-specific checkpoint loading, tokenizers, schedulers, and pipelines stay in each model package.

## Module Map

| Module | Purpose |
| --- | --- |
| `laygen.modeling_outputs` | Canonical layout output on `transformers.utils.ModelOutput`. |
| `laygen.pipelines.pipeline_output` | Diffusers pipeline output on `diffusers.utils.BaseOutput`; Diffusers is a normal laygen dependency because shared schedulers also use it. |
| `laygen.common.bbox` | Normalized layout box conversion, clamping, discretization, and continuization helpers. |
| `laygen.common.conditions` | Canonical `ConditionType` enum and vendor alias normalization. |
| `laygen.common.labels` | Dataset label vocabularies and `id2label` / `label2id` maps. |
| `laygen.common.discrete` | Discrete diffusion tensor helpers and categorical sampling utilities. |
| `laygen.common.visualization` | Lightweight layout rendering helpers for quick inspection. |
| `laygen.common.testing` | Duck-typed schema assertions shared by model tests. |
| `laygen.common.model_card` | Model-card builders used by converted layout model packages. |
| `laygen.nn.activations` | Shared activation enum and resolver; `gelu2` maps the VQ-Diffusion / QuickGELU branch to Transformers `ACT2FN["quick_gelu"]`. |
| `laygen.nn.embeddings` | VQ-Diffusion-derived timestep embeddings plus the LayoutDM-specific element/attribute positional embedding. |
| `laygen.nn.norms` | VQ-Diffusion-derived adaptive layer and instance normalization used by LayoutDM and LACE. |
| `laygen.nn.blocks` | VQ-Diffusion-derived timestep transformer encoder block and encoder stack kept key-compatible with converted checkpoints. |
| `laygen.nn.module_utils` | Shared module cloning helper used by the VQ-Diffusion-derived transformer stacks and LayoutFlow. |
| `laygen.schedulers.continuous` | CompVis latent-diffusion-style beta and DDIM timestep helpers; common schedules delegate to Diffusers schedulers. |

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

- #2 tracks the original layout-generation model split.
- #64 defines the `lib/laygen`, `laygen.common.*`, `laygen.modeling_outputs`, `laygen.pipelines.*`, and `lib/posgen` migration layout.
- #81 defines the `laygen.nn` and `laygen.schedulers` shared-module restructuring.
