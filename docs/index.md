# design-generators

`design-generators` packages layout, poster, and graphic-design generation methods behind familiar Transformers, Diffusers, and Pydantic AI interfaces. Use it to load locally converted checkpoints or prompt configurations, generate labeled bounding boxes, and compare methods through one public output schema.

## Start Here

- [Getting Started](getting-started.md) shows workspace install, local checkpoint conversion, and first inference.
- [Models](models.md) lists the packaged methods, runtime style, datasets, and planned Hub ids.
- [Conventions](conventions.md) documents output fields, box coordinates, masks, labels, and supported `condition_type` names.
- [API Reference](api/index.md) is generated from workspace packages under `lib/*` and `models/*`.

## Quickstart

Install the LayoutDM workspace member and check the public API imports. Hub ids are planned until checkpoint publication is complete, so [Getting Started](getting-started.md) shows the local conversion commands before the `from_pretrained` inference snippet.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-dm
uv run --package layout-dm python - <<'PY'
from layout_dm import LayoutDMPipeline

print(LayoutDMPipeline.__name__)
PY
```

```text
LayoutDMPipeline
```

Run [Getting Started](getting-started.md) or the LayoutDM [reproducibility guide](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-dm/REPRODUCING.md) to create `.cache/layout-dm/converted/layoutdm-rico25` before loading converted weights.
