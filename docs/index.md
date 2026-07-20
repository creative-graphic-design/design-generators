# design-generators

`design-generators` packages layout, poster, and graphic-design generation methods behind familiar Transformers, Diffusers, and Pydantic AI interfaces. Use it to load locally converted checkpoints or prompt configurations, generate labeled bounding boxes, and compare methods through one public output schema.

## Start Here

- [Getting Started](getting-started.md) shows the workspace install commands and a first local-checkpoint inference snippet.
- [Models](models.md) lists the packaged methods, runtime style, datasets, and planned Hub ids.
- [Conventions](conventions.md) documents output fields, box coordinates, masks, labels, and supported `condition_type` names.
- [API Reference](api/index.md) is generated from workspace packages under `lib/*` and `models/*`.

## Quickstart

Install the LayoutDM workspace member and load a locally converted checkpoint directory. Hub ids are planned until checkpoint publication is complete, so local conversion paths are the copy-paste path.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-dm
uv run --package layout-dm python
```

```python
from layout_dm import LayoutDMPipeline

path = ".cache/layout-dm/converted/layoutdm-rico25"
# After Hub publication: from_pretrained("creative-graphic-design/layoutdm-rico25")
pipe = LayoutDMPipeline.from_pretrained(path)
out = pipe(batch_size=1, seed=0, sampling="deterministic")

print(out.bbox.shape)
print(out.labels.shape)
print(out.id2label[0])
```

```text
torch.Size([1, 25, 4])
torch.Size([1, 25])
Text
```

Run the model package's `REPRODUCING.md` commands first to create the `.cache/.../converted/...` directory used by the snippet.
