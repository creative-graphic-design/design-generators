# Conventions

This project exposes research implementations through package interfaces that behave like regular Transformers or Diffusers components. Public code should be importable, documented, and runnable without reading the original vendor repository.

For setup and first inference, start with [Getting Started](getting-started.md).

## User Interface

These conventions apply to generated outputs and public pipeline arguments.

### Public Outputs

Layout generation APIs return the common schema fields `bbox`, `labels`, `mask`, and `id2label`. Optional fields include `sequences`, `scores`, `trajectory`, and `intermediates`.

Transformers-style APIs return `laygen.modeling_outputs.LayoutGenerationOutput`; Diffusers pipelines return `laygen.pipelines.pipeline_output.LayoutGenerationOutput`. Both output classes are explicit dataclasses with matching field names, order, and defaults.

Transformers-side layout pipelines inherit from `laygen.pipelines.LayoutGenerationPipeline` rather than `transformers.Pipeline`. The shared base handles root `PretrainedConfig` loading, declared subfolder components, `save_pretrained`, `to(device, dtype)`, and generator-over-seed precedence; each model package implements its own `__call__` orchestration and returns the canonical Transformers-style layout output.

Public `bbox` values are normalized center `xywh` coordinates in `[0, 1]`, even when vendor code uses `ltwh`, `ltrb`, bins, analog bits, or text tokens internally. Public `mask=True` means a valid element, and padding is represented by `mask` rather than a reserved public label id.

```python
import torch
from laygen.modeling_outputs import LayoutGenerationOutput

out = LayoutGenerationOutput(
    bbox=torch.tensor([[[0.50, 0.50, 0.25, 0.20]]]),
    labels=torch.tensor([[0]]),
    mask=torch.tensor([[True]]),
    id2label={0: "Text"},
)

print(out.bbox.shape)
print(out.id2label[int(out.labels[0, 0])])
```

```text
torch.Size([1, 1, 4])
Text
```

### Conditioning

Canonical `condition_type` names are:

- `unconditional`: generate a layout without user-provided element constraints.
- `label`: condition on element categories.
- `label_size`: condition on element categories and sizes.
- `completion`: fill missing elements around a partially observed layout.
- `refinement`: improve or denoise an existing layout.
- `text`: condition on a natural-language prompt.
- `content_image`: condition on an image or saliency/content representation.
- `relation`: condition on pairwise or graph-style element relations.
- `hierarchical`: condition on a tree or grouped layout structure.
- `retrieval`: condition on retrieved exemplar layouts or records.

Unsupported conditions should raise explicit errors. `generator` is the reproducibility API and takes precedence over `seed`.

### Dataset References

Datasets hosted by the `creative-graphic-design` Hugging Face organization are preferred when they exist. Model processors own dataset-specific loading and normalization details so data sources can be changed without changing model code.

## Contributor Conventions

These conventions apply when adding or maintaining workspace packages.

### Workspace Packages

Workspace members live under `lib/*` and `models/*`. Shared Transformers-style layout outputs use `laygen.modeling_outputs`, Transformers-side layout pipelines subclass `laygen.pipelines.LayoutGenerationPipeline`, Diffusers pipeline outputs use `laygen.pipelines.pipeline_output`, utility functions use `laygen.common`, neural-network modules live under `laygen.nn`, and scheduler adapters live under `laygen.schedulers`. Poster and content-aware helpers use `posgen.common` when shared code is needed.

Run member-specific commands with the package selected:

```bash
uv run --package <name> pytest
```

Original implementations stay under `vendor/` and are treated as read-only references. Here, vendor means the upstream research repository used to verify conversion behavior. Dependencies needed only for vendor parity belong behind a model package's `vendor` optional extra.

### Data And Parity

Vendor parity fixtures are reference outputs regenerated from the original implementation with fixed seeds. Large tensors, images, model weights, and downloaded datasets are not committed; only metadata needed to regenerate them is committed.

### Documentation

Each model package README follows a model-card style: overview, install and usage snippet, supported checkpoints and Hub ids, datasets, reproducibility summary with vendor-parity numbers, license, citation, and original implementation link.

Each model README includes a `Reproducibility` section that opens with one sentence stating how to reproduce the original-implementation agreement checks, followed by copy-pasteable commands for downloading assets, generating vendor references, running parity tests, converting checkpoints, and running `from_pretrained` smoke tests.

Public API docstrings are the source for the API reference. Use google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples` sections. Examples should be runnable doctest-style snippets when the API can run without heavyweight assets.
