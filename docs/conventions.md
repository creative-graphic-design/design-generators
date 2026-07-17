# Conventions

This project exposes research implementations through package interfaces that behave like regular Transformers or Diffusers components. Public code should be importable, documented, and runnable without reading the original vendor repository.

## Workspace Packages

Workspace members live under `lib/*` and `models/*`. Shared layout-generation helpers use the `laygen.common` namespace, while poster and content-aware helpers use `posgen.common` when shared code is needed.

Run member-specific commands with the package selected:

```bash
uv run --package <name> pytest
```

Original implementations stay under `vendor/` and are treated as read-only references. Dependencies needed only for vendor parity belong behind a model package's `vendor` optional extra.

## Public Outputs

Layout generation APIs return the common schema fields `bbox`, `labels`, `mask`, and `id2label`. Optional fields include `sequences`, `scores`, `trajectory`, and `intermediates`.

Public `bbox` values are normalized center `xywh` coordinates in `[0, 1]`, even when vendor code uses `ltwh`, `ltrb`, bins, analog bits, or text tokens internally. Public `mask=True` means a valid element, and padding is represented by `mask` rather than a reserved public label id.

Canonical `condition_type` names are:

- `unconditional`
- `label`
- `label_size`
- `completion`
- `refinement`
- `text`
- `content_image`
- `relation`
- `hierarchical`
- `retrieval`

Unsupported conditions should raise explicit errors. `generator` is the reproducibility API and takes precedence over `seed`.

## Data And Parity

Datasets hosted by the `creative-graphic-design` Hugging Face organization are preferred when they exist. Model processors own dataset-specific loading and normalization details so data sources can be changed without changing model code.

Vendor parity fixtures are regenerated from the original implementation with fixed seeds. Large tensors, images, model weights, and downloaded datasets are not committed; only metadata needed to regenerate them is committed.

## Documentation

Each model package README follows a model-card style: overview, install and usage snippet, supported checkpoints and Hub ids, datasets, vendor-parity summary, license, citation, and original implementation link.

Each model README includes a `Reproducibility` section with copy-pasteable commands for downloading assets, generating original-implementation references, running parity tests, converting checkpoints, and running `from_pretrained` smoke tests.

Public API docstrings are the source for the API reference. Use google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples` sections. Examples should be runnable doctest-style snippets when the API can run without heavyweight assets.
