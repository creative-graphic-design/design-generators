---
icon: lucide/list-checks
tags:
  - Conventions
  - Contributors
---

# Conventions

This project exposes research implementations through package interfaces that behave like regular Transformers or Diffusers components. Public code should be importable, documented, and runnable without reading the original vendor repository.

For setup and first inference, start with [Getting Started](getting-started/).

## User Interface

These conventions apply to generated outputs and public pipeline arguments.

### Public Outputs

Layout generation APIs return the common schema fields `bbox`, `labels`, `mask`, and `id2label`. Optional fields include `sequences`, `scores`, `trajectory`, and `intermediates`.

Transformers-style APIs return `laygen.modeling_outputs.LayoutGenerationOutput`; Diffusers pipelines return `laygen.pipelines.pipeline_output.LayoutGenerationOutput`. Both output classes are explicit dataclasses with matching field names, order, and defaults.

Transformers-side layout pipelines inherit from `laygen.pipelines.LayoutGenerationPipeline` rather than `transformers.Pipeline`. The shared base handles root [`PretrainedConfig`](https://huggingface.co/docs/transformers/main_classes/configuration) loading, declared subfolder components, [`save_pretrained`](https://huggingface.co/docs/transformers/main_classes/model), `to(device, dtype)`, and generator-over-seed precedence; each model package implements its own `__call__` orchestration and returns the canonical Transformers-style layout output.

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

#### Model catalog classification

`Content` is `content-agnostic` when package inference uses no canvas/background image or saliency input and `content-aware` when it does; element-level image or text feature fields, such as Flex-DM's feature infilling, do not count.

`Task` is classified as follows:

- `task-agnostic`: one trained checkpoint or configuration serves every accepted generation condition by applying the condition at inference; this value requires two or more accepted generation conditions.
- `task-aware`: a separate checkpoint or training run exists for each accepted generation condition; this value requires two or more accepted generation conditions.
- `single-task`: the package accepts exactly one condition; packages whose `conditioning:` value of the model card's `Model type:` line is `none` count as single-task because their only accepted condition is `content_image`.
- `evaluation`: the package evaluates layouts rather than generating them.
- `saliency`: the package predicts saliency rather than generating layouts.

The literature sometimes uses `task-agnostic` for the same single-checkpoint property used by this catalog.

The `conditioning:` value of the model card's `Model type:` line may use these catalog-only values, which are not `condition_type` arguments:

- `evaluation`: evaluate layouts rather than generate them.
- `saliency`: predict saliency as the package's primary public operation.
- `none`: the package's only accepted condition is `content_image`.

The `content_image` condition is excluded from model-card Conditioning because it classifies the Content axis.

### Seeded Sampling

Seeded sampling uses a CPU `torch.Generator` when the caller supplies a seed. The shared `laygen.common.randomness` wrappers (`randn`, `rand`, `randint`, `randperm`, `multinomial`, `bernoulli`, `normal`, and `poisson`) draw on an explicit generator's device, or on the requested output device when no generator is supplied, and then move the result to the requested device and dtype. Explicit generators keep their identity and device, so a CUDA generator remains a CUDA-local stream while a CPU generator gives the same stream for CPU and GPU targets. When a CPU generator samples GPU-resident tensor arguments, `multinomial`, `bernoulli`, `normal`, and `poisson` copy those arguments device-to-host for the CPU draw, synchronizing the stream, and copy the result host-to-device on every draw; this cost is paid on every step of discrete-diffusion and autoregressive decode loops.

### Dataset References

Datasets hosted by the `creative-graphic-design` Hugging Face organization are preferred when they exist. Model processors own dataset-specific loading and normalization details so data sources can be changed without changing model code.

### Framework Selection

- Diffusion and flow-matching models use Diffusers.
- A Diffusers denoiser follows [`ModelMixin`](https://huggingface.co/docs/diffusers/api/models/overview) and [`ConfigMixin`](https://huggingface.co/docs/diffusers/api/configuration).
- A Diffusers noising process follows [`SchedulerMixin`](https://huggingface.co/docs/diffusers/api/schedulers/overview).
- Diffusers generation uses [`DiffusionPipeline`](https://huggingface.co/docs/diffusers/api/diffusion_pipeline).
- Add a repository scheduler when no built-in scheduler expresses the required process.
- Autoregressive, sequence-to-sequence, and GAN models use a Transformers [`PreTrainedModel`](https://huggingface.co/docs/transformers/main_classes/model).
- LLM fine-tuning models reuse existing Hugging Face model classes and prioritize usage recipes plus processors over conversion.
- Methods that call an external language model in context use Pydantic AI agents with typed layout outputs and provider-independent model configuration.

## Contributor Conventions

These conventions apply when adding or maintaining workspace packages.

### Workspace Packages

Shared Transformers-style layout outputs use `laygen.modeling_outputs`, Transformers-side layout pipelines subclass `laygen.pipelines.LayoutGenerationPipeline`, Diffusers pipeline outputs use `laygen.pipelines.pipeline_output`, utility functions use `laygen.common`, neural-network modules live under `laygen.nn`, and scheduler adapters live under `laygen.schedulers`. Poster and content-aware helpers use `posgen.common` when shared code is needed.

Run member-specific commands with the package selected:

```bash
uv run --package <name> pytest
```

Original implementations stay under `vendor/` and are treated as read-only references. Here, vendor means the upstream research repository used to verify conversion behavior. Dependencies needed only for vendor parity belong behind a model package's `vendor` optional extra.

### Data And Parity

Vendor parity fixtures are reference outputs regenerated from the original implementation with fixed seeds. Large tensors, images, model weights, and downloaded datasets are not committed; only metadata needed to regenerate them is committed.

### Documentation

Every `docs/*.md` page carries YAML frontmatter with `icon` and `tags` so the docs site has consistent navigation metadata.

Each model package README follows a model-card style: overview, install and usage snippet, supported checkpoints and Hub ids, datasets, reproducibility summary with vendor-parity numbers, license, citation, and original implementation link.

Each model README includes a `Reproducibility` section that opens with one sentence stating how to reproduce the original-implementation agreement checks, followed by copy-pasteable commands for downloading assets, generating vendor references, running parity tests, converting checkpoints, and running [`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) smoke tests.

Public API docstrings are the source for the API reference. Use google-style docstrings with `Args`, `Returns`, `Raises`, and `Examples` sections. Examples should be runnable doctest-style snippets when the API can run without heavyweight assets.
