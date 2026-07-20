---
tags:
  - Models
  - Documentation
---

# Models

This generated table summarizes model package metadata declared in each `models/*/pyproject.toml`.

| Package | Framework | Task | Conditions | Datasets |
| --- | --- | --- | --- | --- |
| [Coarse-to-Fine](api/models/coarse-to-fine/index.md) | `transformers` | `content-agnostic-layout-generation` | `unconditional` | `rico25`, `publaynet` |
| [LACE](api/models/lace/index.md) | `diffusers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `label_size`, `completion`, `refinement` | `publaynet`, `rico25` |
| [LayouSyn](api/models/layousyn/index.md) | `diffusers` | `content-aware-layout-generation` | `text` | `grit`, `coco-grounded` |
| [Layout-Corrector](api/models/layout-corrector/index.md) | `diffusers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `label_size`, `completion`, `refinement` | `rico25`, `publaynet`, `crello` |
| [LayoutDM](api/models/layout-dm/index.md) | `diffusers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `label_size`, `completion`, `refinement` | `rico25`, `publaynet` |
| [LayoutDiffusion](api/models/layoutdiffusion/index.md) | `diffusers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `refinement` | `rico25`, `publaynet` |
| [LayoutFlow](api/models/layout-flow/index.md) | `diffusers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `label_size`, `completion`, `refinement` | `rico25`, `publaynet` |
| [LayoutFormer++](api/models/layoutformerpp/index.md) | `transformers` | `content-agnostic-layout-generation` | `unconditional`, `label`, `label_size`, `relation`, `completion`, `refinement` | `rico25`, `publaynet` |
| [LayoutGAN++](api/models/layoutganpp/index.md) | `transformers` | `content-agnostic-layout-generation` | `label` | `rico25`, `publaynet`, `magazine` |
| [LayoutGPT](api/models/layout-gpt/index.md) | `pydantic-ai` | `content-aware-layout-generation` | `text`, `unconditional` | `nsr-1k` |
| [LayoutPrompter](api/models/layoutprompter/index.md) | `pydantic-ai` | `content-aware-layout-generation` | `label`, `label_size`, `relation`, `completion`, `refinement`, `text`, `content_image` | `publaynet`, `rico25`, `posterlayout`, `webui` |
| [Parse-Then-Place](api/models/parse-then-place/index.md) | `transformers` | `content-aware-layout-generation` | `text` | `rico25`, `web` |
