---
icon: lucide/layout-template
tags:
  - Overview
  - Documentation
---

# design-generators

[![CI](https://img.shields.io/github/actions/workflow/status/creative-graphic-design/design-generators/ci.yml?branch=main&label=CI&style=flat-square&logo=githubactions&logoColor=white)](https://github.com/creative-graphic-design/design-generators/actions/workflows/ci.yml)
[![docs](https://img.shields.io/github/deployments/creative-graphic-design/design-generators/github-pages?label=docs&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/static/v1?label=uv&message=workspace&color=informational&style=flat-square&logo=uv&logoColor=white)
![models](https://img.shields.io/static/v1?label=models&message=16&color=purple&style=flat-square)

`design-generators` ports layout, poster, and graphic-design generation research repositories into 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-, [`diffusers`](https://huggingface.co/docs/diffusers/index)-, and [`pydantic-ai`](https://ai.pydantic.dev/)-style packages that can load converted weights or prompt configuration and run inference through a consistent public schema.

## Models

| Model | Method | Runtime | Primary datasets | Weights |
| --- | --- | --- | --- | --- |
| [`models/coarse-to-fine`](api/models/coarse-to-fine/index.md) | Coarse-to-Fine | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/coarse-to-fine/REPRODUCING.md)) |
| [`models/ds-gan`](api/models/ds-gan/index.md) | DS-GAN | `transformers` | PosterLayout | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/REPRODUCING.md)) |
| [`models/lace`](api/models/lace/index.md) | LACE | `diffusers` | RICO25, RICO13, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/lace/REPRODUCING.md)) |
| [`models/layousyn`](api/models/layousyn/index.md) | LayouSyn | `diffusers` | GRIT, COCO grounded | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layousyn/REPRODUCING.md)) |
| [`models/layout-corrector`](api/models/layout-corrector/index.md) | Layout-Corrector | `diffusers` | RICO25, PubLayNet, Crello | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-corrector/REPRODUCING.md)) |
| [`models/layout-dm`](api/models/layout-dm/index.md) | LayoutDM | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-dm/REPRODUCING.md)) |
| [`models/layout-action`](api/models/layout-action/index.md) | LayoutAction | `transformers` | RICO13, PubLayNet, InfoPPT | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-action/REPRODUCING.md)) |
| [`models/layout-flow`](api/models/layout-flow/index.md) | LayoutFlow | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-flow/REPRODUCING.md)) |
| [`models/layout-gpt`](api/models/layout-gpt/index.md) | LayoutGPT | `pydantic-ai` | NSR-1K | none (prompt-based) |
| [`models/layout-transformer`](api/models/layout-transformer/index.md) | LayoutTransformer | `transformers` | COCO, VG-MSDN | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-transformer/REPRODUCING.md)) |
| [`models/layoutdiffusion`](api/models/layoutdiffusion/index.md) | LayoutDiffusion | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutdiffusion/REPRODUCING.md)) |
| [`models/layoutformerpp`](api/models/layoutformerpp/index.md) | LayoutFormer++ | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutformerpp/REPRODUCING.md)) |
| [`models/layoutganpp`](api/models/layoutganpp/index.md) | LayoutGAN++ | `transformers` | RICO25, PubLayNet, Magazine | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutganpp/REPRODUCING.md)) |
| [`models/layoutprompter`](api/models/layoutprompter/index.md) | LayoutPrompter | `pydantic-ai` | PubLayNet, RICO25, PosterLayout | none (prompt-based) |
| [`models/parse-then-place`](api/models/parse-then-place/index.md) | Parse-Then-Place | `transformers` | RICO25, Web | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/parse-then-place/REPRODUCING.md)) |
| [`models/ralf`](api/models/ralf/index.md) | RALF | `transformers` | CGL, PKU | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ralf/REPRODUCING.md)) |

Shared libraries live under `lib/*`: [`laygen`](api/libraries/laygen/index.md) contains layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers; [`posgen`](api/libraries/posgen/index.md) reserves small poster/content-aware placement contracts for future consumers.

## Quick Start

Install one workspace member and run a smoke import from the repository root.

```bash
uv sync --package layout-dm
uv run --package layout-dm python
```

```python
from layout_dm import LayoutDMPipeline

print(LayoutDMPipeline.__name__)
```

Converted checkpoint directories and vendor fixtures are generated under `.cache/` by each model README's reproducibility commands. Do not commit downloaded weights, generated tensors, images, or other large artifacts.

## Workspace Commands

```bash
uv sync --all-packages
uv run --package laygen pytest lib/laygen/tests
uv run --package layout-dm pytest models/layout-dm/tests -m "not vendor_parity and not integration"
uv run python scripts/check_model_readmes.py
uv run python scripts/check_readme_badges.py
uv run --group docs python scripts/gen_ref_pages.py
uv run --group docs zensical build --strict -f mkdocs.generated.yml
```

Use [`uv run --package <member> ...`](https://docs.astral.sh/uv/concepts/projects/workspaces/) for member-specific commands so extras, dependency source mappings, and package metadata resolve from the correct workspace member.

## Documentation

The documentation site is published at [design-generators documentation](https://creative-graphic-design.github.io/design-generators/). API pages are generated from workspace members below `lib/*/src` and `models/*/src`. Public API docstrings are the source text for the API reference.

## Reproducibility Policy

Each model package links from `## Reproducibility` to `REPRODUCING.md`, which provides copy-pasteable commands for vendor asset download, vendor reference generation, parity tests, checkpoint conversion, and `from_pretrained` or prompt-configuration smoke tests. Prompt-only packages explicitly document the absence of learned checkpoints.

## License

Repository code is licensed under Apache-2.0; see [LICENSE](https://github.com/creative-graphic-design/design-generators/blob/main/LICENSE). Converted weights, datasets, and vendored upstream code carry their original licenses.
