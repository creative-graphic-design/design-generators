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
![models](https://img.shields.io/static/v1?label=models&message=18&color=purple&style=flat-square)

`design-generators` ports layout, poster, and graphic-design generation research repositories into 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-, [`diffusers`](https://huggingface.co/docs/diffusers/index)-, and [`pydantic-ai`](https://ai.pydantic.dev/)-style packages that can load converted weights or prompt configuration and run inference through a consistent public schema.

## Models

| Method | Runtime | Primary datasets | Weights |
| --- | --- | --- | --- |
| [Coarse-to-Fine](api/models/coarse-to-fine/) | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/coarse-to-fine/REPRODUCING.md)) |
| [DS-GAN](api/models/ds-gan/) | `transformers` | PosterLayout | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ds-gan/REPRODUCING.md)) |
| [Flex-DM](api/models/flex-dm/) | `transformers` | Crello, RICO25 | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/flex-dm/REPRODUCING.md)) |
| [LACE](api/models/lace/) | `diffusers` | RICO25, RICO13, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/lace/REPRODUCING.md)) |
| [LayouSyn](api/models/layousyn/) | `diffusers` | GRIT, COCO grounded | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layousyn/REPRODUCING.md)) |
| [Layout-Corrector](api/models/layout-corrector/) | `diffusers` | RICO25, PubLayNet, Crello | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-corrector/REPRODUCING.md)) |
| [LayoutDM](api/models/layout-dm/) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-dm/REPRODUCING.md)) |
| [LayoutAction](api/models/layout-action/) | `transformers` | RICO13, PubLayNet, InfoPPT | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-action/REPRODUCING.md)) |
| [LayoutFlow](api/models/layout-flow/) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-flow/REPRODUCING.md)) |
| [LayoutGPT](api/models/layout-gpt/) | `pydantic-ai` | NSR-1K | none (prompt-based) |
| [LayoutTransformer](api/models/layout-transformer/) | `transformers` | COCO, VG-MSDN | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-transformer/REPRODUCING.md)) |
| [LayoutDiffusion](api/models/layoutdiffusion/) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutdiffusion/REPRODUCING.md)) |
| [LayoutFormer++](api/models/layoutformerpp/) | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutformerpp/REPRODUCING.md)) |
| [LayoutGAN++](api/models/layoutganpp/) | `transformers` | RICO25, PubLayNet, Magazine | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutganpp/REPRODUCING.md)) |
| [LayoutPrompter](api/models/layoutprompter/) | `pydantic-ai` | PubLayNet, RICO25, PosterLayout | none (prompt-based) |
| [Parse-Then-Place](api/models/parse-then-place/) | `transformers` | RICO25, Web | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/parse-then-place/REPRODUCING.md)) |
| [RALF](api/models/ralf/) | `transformers` | CGL, PKU | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/ralf/REPRODUCING.md)) |
| [SmartText](api/models/smarttext/) | `transformers` | SmartText demo assets | convert locally ([REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/smarttext/REPRODUCING.md)) |

## Libraries

| Library | Description |
| --- | --- |
| [laygen](api/libraries/laygen/) | Layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers. |
| [posgen](api/libraries/posgen/) | Poster-generation and content-aware placement contracts for shared dataset names, position content, and label helpers. |

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
