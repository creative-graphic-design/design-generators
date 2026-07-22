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
| [Coarse-to-Fine](models/coarse-to-fine/README.md) | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](models/coarse-to-fine/REPRODUCING.md)) |
| [DS-GAN](models/ds-gan/README.md) | `transformers` | PosterLayout | convert locally ([REPRODUCING.md](models/ds-gan/REPRODUCING.md)) |
| [Flex-DM](models/flex-dm/README.md) | `transformers` | Crello, RICO25 | convert locally ([REPRODUCING.md](models/flex-dm/REPRODUCING.md)) |
| [LACE](models/lace/README.md) | `diffusers` | RICO25, RICO13, PubLayNet | convert locally ([REPRODUCING.md](models/lace/REPRODUCING.md)) |
| [LayouSyn](models/layousyn/README.md) | `diffusers` | GRIT, COCO grounded | convert locally ([REPRODUCING.md](models/layousyn/REPRODUCING.md)) |
| [Layout-Corrector](models/layout-corrector/README.md) | `diffusers` | RICO25, PubLayNet, Crello | convert locally ([REPRODUCING.md](models/layout-corrector/REPRODUCING.md)) |
| [LayoutDM](models/layout-dm/README.md) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](models/layout-dm/REPRODUCING.md)) |
| [LayoutAction](models/layout-action/README.md) | `transformers` | RICO13, PubLayNet, InfoPPT | convert locally ([REPRODUCING.md](models/layout-action/REPRODUCING.md)) |
| [LayoutFlow](models/layout-flow/README.md) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](models/layout-flow/REPRODUCING.md)) |
| [LayoutGPT](models/layout-gpt/README.md) | `pydantic-ai` | NSR-1K | none (prompt-based) |
| [LayoutTransformer](models/layout-transformer/README.md) | `transformers` | COCO, VG-MSDN | convert locally ([REPRODUCING.md](models/layout-transformer/REPRODUCING.md)) |
| [LayoutDiffusion](models/layoutdiffusion/README.md) | `diffusers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](models/layoutdiffusion/REPRODUCING.md)) |
| [LayoutFormer++](models/layoutformerpp/README.md) | `transformers` | RICO25, PubLayNet | convert locally ([REPRODUCING.md](models/layoutformerpp/REPRODUCING.md)) |
| [LayoutGAN++](models/layoutganpp/README.md) | `transformers` | RICO25, PubLayNet, Magazine | convert locally ([REPRODUCING.md](models/layoutganpp/REPRODUCING.md)) |
| [LayoutPrompter](models/layoutprompter/README.md) | `pydantic-ai` | PubLayNet, RICO25, PosterLayout | none (prompt-based) |
| [Parse-Then-Place](models/parse-then-place/README.md) | `transformers` | RICO25, Web | convert locally ([REPRODUCING.md](models/parse-then-place/REPRODUCING.md)) |
| [RALF](models/ralf/README.md) | `transformers` | CGL, PKU | convert locally ([REPRODUCING.md](models/ralf/REPRODUCING.md)) |
| [SmartText](models/smarttext/README.md) | `transformers` | SmartText demo assets | convert locally ([REPRODUCING.md](models/smarttext/REPRODUCING.md)) |

## Libraries

| Library | Description |
| --- | --- |
| [laygen](lib/laygen/README.md) | Layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers. |
| [posgen](lib/posgen/README.md) | Poster-generation and content-aware placement contracts for shared dataset names, position content, and label helpers. |

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

Repository code is licensed under Apache-2.0; see [LICENSE](LICENSE). Converted weights, datasets, and vendored upstream code carry their original licenses.
