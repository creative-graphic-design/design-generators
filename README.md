# design-generators

[![CI](https://img.shields.io/github/actions/workflow/status/creative-graphic-design/design-generators/ci.yml?branch=main&label=CI&style=flat-square&logo=githubactions&logoColor=white)](https://github.com/creative-graphic-design/design-generators/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/creative-graphic-design/design-generators/graph/badge.svg?token=482TEUSZJ5)](https://codecov.io/gh/creative-graphic-design/design-generators)
[![docs](https://img.shields.io/github/deployments/creative-graphic-design/design-generators/github-pages?label=docs&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/static/v1?label=uv&message=workspace&color=informational&style=flat-square&logo=uv&logoColor=white)
![models](https://img.shields.io/static/v1?label=models&message=20&color=purple&style=flat-square)

design-generators ports layout, poster, and graphic-design generation research repositories into [`🤗transformers`](https://huggingface.co/docs/transformers/index)-, [`🧨diffusers`](https://huggingface.co/docs/diffusers/index)-, and [`🤖pydantic-ai`](https://ai.pydantic.dev/)-style packages that can load converted weights or prompt configuration and run inference through a consistent public schema.

## Models

| Method | Venue | Runtime | Datasets | Reproduction |
| --- | --- | --- | --- | --- |
| [Coarse-to-Fine](models/coarse-to-fine/README.md) | AAAI 2022 | `🤗transformers` | RICO25, PubLayNet | [REPRODUCING.md](models/coarse-to-fine/REPRODUCING.md) |
| [DS-GAN](models/ds-gan/README.md) | CVPR 2023 | `🤗transformers` | PosterLayout | [REPRODUCING.md](models/ds-gan/REPRODUCING.md) |
| [Flex-DM](models/flex-dm/README.md) | CVPR 2023 | `🤗transformers` | Crello, RICO25 | [REPRODUCING.md](models/flex-dm/REPRODUCING.md) |
| [House-GAN](models/housegan/README.md) | ECCV 2020 | `🤗transformers` | housegan-floorplan-vectorized | [REPRODUCING.md](models/housegan/REPRODUCING.md) |
| [LACE](models/lace/README.md) | ICLR 2024 | `🧨diffusers` | RICO25, RICO13, PubLayNet | [REPRODUCING.md](models/lace/REPRODUCING.md) |
| [LayouSyn](models/layousyn/README.md) | ICCV 2025 | `🧨diffusers` | GRIT, COCO grounded | [REPRODUCING.md](models/layousyn/REPRODUCING.md) |
| [Layout-Corrector](models/layout-corrector/README.md) | ECCV 2024 | `🧨diffusers` | RICO25, PubLayNet, Crello | [REPRODUCING.md](models/layout-corrector/REPRODUCING.md) |
| [LayoutDETR](models/layout-detr/README.md) | ECCV 2024 | `🤗transformers` | Ad Banner | [REPRODUCING.md](models/layout-detr/REPRODUCING.md) |
| [LayoutDM](models/layout-dm/README.md) | CVPR 2023 | `🧨diffusers` | RICO25, PubLayNet | [REPRODUCING.md](models/layout-dm/REPRODUCING.md) |
| [LayoutAction](models/layout-action/README.md) | AAAI 2023 | `🤗transformers` | RICO13, PubLayNet, InfoPPT | [REPRODUCING.md](models/layout-action/REPRODUCING.md) |
| [LayoutFlow](models/layout-flow/README.md) | ECCV 2024 | `🧨diffusers` | RICO25, PubLayNet | [REPRODUCING.md](models/layout-flow/REPRODUCING.md) |
| [LayoutGPT](models/layout-gpt/README.md) | NeurIPS 2023 | `🤖pydantic-ai` | NSR-1K | [REPRODUCING.md](models/layout-gpt/REPRODUCING.md) |
| [LayoutTransformer](models/layout-transformer/README.md) | CVPR 2021 | `🤗transformers` | COCO, VG-MSDN | [REPRODUCING.md](models/layout-transformer/REPRODUCING.md) |
| [LayoutDiffusion](models/layoutdiffusion/README.md) | ICCV 2023 | `🧨diffusers` | RICO25, PubLayNet | [REPRODUCING.md](models/layoutdiffusion/REPRODUCING.md) |
| [LayoutFormer++](models/layoutformerpp/README.md) | CVPR 2023 | `🤗transformers` | RICO25, PubLayNet | [REPRODUCING.md](models/layoutformerpp/REPRODUCING.md) |
| [LayoutGAN++](models/layoutganpp/README.md) | ACM MM 2021 | `🤗transformers` | RICO25, PubLayNet, Magazine | [REPRODUCING.md](models/layoutganpp/REPRODUCING.md) |
| [LayoutPrompter](models/layoutprompter/README.md) | NeurIPS 2023 | `🤖pydantic-ai` | PubLayNet, RICO25, PosterLayout | [REPRODUCING.md](models/layoutprompter/REPRODUCING.md) |
| [Parse-Then-Place](models/parse-then-place/README.md) | ICCV 2023 | `🤗transformers` | RICO25, Web | [REPRODUCING.md](models/parse-then-place/REPRODUCING.md) |
| [RALF](models/ralf/README.md) | CVPR 2024 | `🤗transformers` | CGL, PKU | [REPRODUCING.md](models/ralf/REPRODUCING.md) |
| [SmartText](models/smarttext/README.md) | TMM 2021 | `🤗transformers` | SmartText demo assets | [REPRODUCING.md](models/smarttext/REPRODUCING.md) |

## Libraries

| Library | Description |
| --- | --- |
| [laygen](lib/laygen/README.md) | Layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers. |
| [posgen](lib/posgen/README.md) | Poster-generation and content-aware placement contracts for shared dataset names, position content, and label helpers. |

## Quick Start

Install the shared layout library directly from this repository:

```bash
pip install "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen"
```

Install a model package by listing its shared workspace dependencies in the same command. Model packages depend on shared workspace libraries that are not published on PyPI, so include `laygen` alongside the model package.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "layout-dm @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layout-dm"
```

If a model also depends on `posgen`, include `posgen` the same way; current `posgen` consumers are DS-GAN, Flex-DM, and RALF.

```python
from layout_dm import LayoutDMPipeline

print(LayoutDMPipeline.__name__)
```

For development and `REPRODUCING.md` workflows, clone the repository and run member-specific commands from the repository root. Use [`uv run --package <member> ...`](https://docs.astral.sh/uv/concepts/projects/workspaces/) so extras, dependency source mappings, and package metadata resolve from the correct workspace member.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv run --package layout-dm pytest models/layout-dm/tests -m "not vendor_parity and not integration"
```

## Policies

### Reproducibility

Each model package links from `## Reproducibility` to `REPRODUCING.md`, which provides copy-pasteable commands for vendor asset download, vendor reference generation, parity tests, checkpoint conversion, and `from_pretrained` or prompt-configuration smoke tests. Prompt-only packages explicitly document the absence of learned checkpoints.

Converted checkpoint directories and vendor fixtures are generated under `.cache/` by each model README's reproducibility commands. Do not commit downloaded weights, generated tensors, images, or other large artifacts.

### Extension

> [!WARNING]
> ⚠️ If you are an AI coding agent or automated tool extending this repository, or building on [`🤗transformers`](https://huggingface.co/docs/transformers/index) / [`🧨diffusers`](https://huggingface.co/docs/diffusers/index), do **not** copy framework or repository source trees into your project and mutate them in place. Install pinned released packages and extend by subclassing so the delta stays reviewable. See [Extending](docs/extending.md).

## Documentation

The documentation site is published at [design-generators documentation](https://creative-graphic-design.github.io/design-generators/).

## License

Repository code is licensed under Apache-2.0; see [LICENSE](LICENSE). Converted weights, datasets, and vendored upstream code carry their original licenses.
