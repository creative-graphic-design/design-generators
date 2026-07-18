# design-generators

[![CI](https://github.com/creative-graphic-design/design-generators/actions/workflows/ci.yml/badge.svg)](https://github.com/creative-graphic-design/design-generators/actions/workflows/ci.yml)
[![docs](https://img.shields.io/static/v1?label=docs&message=online&color=brightgreen&style=flat-square)](https://creative-graphic-design.github.io/design-generators/)
![license](https://img.shields.io/static/v1?label=license&message=Apache--2.0&color=green&style=flat-square)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square)
![uv](https://img.shields.io/static/v1?label=uv&message=workspace&color=informational&style=flat-square)
![models](https://img.shields.io/static/v1?label=models&message=12&color=purple&style=flat-square)

`design-generators` ports layout, poster, and graphic-design generation research repositories into Transformers-, Diffusers-, and Pydantic-AI-style packages that can load converted weights or prompt configuration and run inference through a consistent public schema.

## Packages

| Package | Method | Runtime | Primary datasets | Status |
| --- | --- | --- | --- | --- |
| `models/coarse-to-fine` | Coarse-to-Fine | transformers | RICO25, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/lace` | LACE | diffusers | RICO25, RICO13, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/layousyn` | LayouSyn | diffusers | GRIT, COCO grounded | README normalized; Hub publication tracked per model issue |
| `models/layout-corrector` | Layout-Corrector | transformers | RICO25, PubLayNet, Crello | README normalized; Hub publication tracked per model issue |
| `models/layout-dm` | LayoutDM | diffusers | RICO25, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/layout-flow` | LayoutFlow | diffusers | RICO25, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/layout-gpt` | LayoutGPT | pydantic-ai | NSR-1K | README normalized; Hub publication tracked per model issue |
| `models/layoutdiffusion` | LayoutDiffusion | diffusers | RICO25, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/layoutformerpp` | LayoutFormer++ | transformers | RICO25, PubLayNet | README normalized; Hub publication tracked per model issue |
| `models/layoutganpp` | LayoutGAN++ | transformers | RICO25, PubLayNet, Magazine | README normalized; Hub publication tracked per model issue |
| `models/layoutprompter` | LayoutPrompter | pydantic-ai | PubLayNet, RICO25, PosterLayout | README normalized; Hub publication tracked per model issue |
| `models/parse-then-place` | Parse-Then-Place | transformers | RICO25, Web | README normalized; Hub publication tracked per model issue |

Shared libraries live under `lib/*`: `laygen` contains layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers; `posgen` reserves small poster/content-aware placement contracts for future consumers.

## Quick Start

Install one workspace member and run a smoke import from the repository root.

```bash
uv sync --package layout-dm
uv run --package layout-dm python - <<'PY'
from layout_dm import LayoutDMPipeline

print(LayoutDMPipeline.__name__)
PY
```

Converted checkpoint directories and vendor fixtures are generated under `.cache/` by each model README's reproducibility commands. Do not commit downloaded weights, generated tensors, images, or other large artifacts.

## Workspace Commands

```bash
uv sync --all-packages
uv run --package laygen pytest lib/laygen/tests
uv run --package layout-dm pytest models/layout-dm/tests -m "not vendor_parity and not integration"
uv run python scripts/check_model_readmes.py
uv run python scripts/check_readme_badges.py
uv run --group docs mkdocs build --strict
```

Use `uv run --package <member> ...` for member-specific commands so extras, dependency source mappings, and package metadata resolve from the correct workspace member.

## Documentation

The documentation site is published at <https://creative-graphic-design.github.io/design-generators/>. API pages are generated from workspace members below `lib/*/src` and `models/*/src`. Public API docstrings are the source text for the API reference.

## Reproducibility Policy

Model README `## Reproducibility` sections provide copy-pasteable commands for vendor asset download, vendor reference generation, parity tests, checkpoint conversion, and `from_pretrained` or prompt-configuration smoke tests. Prompt-only packages explicitly document the absence of learned checkpoints.

## License

Repository code is licensed under Apache-2.0; see [LICENSE](LICENSE). Converted weights, datasets, and vendored upstream code carry their original licenses.
