# design-generators

[![CI](https://img.shields.io/github/actions/workflow/status/creative-graphic-design/design-generators/ci.yml?branch=main&label=CI&style=flat-square&logo=githubactions&logoColor=white)](https://github.com/creative-graphic-design/design-generators/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/creative-graphic-design/design-generators/graph/badge.svg?token=482TEUSZJ5)](https://codecov.io/gh/creative-graphic-design/design-generators)
[![docs](https://img.shields.io/github/deployments/creative-graphic-design/design-generators/github-pages?label=docs&style=flat-square&logo=readthedocs&logoColor=white)](https://creative-graphic-design.github.io/design-generators/)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![python](https://img.shields.io/static/v1?label=python&message=%3E%3D3.11&color=blue&style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/static/v1?label=uv&message=workspace&color=informational&style=flat-square&logo=uv&logoColor=white)
![models](https://img.shields.io/static/v1?label=models&message=29&color=purple&style=flat-square)

design-generators ports layout, poster, and graphic-design generation research repositories into framework-specific packages for [`🤗transformers`](https://huggingface.co/docs/transformers/index), [`🧨diffusers`](https://huggingface.co/docs/diffusers/index), and [`🤖pydantic-ai`](https://ai.pydantic.dev/) that can load converted weights or prompt configuration and run inference through a consistent public schema.

## Models

Framework, task, and dataset details are generated in the [Models documentation](https://creative-graphic-design.github.io/design-generators/models/).

| Model | Venue | Ckpt | Train |
| --- | --- | --- | --- |
| [`BASNet`](models/basnet/README.md) | ![venue: CVPR 2019](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202019&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/basnet/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`CGB-DM`](models/cgb-dm/README.md) | ![venue: arXiv 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=arXiv%202024&color=b31b1b) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/cgb-dm/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/cgb-dm/TRAINING.md) |
| [`Coarse-to-Fine`](models/coarse-to-fine/README.md) | ![venue: AAAI 2022](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=AAAI%202022&color=2f5f8f) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/coarse-to-fine/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`DLT`](models/dlt/README.md) | ![venue: ICCV 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICCV%202023&color=0066cc) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/dlt/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/dlt/TRAINING.md) |
| [`DS-GAN`](models/ds-gan/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/ds-gan/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`Flex-DM`](models/flex-dm/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/flex-dm/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`House-GAN`](models/housegan/README.md) | ![venue: ECCV 2020](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202020&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/housegan/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LACE`](models/lace/README.md) | ![venue: ICLR 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICLR%202024&color=00a88f) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/lace/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayouSyn`](models/layousyn/README.md) | ![venue: ICCV 2025](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICCV%202025&color=0066cc) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layousyn/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`Layout-Corrector`](models/layout-corrector/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-corrector/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutDETR`](models/layout-detr/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-detr/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutDM`](models/layout-dm/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layout-dm/TRAINING.md) |
| [`Layout FID`](models/layout-fid/README.md) | ![venue: ACM MM 2021](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ACM%20MM%202021&color=0085ca) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-fid/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutAction`](models/layout-action/README.md) | ![venue: AAAI 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=AAAI%202023&color=2f5f8f) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-action/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutFlow`](models/layout-flow/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-flow/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layout-flow/TRAINING.md) |
| [`LayoutGPT`](models/layout-gpt/README.md) | ![venue: NeurIPS 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=NeurIPS%202023&color=4b2e83) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-gpt/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LT-Net`](models/ltnet/README.md) | ![venue: CVPR 2021](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202021&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/ltnet/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutDiffusion`](models/layoutdiffusion/README.md) | ![venue: ICCV 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICCV%202023&color=0066cc) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutdiffusion/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layoutdiffusion/TRAINING.md) |
| [`LayoutFormer++`](models/layoutformerpp/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutformerpp/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutGAN++`](models/layoutganpp/README.md) | ![venue: ACM MM 2021](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ACM%20MM%202021&color=0085ca) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutganpp/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutVAE`](models/layoutvae/README.md) | ![venue: ICCV 2019](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICCV%202019&color=0066cc) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutvae/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`LayoutPrompter`](models/layoutprompter/README.md) | ![venue: NeurIPS 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=NeurIPS%202023&color=4b2e83) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutprompter/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`Parse-Then-Place`](models/parse-then-place/README.md) | ![venue: ICCV 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ICCV%202023&color=0066cc) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/parse-then-place/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`PosterLlama`](models/posterllama/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/posterllama/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`PosterLLaVA`](models/posterllava/README.md) | ![venue: TMM 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=TMM%202024&color=purple) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/posterllava/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`PosterO`](models/postero/README.md) | ![venue: CVPR 2025](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202025&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/postero/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`RADM`](models/radm/README.md) | ![venue: CIKM 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CIKM%202023&color=6b7280) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/radm/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`RALF`](models/ralf/README.md) | ![venue: CVPR 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202024&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/ralf/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |
| [`SmartText`](models/smarttext/README.md) | ![venue: TMM 2021](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=TMM%202021&color=00629b) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/smarttext/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries

| Library                                                                                                                                                | Description                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| [![library: laygen](https://img.shields.io/static/v1?label=%F0%9F%93%A6&message=laygen&color=2f80ed)](lib/laygen/README.md)                            | Layout-generation schemas, pipeline helpers, bbox utilities, schedulers, model-card helpers, and testing helpers.      |
| [![library: posgen](https://img.shields.io/static/v1?label=%F0%9F%93%A6&message=posgen&color=00a88f)](lib/posgen/README.md)                            | Poster-generation and content-aware placement contracts for shared dataset names, position content, and label helpers. |
| [![library: traingen](https://img.shields.io/static/v1?label=%F0%9F%93%A6&message=traingen&color=27ae60)](lib/traingen/README.md)                      | Training utilities for package-local PyTorch Lightning CLI integration in train-ourselves packages.                    |
| [![library: traingen-parity](https://img.shields.io/static/v1?label=%F0%9F%93%A6&message=traingen-parity&color=9b51e0)](lib/traingen-parity/README.md) | Deterministic trace capture and comparison helpers for training-parity checks.                                         |

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

For development and `REPRODUCING.md` workflows, clone the repository and run member-specific commands from the repository root. Use [`uv run --package <member> ...`](https://docs.astral.sh/uv/concepts/projects/workspaces/) so extras, dependency source mappings, and package metadata resolve from the correct workspace member. Use the root `evaluation` extra when running scripts that load the org's Hugging Face `evaluate` layout-metric modules; `scripts/verify_evaluate_layout_metrics.py` checks the 12 org metric repos against the layout-dm and PosterLLaMA/PKU vendor implementations.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv run --package layout-dm pytest models/layout-dm/tests -m "not vendor_parity and not integration"
```

## Policies

### Reproducibility

Model-specific reproduction commands live in each package's `REPRODUCING.md`, covering source asset download, reference generation, parity tests, checkpoint conversion, and `from_pretrained` or prompt-configuration smoke tests. Prompt-only packages explicitly document the absence of learned checkpoints.

Converted checkpoint directories and reference fixtures are generated under `.cache/` by each model README's reproducibility commands. Do not commit downloaded weights, generated tensors, images, or other large artifacts.

### Training

For models we reproduce by training ourselves, each package provides [`PyTorch Lightning`](https://lightning.ai/docs/pytorch/stable/) modules, [`LightningCLI`](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) configs, and package-specific training commands. Shared training helpers live in `lib/traingen`, parity helpers live in `lib/traingen-parity`, and runnable workflows are documented in each package's `TRAINING.md`.

### Extension

> [!WARNING]
> ⚠️ If you are an AI coding agent or automated tool extending this repository, or building on [`🤗transformers`](https://huggingface.co/docs/transformers/index) / [`🧨diffusers`](https://huggingface.co/docs/diffusers/index), do **not** copy framework or repository source trees into your project and mutate them in place. Install pinned released packages and extend by subclassing so the delta stays reviewable. See [Extending](docs/extending.md).

## Documentation

See the [design-generators documentation](https://creative-graphic-design.github.io/design-generators/) for guides and the generated API reference for workspace libraries and model packages.

## License

Repository code is licensed under Apache-2.0; see [LICENSE](LICENSE). Converted weights, datasets, and vendored upstream code carry their original licenses.
