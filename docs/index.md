---
icon: lucide/house
tags:
  - Overview
  - Documentation
---

# design-generators

design-generators ports layout, poster, and graphic-design research implementations into installable packages with consistent Transformers-, Diffusers-, and Pydantic AI-style interfaces.

## Models

This catalog groups each package by the task family it supports, links the original research paper, and points to the package's generated API reference.

| Model | Task family | Original paper | API reference |
| --- | --- | --- | --- |
| BASNet | Saliency detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html) | [API](api/models/basnet/) |
| CGB-DM | Poster layout generation | [Paper](https://arxiv.org/abs/2407.15233) | [API](api/models/cgb-dm/) |
| Coarse-to-Fine | Layout generation | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19994) | [API](api/models/coarse-to-fine/) |
| DLT | Layout generation | [Paper](https://arxiv.org/abs/2303.03755) | [API](api/models/dlt/) |
| DS-GAN | Poster layout generation | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html) | [API](api/models/ds-gan/) |
| Flex-DM | Layout generation | [Paper](https://arxiv.org/abs/2303.18248) | [API](api/models/flex-dm/) |
| House-GAN | Floorplan generation | [Paper](https://arxiv.org/abs/2003.06988) | [API](api/models/housegan/) |
| LACE | Layout generation | [Paper](https://openreview.net/forum?id=kJ0qp9Xdsh) | [API](api/models/lace/) |
| LayouSyn | Layout generation | [Paper](https://arxiv.org/abs/2505.04718) | [API](api/models/layousyn/) |
| LayoutAction | Layout generation | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26277) | [API](api/models/layout-action/) |
| Layout-Corrector | Layout generation | [Paper](https://arxiv.org/abs/2409.16689) | [API](api/models/layout-corrector/) |
| LayoutDETR | Content-image layout generation | [Paper](https://arxiv.org/abs/2212.09877) | [API](api/models/layout-detr/) |
| LayoutDM | Layout generation | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html) | [API](api/models/layout-dm/) |
| Layout FID | Layout evaluation | [Paper](https://arxiv.org/abs/2108.00871) | [API](api/models/layout-fid/) |
| LayoutFlow | Layout generation | [Paper](https://arxiv.org/abs/2403.18187) | [API](api/models/layout-flow/) |
| LayoutGPT | Prompt-based layout generation | [Paper](https://arxiv.org/abs/2305.15393) | [API](api/models/layout-gpt/) |
| LayoutDiffusion | Layout generation | [Paper](https://arxiv.org/abs/2303.11589) | [API](api/models/layoutdiffusion/) |
| LayoutFormer++ | Layout generation | [Paper](https://arxiv.org/abs/2208.08037) | [API](api/models/layoutformerpp/) |
| LayoutGAN++ | Layout generation | [Paper](https://doi.org/10.1145/3474085.3475497) | [API](api/models/layoutganpp/) |
| LayoutPrompter | Prompt-based layout generation | [Paper](https://arxiv.org/abs/2311.06495) | [API](api/models/layoutprompter/) |
| LayoutVAE | Layout generation | [Paper](https://arxiv.org/abs/1907.10719) | [API](api/models/layoutvae/) |
| LT-Net | Scene-graph-to-layout generation | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html) | [API](api/models/ltnet/) |
| Parse-Then-Place | Layout generation | [Paper](https://arxiv.org/abs/2308.12700) | [API](api/models/parse-then-place/) |
| PosterLLaVA | Poster layout generation | [Paper](https://arxiv.org/abs/2406.02884) | [API](api/models/posterllava/) |
| PosterLlama | Poster layout generation | [Paper](https://arxiv.org/abs/2404.00995) | [API](api/models/posterllama/) |
| PosterO | Poster layout generation | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Hsu_PosterO_Structuring_Layout_Trees_to_Enable_Language_Models_in_Generalized_CVPR_2025_paper.html) | [API](api/models/postero/) |
| RALF | Retrieval-augmented layout generation | [Paper](https://arxiv.org/abs/2311.13602) | [API](api/models/ralf/) |
| SmartText | Text placement | [Paper](https://ieeexplore.ieee.org/document/9520053) | [API](api/models/smarttext/) |

The [Models](models/) page links each package's README, reproduction guide, and training guide when available.

## Getting started

[Getting Started](getting-started/) explains how to install a workspace package and run a first inference with a converted checkpoint.

## How the repository is organized

[Architecture](architecture/) explains the boundary between shared libraries and model packages, while [Conventions](conventions/) defines the common public output schema and conditioning names.

## API reference

The [API Reference](api/) documents the public package trees for the shared libraries and model packages.
