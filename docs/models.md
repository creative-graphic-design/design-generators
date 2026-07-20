# Models

Weight-backed packages lead with local converted checkpoint paths until the planned Hub repos are published. Prompt-only packages save and load prompt configuration instead of learned weights.

| Package | Method | Runtime | Primary datasets | Planned Hub ids |
| --- | --- | --- | --- | --- |
| [`coarse-to-fine`](api/models/coarse-to-fine/index.md) | Coarse-to-Fine | Transformers | RICO25, PubLayNet | `creative-graphic-design/coarse-to-fine-rico25`, `creative-graphic-design/coarse-to-fine-publaynet` |
| [`ds-gan`](api/models/ds-gan/index.md) | DS-GAN | Transformers | PosterLayout | `creative-graphic-design/ds-gan-pku-posterlayout` |
| [`lace`](api/models/lace/index.md) | LACE | Diffusers | RICO25, RICO13, PubLayNet | `creative-graphic-design/lace-rico25`, `creative-graphic-design/lace-rico13`, `creative-graphic-design/lace-publaynet` |
| [`layousyn`](api/models/layousyn/index.md) | LayouSyn | Diffusers | GRIT, COCO grounded | `creative-graphic-design/layousyn-grit`, `creative-graphic-design/layousyn-coco-grounded`, `creative-graphic-design/layousyn-grit-ft-coco-grounded` |
| [`layout-corrector`](api/models/layout-corrector/index.md) | Layout-Corrector | Diffusers | RICO25, PubLayNet, Crello | `creative-graphic-design/layout-corrector-rico25`, `creative-graphic-design/layout-corrector-publaynet`, `creative-graphic-design/layout-corrector-crello` |
| [`layout-dm`](api/models/layout-dm/index.md) | LayoutDM | Diffusers | RICO25, PubLayNet | `creative-graphic-design/layoutdm-rico25`, `creative-graphic-design/layoutdm-publaynet` |
| [`layout-flow`](api/models/layout-flow/index.md) | LayoutFlow | Diffusers | RICO25, PubLayNet | `creative-graphic-design/layout-flow-rico25`, `creative-graphic-design/layout-flow-publaynet` |
| [`layout-gpt`](api/models/layout-gpt/index.md) | LayoutGPT | Pydantic AI | NSR-1K | prompt configuration, no learned checkpoint |
| [`layout-transformer`](api/models/layout-transformer/index.md) | LayoutTransformer | Transformers | COCO, VG-MSDN | `creative-graphic-design/layout-transformer-coco`, `creative-graphic-design/layout-transformer-vg-msdn` |
| [`layoutdiffusion`](api/models/layoutdiffusion/index.md) | LayoutDiffusion | Diffusers | RICO25, PubLayNet | `creative-graphic-design/layoutdiffusion-rico25`, `creative-graphic-design/layoutdiffusion-publaynet` |
| [`layoutformerpp`](api/models/layoutformerpp/index.md) | LayoutFormer++ | Transformers | RICO25, PubLayNet | `creative-graphic-design/layoutformerpp-rico25-label`, `creative-graphic-design/layoutformerpp-rico25-label-size`, `creative-graphic-design/layoutformerpp-rico25-relation`, `creative-graphic-design/layoutformerpp-rico25-refinement`, `creative-graphic-design/layoutformerpp-rico25-completion`, `creative-graphic-design/layoutformerpp-rico25-unconditional`, `creative-graphic-design/layoutformerpp-publaynet-label`, `creative-graphic-design/layoutformerpp-publaynet-label-size`, `creative-graphic-design/layoutformerpp-publaynet-relation`, `creative-graphic-design/layoutformerpp-publaynet-refinement`, `creative-graphic-design/layoutformerpp-publaynet-completion`, `creative-graphic-design/layoutformerpp-publaynet-unconditional` |
| [`layoutganpp`](api/models/layoutganpp/index.md) | LayoutGAN++ | Transformers | RICO25, PubLayNet, Magazine | `creative-graphic-design/layoutganpp-rico`, `creative-graphic-design/layoutganpp-publaynet`, `creative-graphic-design/layoutganpp-magazine` |
| [`layoutprompter`](api/models/layoutprompter/index.md) | LayoutPrompter | Pydantic AI | PubLayNet, RICO25, PosterLayout | prompt configuration, no learned checkpoint |
| [`parse-then-place`](api/models/parse-then-place/index.md) | Parse-Then-Place | Transformers | RICO25, Web | `creative-graphic-design/parse-then-place-rico-pretrain`, `creative-graphic-design/parse-then-place-rico-finetune`, `creative-graphic-design/parse-then-place-web-pretrain`, `creative-graphic-design/parse-then-place-web-finetune` |
| [`ralf`](api/models/ralf/index.md) | RALF | Transformers | CGL, PKU | `creative-graphic-design/ralf-cgl-unconditional`, `creative-graphic-design/ralf-pku-unconditional` |

Canonical dataset ids are `creative-graphic-design/Rico` with config `ui-screenshots-and-hierarchies-with-semantic-annotations` for RICO25 and `creative-graphic-design/PubLayNet` for PubLayNet.
