---
language:
  - en
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - tkg-dm
  - image-generation
  - text-to-image
datasets: []
---

<!-- --8<-- [start:card] -->

# Model Card for creative-graphic-design/tkg-dm

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2411.15580&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2411.15580)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2025&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![task](https://img.shields.io/static/v1?label=task&message=text-to-image&color=purple&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=published-code-only&color=brightgreen&style=flat-square&logo=huggingface&logoColor=white)

This package implements TKG-DM to generate foreground assets on a green chroma-key background, so you can key out the background and composite the subject into another design.

Choose either public checkpoint: `stable-diffusion-v1-5/stable-diffusion-v1-5` or `stabilityai/stable-diffusion-xl-base-1.0`. Provide a text prompt; the pipeline returns an RGB image with a green chroma-key background. This package fixes positive shifts to latent channels 2 and 3, producing green. The paper describes changing channels and shift directions to target other colors, which this package does not expose. `target_shift` controls how much the positive-value ratio of two initial-noise channels increases; the pipeline uses `0.07` for SD 1.5 and `0.11` for SDXL.

This package contains no TKG-DM weights; it loads the selected public Stable Diffusion checkpoint. An independent rerun of the pinned SDXL comparison with the [author's implementation](https://github.com/ryugo417/TKG-DM) confirmed zero measured image error (PSNR ∞ dB, MAE 0).

## Model Details

### Model Description

The pipeline shifts the positive-value ratio of two latent channels and blends the result with the original noise through a centered Gaussian mask before denoising. It keeps the standard [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) loading and generation interfaces and supports DDIM schedulers.

- **Developed by:** Ryugo Morita, Stanislav Frolov, Brian Bernhard Moser, Takahiro Shirakawa, Ko Watanabe, Andreas Dengel, and Jinjia Zhou.
- **Shared by:** creative-graphic-design.
- **Model type:** image generation; task: text-to-image; conditioning: text.
- **Language(s):** English prompts.
- **Code license:** Apache-2.0.
- **Reference implementation license:** The author repository does not declare a license. Its source code is used only as a temporary execution reference and is not included in this package.
- **Base checkpoints:** Stable Diffusion 1.5 and SDXL retain their respective upstream licenses and terms.

The paper reports `target_shift=0.07` for both model families, where `0.07` raises the positive-value ratio by seven percentage points. The author SDXL implementation uses `0.11`, which is the SDXL pipeline default and end-to-end parity reference. SD 1.5 uses `0.07` because the author repository does not provide an SD 1.5 pipeline.

### Model Sources

- **Repository:** [author implementation](https://github.com/ryugo417/TKG-DM).
- **Paper:** [arXiv:2411.15580](https://arxiv.org/abs/2411.15580).
- **SD 1.5 base weights:** [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- **SDXL base weights:** [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- **Released TKG-DM weights:** None; when published, the planned Hub repository will contain pipeline code only.

## Supported Checkpoints

| Checkpoint                | Hub ID                                        | Revision                                   | Status                                                      |
| ------------------------- | --------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| Stable Diffusion 1.5 base | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | Upstream base weights; terms apply.                         |
| SDXL base                 | `stabilityai/stable-diffusion-xl-base-1.0`    | `462165984030d82259a11f4367a4eed129e94a7b` | Upstream base weights; terms apply.                         |
| TKG-DM pipelines          | `creative-graphic-design/tkg-dm`              | —                                          | Not published; planned as code only, with no model weights. |

## Uses

### Direct Use

Use the pipelines for inference with compatible Stable Diffusion 1.5 or SDXL checkpoints when a chroma-key background is useful for image compositing.

### Downstream Use

Generated images can serve as draft foreground assets in design and compositing workflows after inspection and any required editing.

### Out-of-Scope Use

The package does not guarantee that every output has a uniformly removable background, and it does not provide a safety guarantee or establish rights to generated content.

## Bias, Risks, and Limitations

Generation behavior inherits the training data, biases, license, and access terms of the selected base checkpoint. The author repository has no declared license; this package does not redistribute its source. Outputs vary with the checkpoint, scheduler, device, precision, and library versions.

### Recommendations

Inspect each generated image before compositing. Record the base checkpoint revision, scheduler, target shift, dimensions, seed, precision, and inference steps when comparing results.

## How to Get Started with the Model

Install the package from the repository source archive:

```bash
pip install "tkg-dm @ https://github.com/creative-graphic-design/design-generators/archive/main.zip#subdirectory=models/tkg-dm"
```

There are no learned checkpoints for TKG-DM in this package; inference uses pretrained Stable Diffusion base weights. To work from the source checkout, clone this repository and install the workspace package:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package tkg-dm
```

Load an upstream checkpoint with the matching TKG-DM pipeline and DDIM scheduler. This example uses SDXL at the pinned revision:

```python
import torch
from diffusers import DDIMScheduler
from tkg_dm import TKGStableDiffusionXLPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
scheduler = DDIMScheduler.from_pretrained(
    MODEL_ID,
    subfolder="scheduler",
    revision=MODEL_REVISION,
)
pipe = TKGStableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
result = pipe(
    prompt="yellow lemon and slice, realistic, photo-realistic, 4K, high resolution, high quality",
    negative_prompt="background, character, cartoon, anime, text, fail, low resolution",
    generator=torch.Generator(device="cpu").manual_seed(1234),
    num_inference_steps=50,
    guidance_scale=7.5,
)
image = result.images[0]
print(f"image size: {image.width}x{image.height}")
```

```text
image size: 1024x1024
```

After publication, the code-only Hub pipeline can be loaded without installing this package. Use the publisher's full commit SHA as `custom_revision` and pin the base model with `revision`:

```python
import os

import torch
from diffusers import DDIMScheduler, DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    scheduler=DDIMScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
        revision=MODEL_REVISION,
    ),
    custom_pipeline="creative-graphic-design/tkg-dm",
    custom_revision=os.environ["TKG_DM_CODE_REVISION"],
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
result = pipe(
    "yellow lemon and slice, realistic, photo-realistic, 4K, high resolution, high quality",
    negative_prompt="background, character, cartoon, anime, text, fail, low resolution",
    generator=torch.Generator(device="cpu").manual_seed(1234),
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

When published, the Hub repository will provide `pipeline.py` and its package-owned helper module. It will not include base-model weights. Set `TKG_DM_CODE_REVISION` to the full code commit SHA returned by the publisher. The base-model `revision` remains the checkpoint SHA shown above.

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/tkg-dm/REPRODUCING.md) for parity and local loading instructions.

## Examples

Both images use the displayed prompt with the negative prompt in the usage example, 50 DDIM steps, guidance scale `7.5`, seed `1234`, and fp16 inference. The SD 1.5 output is 512×512 with `target_shift=0.07`; the SDXL output is 1024×1024 with `target_shift=0.11`. The base model revisions and tested Colab GPU/runtime are recorded in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/tkg-dm/REPRODUCING.md).

### Stable Diffusion 1.5

[![TKG-DM Stable Diffusion 1.5 generation of a yellow lemon and slice on a chroma-key background](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/tkg-dm/tkg-stable-diffusion-v1-5-yellow-lemon-seed1234.png)](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/tkg-dm/tkg-stable-diffusion-v1-5-yellow-lemon-seed1234.png)

### SDXL

[![TKG-DM SDXL generation of a yellow lemon and slice on a chroma-key background](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/tkg-dm/tkg-stable-diffusion-xl-yellow-lemon-seed1234.png)](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/tkg-dm/tkg-stable-diffusion-xl-yellow-lemon-seed1234.png)

## Training Details

### Training Data

Not applicable. TKG-DM is training-free and introduces no learned checkpoint.

### Training Procedure

No model training or fine-tuning is performed by this package.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The comparison uses the pinned SDXL base checkpoint and the fixed lemon prompt from the usage example. SD 1.5 uses the shared noise transformation and repeatable package inference because the author repository does not provide an SD 1.5 end-to-end pipeline.

#### Factors

Both implementations use seed `1234`, fp16, DDIM, 50 steps, guidance scale `7.5`, and their pinned base revisions. The SDXL image comparison uses `target_shift=0.11`; the SD 1.5 package uses its `0.07` default.

#### Metrics

Noise tensors are compared with `rtol=1e-5` and `atol=1e-6`. SDXL images are converted to normalized RGB values and compared with PSNR and mean absolute error (MAE).

### Parity Results

An independent acceptance rerun confirmed these parity results. The acceptance criteria were fixed before running either implementation. Image comparisons use the complete normalized RGB images without cropping.

| Check                                 | Cases | Criterion                                                    | Result                                                                                                |
| ------------------------------------- | ----: | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Noise transformation, SD 1.5 and SDXL |     2 | `rtol=1e-5`, `atol=1e-6`                                     | Pass; maximum absolute error was 0 for both model families.                                           |
| SDXL final image                      |     1 | PSNR ≥ 48 dB and MAE ≤ 1/255                                 | Pass; PSNR was ∞ dB and MAE was 0 at 1024×1024.                                                       |
| SD 1.5 repeated package inference     |     2 | Identical output image hashes for the same seed and settings | Pass; both RGB images had SHA-256 `4c76d1b6af804a6660f0a5ca2b3d335118b0af497cb361e997c418c5f6becdc3`. |

## Reproducibility

Reproduce the comparisons with the original implementation using [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/tkg-dm/REPRODUCING.md). It documents the temporary reference checkout, pinned models, fixed settings, acceptance criteria, and required-asset commands.

## Environmental Impact

This package does not train models. Inference compute and energy use depend on the selected base checkpoint, GPU, precision, and generation settings.

## Citation

```bibtex
@InProceedings{Morita_2025_CVPR,
    author    = {Morita, Ryugo and Frolov, Stanislav and Moser, Brian Bernhard and Shirakawa, Takahiro and Watanabe, Ko and Dengel, Andreas and Zhou, Jinjia},
    title     = {TKG-DM: Training-free Chroma Key Content Generation Diffusion Model},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {13031-13040}
}
```

## Model Card Authors

creative-graphic-design maintainers.

## Model Card Contact

Open an issue or pull request in the [creative-graphic-design/design-generators repository](https://github.com/creative-graphic-design/design-generators).

<!-- --8<-- [end:card] -->
