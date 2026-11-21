---
language:
  - en
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - sawna
  - image-generation
  - text-to-image
datasets: []
---

<!-- --8<-- [start:card] -->

# Model Card for creative-graphic-design/sawna

[![DOI: 10.1145/3721250.3743023](https://img.shields.io/static/v1?label=DOI&message=10.1145%2F3721250.3743023&color=blue&style=flat-square&logo=doi&logoColor=white)](https://doi.org/10.1145/3721250.3743023)

![venue: SIGGRAPH Posters 2025](https://img.shields.io/static/v1?label=venue&message=SIGGRAPH+Posters+2025&color=purple&style=flat-square)

![license: Apache-2.0](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)

![base: diffusers](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)

![task: text-to-image](https://img.shields.io/static/v1?label=task&message=text-to-image&color=purple&style=flat-square)

![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)

[![hub: published-code-only](https://img.shields.io/static/v1?label=hub&message=published-code-only&color=brightgreen&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/creative-graphic-design/sawna)

This package applies SAWNA to reserve image regions for later design elements such as a headline or logo. It blends ordinary initial noise with TKG-DM-shifted noise inside selected regions before denoising begins.

The method is training-free and retains the usual [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) output interface. This package supports Stable Diffusion 1.5, Stable Diffusion 2.1, and Stable Diffusion XL. It includes no base-model weights.

## Model Details

### Model Description

SAWNA converts normalized `(x1, y1, x2, y2)` boxes to a binary latent-space mask and applies a separable Gaussian blur. It blends the shifted noise with `epsilon_masked = epsilon + mask_blurred * (epsilon_shifted - epsilon)`. Automatic blur uses two percent of the shorter latent-grid side. The [pinned Space implementation](https://huggingface.co/spaces/EQUES/Space-Aware_Text-to-Image_Generation/blob/e0a8f843a0b4e62c0650b03d5f335271ecb36d0b/tkg_dm.py#L111-L113) sets this default.

- **Developed by:** Ryugo Morita, Sho Kuno, Ryunosuke Tanaka, Rongzhi Li, Hoang Dai Dinh, and Issey Sukeda.
- **Shared by:** creative-graphic-design.
- **Model type:** image generation; task: text-to-image; conditioning: text.
- **Language(s):** English prompts.
- **Code license:** Apache-2.0, following the public Space license.
- **Base checkpoints:** Each checkpoint retains its upstream license and access terms.
- **Training:** No training or fine-tuning is performed by this method.

### Model Sources

- **Paper:** [SAWNA: Space-Aware Text to Image Generation](https://doi.org/10.1145/3721250.3743023), SIGGRAPH Posters 2025.
- **Presentation record:** [SIGGRAPH 2025 schedule](https://s2025schedule.siggraph.org/index.html?post_type=page&p=14&id=pos_281&sess=sess439.html).
- **Behavioral reference:** [EQUES Space source](https://huggingface.co/spaces/EQUES/Space-Aware_Text-to-Image_Generation/tree/e0a8f843a0b4e62c0650b03d5f335271ecb36d0b), pinned at `e0a8f843a0b4e62c0650b03d5f335271ecb36d0b`.
- **Shared channel shift:** [TKG-DM](https://github.com/creative-graphic-design/design-generators/tree/main/models/tkg-dm) is a runtime dependency.
- **Package source:** [design-generators](https://github.com/creative-graphic-design/design-generators/tree/main/models/sawna).

The Space names `runwayml/stable-diffusion-v1-5` for SD 1.5. This package uses the available checkpoint mirror below. The SD 2.1 checkpoint is a mirror of the deprecated Stability AI repository.

## Supported Checkpoints

| Base checkpoint      | Pipeline class                   | Hub ID                                        | Revision                                        | Default size |
| -------------------- | -------------------------------- | --------------------------------------------- | ----------------------------------------------- | -----------: |
| Stable Diffusion 1.5 | `SAWNAStableDiffusionPipeline`   | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`      |    512 × 512 |
| Stable Diffusion 2.1 | `SAWNAStableDiffusionPipeline`   | `sd2-community/stable-diffusion-2-1`          | `bb2154823665391b4fb29b0b9cf82a198964ee05`      |    768 × 768 |
| Stable Diffusion XL  | `SAWNAStableDiffusionXLPipeline` | `stabilityai/stable-diffusion-xl-base-1.0`    | `462165984030d82259a11f4367a4eed129e94a7b`      |  1024 × 1024 |
| SAWNA pipeline code  | Custom pipeline                  | `creative-graphic-design/sawna`               | Set `SAWNA_CODE_REVISION` to a published commit |    Code only |

Each base checkpoint uses four latent channels and an eightfold spatial scale. SDXL also uses its standard size and crop conditioning inputs. The Space's Dreamlike checkpoint is outside this package's scope.

## Uses

### Direct Use

Use a bounding box to reserve a region for later composition, such as a title, brand mark, or product label. The pipeline returns a standard `diffusers` image result.

### Downstream Use

Generated images can be reviewed and edited as draft assets for posters, banners, and other visual designs. Inspect the reserved region and the rest of each image before publication.

### Out-of-Scope Use

SAWNA does not guarantee an empty region, a specific subject placement, or rights to generated content. It does not add a safety guarantee or replace checkpoint-specific review.

## Bias, Risks, and Limitations

Generation behavior inherits the training data, limitations, license, and access terms of the selected base checkpoint. Results vary with checkpoint revision, scheduler, runtime versions, device, precision, and generation settings. Bounding boxes steer initial noise; they do not impose a hard pixel constraint throughout denoising.

### Recommendations

Record the base revision, scheduler configuration, box coordinates, target shift, blur width, seed, precision, and inference settings when comparing results. Review each image for artifacts and unwanted content before using it in a design.

## How to Get Started with the Model

The package has no learned checkpoints; inference downloads the selected upstream base model. To install from a repository checkout, run:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package sawna
```

Install both workspace packages from the same repository revision. This keeps the pipeline and its TKG-DM dependency aligned.

```bash
pip install \
  "sawna @ https://github.com/creative-graphic-design/design-generators/archive/main.zip#subdirectory=models/sawna" \
  "tkg-dm @ https://github.com/creative-graphic-design/design-generators/archive/main.zip#subdirectory=models/tkg-dm"
```

Load the pinned Stable Diffusion 1.5 checkpoint and pass a reserved region through `bounding_boxes`:

```python
import torch
from sawna import SAWNAStableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
pipe = SAWNAStableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
result = pipe(
    prompt="A majestic lion in a natural landscape",
    bounding_boxes=[(0.3, 0.3, 0.7, 0.7)],
    target_shift=0.07,
    blur_sigma=None,
    generator=torch.Generator(device="cpu").manual_seed(1234),
    num_inference_steps=25,
    guidance_scale=7.5,
)
image = result.images[0]
print(f"image size: {image.width}x{image.height}")
```

```text
image size: 512x512
```

After publication, load the custom pipeline with pinned code and base-model revisions:

```python
import os

import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    custom_pipeline="creative-graphic-design/sawna",
    custom_revision=os.environ["SAWNA_CODE_REVISION"],
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
result = pipe(
    prompt="A majestic lion in a natural landscape",
    bounding_boxes=[(0.3, 0.3, 0.7, 0.7)],
    generator=torch.Generator(device="cpu").manual_seed(1234),
    num_inference_steps=25,
    guidance_scale=7.5,
)
```

The Hub repository contains code only, with no base-model weights. The repository's fine-grained `HF_TOKEN` must include write access to this destination.

## Examples

Each example uses the same prompt, center box, target shift `0.07`, automatic blur, 25 steps, guidance scale `7.5`, seed `1234`, and fp16. The GPU, runtime, scheduler, and comparison record appear in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/sawna/REPRODUCING.md).

### Stable Diffusion 1.5

[![SAWNA SD 1.5 example: a lion in a natural landscape with space reserved for later design content](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-1-5-lion-seed1234.png)](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-1-5-lion-seed1234.png)

### Stable Diffusion 2.1

[![SAWNA SD 2.1 example: a lion in a natural landscape with space reserved for later design content](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-2-1-lion-seed1234.png)](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-2-1-lion-seed1234.png)

### Stable Diffusion XL

[![SAWNA SDXL example: a lion in a natural landscape with space reserved for later design content](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-xl-lion-seed1234.png)](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/sawna/sawna-stable-diffusion-xl-lion-seed1234.png)

## Training Details

### Training Data

Not applicable. SAWNA is training-free and adds no learned checkpoint.

### Training Procedure

No training or fine-tuning is performed by this package.

## Evaluation

### Testing Data, Factors & Metrics

The evaluation uses the pinned public Space implementation and the three checkpoint revisions above. Comparisons share the GPU, `diffusers` version, package lock, initial masked noise, scheduler configuration, generator state, and generation settings. Noise and generated image tensors use bitwise equality. Package repeatability uses the same public `generator` argument.

### Parity Results

| Check                         |               Cases | Criterion                                                                  | Result                   |
| ----------------------------- | ------------------: | -------------------------------------------------------------------------- | ------------------------ |
| Masked initial noise          |                   3 | Bitwise equality against the pinned Space implementation                   | 3/3 exact matches        |
| Generated image tensor        |                   3 | Bitwise equality from identical masked latents and sampler generator state | 3/3 exact matches        |
| Package repeatability         |                   3 | Bitwise equality for repeated calls with the same generator                | 3/3 exact matches        |
| Standard `diffusers` behavior | 2 pipeline families | Empty boxes preserve the normal initial-noise path                         | Covered by package tests |

An independent acceptance rerun on an L4 GPU compared the pinned Space reference with all three package pipelines. Masked initial noise, generated image pixels, and repeated package outputs matched bitwise in all three cases, with no skips; the run used 0.16 Colab units. See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/sawna/REPRODUCING.md) for the implementation and independent run records.

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/sawna/REPRODUCING.md) for pinned revisions, GPU commands, acceptance rules, and artifact formats.

## Environmental Impact

The method does not train models. Inference energy use depends on the checkpoint, device, precision, and generation settings.

## Citation

```bibtex
@inproceedings{morita2025sawna,
  author = {Morita, Ryugo and Kuno, Sho and Tanaka, Ryunosuke and Li, Rongzhi and Dinh, Hoang Dai and Sukeda, Issey},
  title = {SAWNA: Space-Aware Text to Image Generation},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Posters},
  pages = {1--2},
  year = {2025},
  doi = {10.1145/3721250.3743023}
}
```

<!-- --8<-- [end:card] -->
