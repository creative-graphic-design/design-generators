---
language:
  - en
license: "apache-2.0"
library_name: "diffusers"
pipeline_tag: "image-to-image"
tags:
  - "collage-diffusion"
  - "image-generation"
  - "wacv-2024"
datasets: []
base_model: "sd2-community/stable-diffusion-2-1-base"
---

<!-- --8<-- [start:card] -->

# Model Card for Collage Diffusion

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2303.00262&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.00262)
![venue](https://img.shields.io/static/v1?label=venue&message=WACV+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![framework](https://img.shields.io/static/v1?label=framework&message=diffusers&color=red&style=flat-square&logo=huggingface&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![task](https://img.shields.io/static/v1?label=task&message=image-to-image&color=purple&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-verified&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=planned-code-only&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package exposes Collage Diffusion as a `🧨diffusers` pipeline that harmonizes text-described RGBA layers into a single image.

## Model Details

### Model Description

Collage Diffusion combines a scene prompt with ordered RGBA image layers. Visible alpha masks guide cross-attention to each layer's text, while per-layer noise levels control how much of each source layer is preserved during diffusion. The pipeline can load caller-provided textual-inversion embeddings. It uses the upstream Stable Diffusion 2.1 base checkpoint and does not train, store, or publish model-specific weights.

- **Developed by:** Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Ré, and Kayvon Fatahalian.
- **Shared by:** creative-graphic-design.
- **Model type:** image generation; task: image-to-image; conditioning: layers, text.
- **Language(s) (NLP):** English scene and layer prompts.
- **License:** Apache-2.0 for this package; the base checkpoint and input assets retain their own terms.
- **Finetuned from:** No Collage Diffusion checkpoint; inference loads `sd2-community/stable-diffusion-2-1-base` at the pinned revision below under the CreativeML Open RAIL++-M terms declared by its model card.

### Model Sources

- **Repository:** [Collage Diffusion source repository](https://github.com/VSAnimator/collage-diffusion); it has no declared repository license. Its notebook is used as a behavioral reference; its code is not included here. Notebook-linked image layers and textual-inversion embeddings are archived, with source links and integrity records, in the [model repository's source-assets directory](https://huggingface.co/creative-graphic-design/collage-diffusion/tree/main/source-assets).
- **Paper:** [Collage Diffusion](https://arxiv.org/abs/2303.00262).
- **Related implementation:** [Collage Diffusion UI](https://github.com/linden-li/collage-diffusion-ui) was inspected as a secondary reference. Its repository also has no declared license; its code is not included here.
- **Released weights:** The package does not contain its own weights. It loads the [Stable Diffusion 2.1 base mirror](https://huggingface.co/sd2-community/stable-diffusion-2-1-base) at a pinned revision.

### Base Checkpoint Provenance and Integrity

The authors' notebook names `stabilityai/stable-diffusion-2-1-base`, but that Hub repository now returns HTTP 404. The comparison and package examples use `sd2-community/stable-diffusion-2-1-base` at revision `4e63672c03103b6c636b8fb4119ba982469b2955`. The model card declares the CreativeML Open RAIL++-M license; users must review and follow those terms.

The main weight files were checked across three available copies. Their SHA-256 values match byte-for-byte:

| Copy                                         | Revision                                   | UNet SHA-256                                                       | VAE SHA-256                                                        | Text encoder SHA-256                                               |
| -------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `sd2-community/stable-diffusion-2-1-base`    | `4e63672c03103b6c636b8fb4119ba982469b2955` | `6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214` | `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` | `cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c` |
| `Manojb/stable-diffusion-2-1-base`           | `0094d483a120f3f33dafbd187ea4aa60d10de75c` | same                                                               | same                                                               | same                                                               |
| `LanguageMachines/stable-diffusion-2-1-base` | `2579f99c90b5a3019fc115ee9566d5ec4f17fccf` | same                                                               | same                                                               | same                                                               |

SHA-256 values for every file used from the selected pinned snapshot:

| File                                         | SHA-256                                                            |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `model_index.json`                           | `518553e098fdaa4153d9a92139e344e44f53bb329c22f383293e3099c4f1bad2` |
| `feature_extractor/preprocessor_config.json` | `2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2` |
| `scheduler/scheduler_config.json`            | `11ac5627d7df0fa344b875c4b5722b1767a8a2aa1684c2cf8b4d614300127234` |
| `text_encoder/config.json`                   | `7e33bfc475f6cf76e69b28b101993ca834d1f22ff4d35caefde32d63d4f42fba` |
| `text_encoder/model.safetensors`             | `cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c` |
| `tokenizer/merges.txt`                       | `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a` |
| `tokenizer/special_tokens_map.json`          | `f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6` |
| `tokenizer/tokenizer_config.json`            | `d562b18a0c6d32a168fce3d7a342fce64579ebacffb98fb71b0affc494fc32d8` |
| `tokenizer/vocab.json`                       | `e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349` |
| `unet/config.json`                           | `ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608` |
| `unet/diffusion_pytorch_model.safetensors`   | `6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214` |
| `vae/config.json`                            | `424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903` |
| `vae/diffusion_pytorch_model.safetensors`    | `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` |

Both sides of the notebook comparison use this same pinned snapshot and the file hashes listed above. The selected model copy and its weights are not redistributed by this package.

## Supported Checkpoints

| Artifact                    | Hub ID                                      | Status                                    |
| --------------------------- | ------------------------------------------- | ----------------------------------------- |
| `🧨diffusers` pipeline code | `creative-graphic-design/collage-diffusion` | planned, code only; no learned checkpoint |

When publishing this pipeline to the Hub, its `model_index.json` and `scheduler/scheduler_config.json` must specify `EulerAncestralDiscreteScheduler`.

## Uses

### Direct Use

Use the pipeline to harmonize a layered collage while guiding object placement through masks and controlling source preservation through per-layer noise levels.

```python
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler
from collage_diffusion import CollageDiffusionPipeline, CollageLayer

base = "sd2-community/stable-diffusion-2-1-base"
revision = "4e63672c03103b6c636b8fb4119ba982469b2955"
# Cross-attention guidance uses the scheduler's sigma values.
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    base, subfolder="scheduler", revision=revision
)
pipe = CollageDiffusionPipeline.from_pretrained(
    base, revision=revision, scheduler=scheduler
)
layers = [
    CollageLayer(
        image=Image.open("toy.png").convert("RGBA"),
        prompt="a small wooden toy",
        position=(64, 96),
        noise_level=0.35,
        positive_attention_weight=0.5,
        negative_attention_weight=0.5,
    ),
]
result = pipe(
    prompt="a small wooden toy on a studio table",
    layers=layers,
    negative_prompt="collage",
    seed=0,
    num_inference_steps=20,
)
result.images[0].save("harmonized.png")
print("Saved harmonized.png")
```

```text
Saved harmonized.png
```

Textual-inversion embeddings are caller-supplied with a token-to-file mapping, and the matching token is set on its `CollageLayer`. ControlNet and textual-inversion training are outside this package's current scope.

### Examples

#### Toy Box collage

This example follows the Toy Box setup in the [Collage Diffusion notebook](https://github.com/VSAnimator/collage-diffusion/blob/main/CollageDiffusionDemo.ipynb), using its [Google Drive image layers](https://drive.google.com/drive/folders/1EfoLfWbn5fl7c092uk96zHtJOVHRlIqI?usp=sharing) and [textual-inversion assets](https://drive.google.com/drive/folders/13Gg8y-0FxdPcuhrPOUw6Ze9wa0Ruu-cI?usp=sharing). The source Drive files may disappear; archived copies, source URLs, retrieval date, byte sizes, and SHA-256 values are listed in the [model repository's source-assets directory](https://huggingface.co/creative-graphic-design/collage-diffusion/tree/main/source-assets). The archive contains author-provided files with no clear per-file license; the Apache-2.0 metadata on the separate documentation-images dataset does not apply to them, so their use and redistribution terms remain unclear.

The left panel shows the source collage and the right panel shows the generated image. Generation uses `sd2-community/stable-diffusion-2-1-base` at revision `4e63672c03103b6c636b8fb4119ba982469b2955`, seed `0`, 50 requested inference steps (40 active at strength `0.8`), and 512 × 512 resolution. The base checkpoint is licensed under the [CreativeML Open RAIL++-M License](https://huggingface.co/sd2-community/stable-diffusion-2-1-base).

![Toy Box input collage on the left and Collage Diffusion output on the right.](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/collage-diffusion/toy-box-diffusers-comparison-de56004.png)

```python
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils import load_image, make_image_grid
from huggingface_hub import hf_hub_download
from collage_diffusion import CollageDiffusionPipeline, CollageLayer, compose_collage_layers


asset_repo_id = "creative-graphic-design/collage-diffusion"
asset_path = "source-assets"
hub_endpoint = "https://huggingface.co"
image_base_url = f"{hub_endpoint}/{asset_repo_id}/resolve/main/{asset_path}"


def load_layer(filename: str):
    return load_image(
        f"{image_base_url}/image_layers/toyBox2/layers/{filename}",
        convert_method=lambda image: image.convert("RGBA"),
    )


textual_inversion_embeddings = {
    "<fda>": hf_hub_download(
        repo_id=asset_repo_id,
        repo_type="model",
        filename=f"{asset_path}/asset_library/toyBox2-teddy_bear/lora_weight_e999_s1000.ti.pt",
    ),
    "<xtc>": hf_hub_download(
        repo_id=asset_repo_id,
        repo_type="model",
        filename=f"{asset_path}/asset_library/toyBox2-wood_train/lora_weight_e999_s1000.ti.pt",
    ),
}

layers = [
    CollageLayer(
        image=load_layer("_0004_background.png"),
        prompt="background",
        noise_level=0.8,
        positive_attention_weight=0.0,
        negative_attention_weight=0.5,
    ),
    CollageLayer(
        image=load_layer("_0003_football.png"),
        prompt="american football",
        noise_level=0.2,
        positive_attention_weight=0.5,
        negative_attention_weight=0.5,
    ),
    CollageLayer(
        image=load_layer("_0002_teddy_bear.png"),
        prompt="teddy bear",
        noise_level=0.8,
        positive_attention_weight=0.2,
        negative_attention_weight=0.5,
        textual_inversion_token="<fda>",
    ),
    CollageLayer(
        image=load_layer("_0000_wood_train.png"),
        prompt="wood train",
        noise_level=0.5,
        positive_attention_weight=0.2,
        negative_attention_weight=0.5,
        textual_inversion_token="<xtc>",
    ),
]

input_collage, _ = compose_collage_layers(layers, width=512, height=512)
base = "sd2-community/stable-diffusion-2-1-base"
revision = "4e63672c03103b6c636b8fb4119ba982469b2955"
# Cross-attention guidance uses the scheduler's sigma values.
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    base, subfolder="scheduler", revision=revision
)
pipe = CollageDiffusionPipeline.from_pretrained(
    base, revision=revision, scheduler=scheduler
)
result = pipe(
    prompt="a teddy bear, a wood train, and an american football, in front of a tan background",
    layers=layers,
    textual_inversion_embeddings=textual_inversion_embeddings,
    negative_prompt="collage",
    strength=0.8,
    num_inference_steps=50,
    guidance_scale=7.5,
    noise_blur=20,
    seed=0,
).images[0]
make_image_grid([input_collage, result], rows=1, cols=2).save("toy-box-diffusers-comparison.png")
print("Saved toy-box-diffusers-comparison.png")
```

```text
Saved toy-box-diffusers-comparison.png
```

### Downstream Use

Generated images may be used for design exploration after human review of object placement, appearance, and the licenses attached to all input layers and model weights.

### Out-of-Scope Use

This pipeline does not guarantee semantic accuracy, rights clearance for caller-provided images or embeddings, or suitability for safety-critical applications.

## Bias, Risks, and Limitations

Results inherit the capabilities and biases of Stable Diffusion 2.1 and the caller's prompts, layers, masks, and textual-inversion embeddings. The original authors and rights holders of the notebook-linked example images and embeddings, and the license for each file, are unknown. The [source-assets inventory](https://huggingface.co/creative-graphic-design/collage-diffusion/blob/main/source-assets/README.md) records the retrieval URLs and integrity data. The paper's ControlNet extension and textual-inversion training are not implemented here.

### Recommendations

Review each output and confirm rights to every input layer, embedding, and generated asset before publication or commercial use.

## How to Get Started with the Model

Install the package with its workspace dependency from this repository:

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "collage-diffusion @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/collage-diffusion"
```

This package has no learned checkpoints. The pipeline loads the upstream Stable Diffusion base model and accepts caller-provided layers and optional textual-inversion embeddings.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package collage-diffusion
uv run --package collage-diffusion python
```

```python
from diffusers import EulerAncestralDiscreteScheduler
from collage_diffusion import CollageDiffusionPipeline

base = "sd2-community/stable-diffusion-2-1-base"
revision = "4e63672c03103b6c636b8fb4119ba982469b2955"
# Cross-attention guidance uses the scheduler's sigma values.
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    base, subfolder="scheduler", revision=revision
)
pipe = CollageDiffusionPipeline.from_pretrained(
    base, revision=revision, scheduler=scheduler
)
print(type(pipe).__name__)
```

## Training Details

### Training Data

No training dataset is packaged. Layer images and optional textual-inversion embeddings are caller inputs.

### Training Procedure

This package contains no training procedure and no Collage Diffusion checkpoint.

#### Training Hyperparameters

- **Training regime:** Not applicable; the pipeline uses an upstream inference checkpoint.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Ordinary tests use small synthetic RGBA layers. The comparison uses the Bento Box, Cake Table, and Toy Box scenes and their notebook-linked embeddings; public archive copies are linked above.

#### Factors

The reference comparison holds the input layers, preprocessing, textual-inversion embedding, scheduler, seed, and number of denoising steps constant between the notebook path and this package.

#### Metrics

Compare normalized final RGB arrays with RMSE, maximum absolute difference, and PSNR, and compare each post-scheduler, post-preservation latent with RMSE. The six comparisons use three 512 × 512 scenes and seeds 0 and 1, the same pinned checkpoint, each scene's layers and embeddings, Euler Ancestral, and 10 requested steps (7 or 8 active steps according to strength). Bento Box and Toy Box use strength `0.8` with `noise_blur=20`; Cake Table uses strength `0.75` with `noise_blur=10`.

### Parity Results

The six worker-run comparisons pass the tolerance thresholds fixed before the Collage runs from six ordinary Stable Diffusion Img2Img version-drift controls. The reference uses `🧨diffusers` `0.10.0`, Torch `2.0.0+cpu`, torchvision `0.15.1+cpu`, and `🤗transformers` `4.28.1`; the package uses `🧨diffusers` `0.39.0`, Torch `2.13.0+cpu`, torchvision `0.28.0+cpu`, and `🤗transformers` `5.13.0`. Both run on CPU with one thread. The coordinator's independent acceptance rerun is pending; vendor parity is not verified until that rerun is complete. The baseline and comparison plan are tracked in [issue #484](https://github.com/creative-graphic-design/design-generators/issues/484).

Each threshold uses `T = 2·Bmax − Bmedian` over the six ordinary Stable Diffusion controls: RGB RMSE has `Bmax=6.00713e-5`, `Bmedian=1.61880e-5`, and `T=1.03955e-4`; maximum per-step latent RMSE has `Bmax=1.01644e-4`, `Bmedian=4.42415e-5`, and `T=1.59046e-4`.

| Check                                       | Cases                    | Criterion and observed result                                                                            |
| ------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| Final normalized RGB                        | 6 pairs                  | RMSE ≤ `1.03955e-4`; worst RMSE `3.00032e-5`, PSNR `90.457`–`105.834 dB`, max abs `4.99487e-4`; 6/6 pass |
| Post-scheduler, post-preservation latents   | 6 runs, 7–8 active steps | Maximum per-step RMSE ≤ `1.59046e-4`; worst `6.88359e-5`; 6/6 pass                                       |
| First U-Net noise prediction                | 6 pairs                  | Reported for diagnosis; worst max abs `9.88245e-5`, worst RMSE `1.79971e-6`                              |
| Inputs, visible masks, noise maps, schedule | 6 pairs                  | Input RGB, masks, noise maps, timesteps, and sigmas matched exactly                                      |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/collage-diffusion/REPRODUCING.md) for CPU test and notebook-comparison steps.

## Environmental Impact

No model training is performed. CPU validation cost depends on the selected Colab memory profile, dependency installation, and comparison resolution.

## Technical Specifications

### Model Architecture and Objective

The pipeline uses Stable Diffusion image-to-image components with a `🧨diffusers` attention processor. It composites layers in back-to-front order, derives visible alpha masks, adjusts cross-attention logits for layer tokens, and blends the initial latent into each denoising step according to the inverted per-layer noise map.

### Compute Infrastructure

CPU is sufficient for small contract tests. Image generation uses the upstream Stable Diffusion checkpoint and is substantially slower on CPU than on a GPU.

#### Hardware

The requested implementation checks and image comparison run on a named Colab CPU session. No GPU is required for the planned comparison.

#### Software

Use Python 3.11 or newer, `🧨diffusers`, `🤗transformers`, PyTorch, Pillow, NumPy, and the repository's `laygen` workspace library. The pipeline uses the same workspace dependency configuration as other model packages.

## License

Package code follows the repository's Apache-2.0 license. The upstream notebook and UI source repositories do not declare a repository license; their implementations are referenced but not copied or distributed. Stable Diffusion 2.1 base is subject to its upstream CreativeML Open RAIL++-M terms. The original authors and rights holders of the notebook-linked example images and embeddings, and the license applicable to each file, are unknown. The archive records the URL from which each file was retrieved and its SHA-256; these records identify retrieval locations, not rights holders or license grants.

## Citation

```bibtex
@article{sarukkai2023collagediffusion,
  title = {Collage Diffusion},
  author = {Sarukkai, Vishnu and Li, Linden and Ma, Arden and Ré, Christopher and Fatahalian, Kayvon},
  journal = {arXiv preprint arXiv:2303.00262},
  year = {2023},
  eprint = {2303.00262},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2303.00262",
}
```

<!-- --8<-- [end:card] -->
