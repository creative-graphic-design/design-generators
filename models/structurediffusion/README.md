---
language:
  - en
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - structured-diffusion-guidance
  - text-to-image
  - layout-generation
datasets: []
base_model: CompVis/stable-diffusion-v1-4
model-index:
  - name: "StructureDiffusion"
    results:
      - task:
          type: "text-to-image"
          name: "Text-to-image generation"
        dataset:
          type: "synthetic"
          name: "CPU parity fixture"
          split: "one prompt at 64x64 and 512x512"
        metrics:
          - type: "vendor-parity"
            value: 0.0
            name: "Bit-exact CPU image comparison against reference images"
---

<!-- --8<-- [start:card] -->

# Model Card for StructureDiffusion

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2212.05032&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2212.05032)
![venue](https://img.shields.io/static/v1?label=venue&message=ICLR+2023&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=diffusers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![task](https://img.shields.io/static/v1?label=task&message=text-to-image&color=purple&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package implements [StructureDiffusion](https://arxiv.org/abs/2212.05032), a training-free text-to-image method from ICLR 2023. It parses a prompt into noun phrases, encodes the full prompt and each phrase, then combines cross-attention keys and values to improve object and attribute binding. The [`🧨diffusers`](https://huggingface.co/docs/diffusers) pipeline uses the [Stable Diffusion 1.4 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v1-4).

## Model Details

### Model Description

`StructureDiffusionPipeline` subclasses [`diffusers.StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), so it uses the framework's standard loading, device placement, callbacks, scheduler, and image output behavior. `parser_type` selects constituency parsing or optional scene-graph parsing. `conjunction=True` uses phrase-specific keys with one combined value; the default uses the full prompt as the key and the full prompt plus phrase encodings as values.

- **Developed by:** Weixi Feng et al.
- **Shared by:** creative-graphic-design.
- **Model type:** Diffusion text-to-image generation with structured cross-attention.
- **Language(s):** English prompts.
- **Code license:** Apache-2.0.
- **Reference implementation license:** MIT.
- **Base checkpoint:** CompVis Stable Diffusion v1.4; its weights have separate CreativeML OpenRAIL-M terms.

### Model Sources

- **Repository:** [official Structured Diffusion Guidance implementation](https://github.com/weixi-feng/Structured-Diffusion-Guidance/tree/773b2ba068c6cc60443d5d747a725413c8c9d46f)
- **Paper:** [arXiv:2212.05032](https://arxiv.org/abs/2212.05032)
- **Released base weights:** [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

## Supported Checkpoints

| Checkpoint                          | Hub ID                                                             | Status                                                                        |
| ----------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Stable Diffusion v1.4               | `CompVis/stable-diffusion-v1-4`                                    | Public base checkpoint; load directly with `from_pretrained`.                 |
| StructureDiffusion package artifact | `creative-graphic-design/structurediffusion-stable-diffusion-v1-4` | Planned; publication is separate and no weights are uploaded by this package. |

## Uses

### Direct Use

Use the pipeline for research inference on English prompts that describe multiple objects, attributes, or relations. The method changes cross-attention conditioning and does not train or fine-tune Stable Diffusion weights.

### Downstream Use

Use generated images as draft visual material for creative-design workflows after inspecting them for prompt adherence and content suitability.

### Out-of-Scope Use

Do not treat generated images as verified depictions of real people or events, or use them as evidence that prompt attributes were rendered correctly. The pipeline does not provide an image safety guarantee.

## How to Get Started with the Model

Install the package from this repository and download the parser models before the first inference:

```bash
pip install "structurediffusion @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/structurediffusion"
python -c "import stanza; stanza.download('en', processors='tokenize,pos,constituency')"
```

```python
import torch
from structurediffusion import StructureDiffusionPipeline

pipe = StructureDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
)
pipe.to("cpu")
result = pipe(
    "A red car and a white sheep",
    parser_type="constituency",
    num_inference_steps=10,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
image = result.images[0]
```

The optional scene-graph parser can be installed with the `scene-graph` extra. See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/structurediffusion/REPRODUCING.md) for local testing and CPU comparison instructions.

## Examples

The following example generates both images with the same Stable Diffusion v1.4 checkpoint, prompt, seed, scheduler, and sampling settings. It uses the regular `🧨diffusers` pipeline for the baseline and `StructureDiffusionPipeline` for structured guidance. Set `HF_TOKEN_FILE` to a private file containing a Hugging Face token after accepting the checkpoint's access terms. Install the package and download the English Stanza parser models as shown above before running this code.

| Stable Diffusion v1.4                                                                                                                                                                                                                                     | StructureDiffusion                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![Stable Diffusion v1.4 output: a red teddy bear in a Christmas hat sitting next to a glass](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/structurediffusion/stable-diffusion-v1-4-red-teddy-bear-glass.png) | ![StructureDiffusion output: a red teddy bear in a Christmas hat sitting next to a glass](https://huggingface.co/datasets/creative-graphic-design/documentation-images/resolve/main/structurediffusion/structurediffusion-red-teddy-bear-glass.png) |

Both images use the prompt `A red teddy bear in a Christmas hat sitting next to a glass`, base model revision `133a221b8aa7292a167afc5127cb63fb5005638b`, seed `42`, 30 inference steps, 512×512 resolution, guidance scale `7.5`, and the checkpoint's PNDM scheduler (`skip_prk_steps=True`). The pinned checkpoint already provides this scheduler configuration, so the example uses it directly. The base checkpoint is licensed under CreativeML OpenRAIL-M.

```python
import gc
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from structurediffusion import StructureDiffusionPipeline

MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
PROMPT = "A red teddy bear in a Christmas hat sitting next to a glass"
SEED = 42
OUTPUT_DIR = Path("structurediffusion-examples")
OUTPUT_DIR.mkdir(exist_ok=True)

torch.set_num_threads(2)
token = Path(os.environ["HF_TOKEN_FILE"]).read_text().strip()


def make_generator():
    return torch.Generator(device="cpu").manual_seed(SEED)


baseline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float32,
    token=token,
)
baseline.to("cpu")
baseline_image = baseline(
    PROMPT,
    height=512,
    width=512,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=make_generator(),
).images[0]
baseline_image.save(OUTPUT_DIR / "stable-diffusion-v1-4-red-teddy-bear-glass.png")
del baseline, baseline_image
gc.collect()

structured = StructureDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float32,
    token=token,
)
structured.to("cpu")
structured_image = structured(
    PROMPT,
    parser_type="constituency",
    height=512,
    width=512,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=make_generator(),
).images[0]
structured_image.save(OUTPUT_DIR / "structurediffusion-red-teddy-bear-glass.png")
del structured, structured_image
gc.collect()
```

## Bias, Risks, and Limitations

Generation behavior inherits the training data, biases, and restrictions of the selected Stable Diffusion checkpoint. The default constituency parser is English-only and requires the [Stanza](https://stanfordnlp.github.io/stanza/) English tokenizer, POS, and constituency models to be installed separately. Scene-graph parsing is an optional extra with its own [spaCy](https://spacy.io/) English model prerequisite. Outputs vary with the checkpoint, parser, scheduler, device, precision, and library versions.

### Recommendations

Inspect generated images before downstream use, and record the checkpoint revision, parser, scheduler, dimensions, seed, and inference steps when comparing results.

## Training Details

### Training Data

This method adds no training data or learned weights. The selected Stable Diffusion checkpoint retains its original data provenance and terms.

### Training Procedure

No training or checkpoint conversion is performed by this package. It loads the released pipeline-format base checkpoint and changes the denoising pipeline's text-conditioning path.

## Evaluation

### Testing Data, Factors & Metrics

The independent CPU comparison covers 64×64 and 512×512 outputs for `A red car and a white sheep`, seed 42, 10 `🧨diffusers` PNDM steps through its PLMS path, guidance scale 7.5, and CPU float32 execution. Each reference/package pair receives the same precomputed starting latent. The vendor demo calls `preprocess_prompts` before parsing and token alignment, which lowercases the prompt; its `get_token_alignment_map` helper is case-sensitive only when called directly with mismatched prompt and tokenizer casing. The package also accepts capitalized input.

The test directly compares vendor-derived noun phrases and token spans, and compares independently assembled key/value conditioning with the package conditioning. For the image comparison, a separate base `🧨diffusers` pipeline uses the pinned vendor attention processors to generate the reference image; the test compares that final image with the `StructureDiffusionPipeline` output. It does not capture or compare per-layer attention outputs. Both model loading and sampling use `🧨diffusers`, so this checks structured conditioning and its end-to-end image result without reproducing the vendor checkpoint loader or sampler implementation.

### Parity Results

| Check                                                | Cases | Criterion                                                                          | Result                                                                                                                                                                                         |
| ---------------------------------------------------- | ----: | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 64×64 CPU image output against the reference image   |     1 | Exact phrase and token-span match; K/V max difference ≤ `1e-6`; image MAE < `1e-3` | `a red car` / `a white sheep`; spans `[1, 4)` / `[5, 8)`; 1 key / 3 values; K/V max differences `0.0` / `0.0`; pixel max/mean/RMS differences `0.0` / `0.0` / `0.0`; bit-exact, SHA-256 match. |
| 512×512 CPU image output against the reference image |     1 | Exact phrase and token-span match; K/V max difference ≤ `1e-6`; image MAE < `1e-3` | `a red car` / `a white sheep`; spans `[1, 4)` / `[5, 8)`; 1 key / 3 values; K/V max differences `0.0` / `0.0`; pixel max/mean/RMS differences `0.0` / `0.0` / `0.0`; bit-exact, SHA-256 match. |

The independent comparison ran on commit `786fcf6cfbcfb68b9c9565d8ee275c32cd5a28ac`; the reviewer recorded the full inputs, initial latents, measurements, and runtimes in the [formal parity result](https://github.com/creative-graphic-design/design-generators/issues/485#issuecomment-6608854871). Both resolutions were bit-exact: the package and reference image SHA-256 values matched.

## Reproducibility

Run the CPU comparison test in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/structurediffusion/REPRODUCING.md) to compare noun phrases, token spans, and K/V conditioning directly, then compare final generated images using the same scheduler, seed, and initial latent. A separate base `🧨diffusers` pipeline runs the pinned reference attention processors to generate the reference image; the test does not compare per-layer attention outputs. `🧨diffusers` supplies model loading and sampling in both paths, so the result does not claim parity with the reference checkpoint loader or sampler. The independently reviewed comparison on commit `786fcf6` matched phrase and span outputs exactly, had zero measured K/V differences, and produced bit-exact 64×64 and 512×512 images with identical SHA-256 values.

## Environmental Impact

No model training is performed. The formal standard-CPU comparison took 126.777 seconds at 64×64 and 948.902 seconds at 512×512; no GPU comparison is included.

## Technical Specifications

### Model Architecture and Objective

The pipeline retains Stable Diffusion's VAE, CLIP text encoder, U-Net, and scheduler. It extracts phrase spans, encodes each phrase independently, aligns phrase tokens to the full prompt, and applies the official method's attention-map and value-composition rules in U-Net cross-attention layers.

### Compute Infrastructure

The package supports `🧨diffusers` CPU and accelerator execution. The recorded comparison timings are test-process runtimes on a standard Colab CPU; setup time and GPU performance are not included.

#### Software

Runtime dependencies are declared in `models/structurediffusion/pyproject.toml`. The default parser requires Stanza models; scene-graph parsing also requires the optional `scene-graph` extra and its spaCy model.

## Citation

```bibtex
@inproceedings{feng2023trainingfree,
  title={Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis},
  author={Feng, Weixi and He, Xuehai and Fu, Tsu-Jui and Jampani, Varun and Akula, Arjun and Narayana, Pradyumna and Basu, Sugato and Wang, Xin Eric and Wang, William Yang},
  booktitle={International Conference on Learning Representations},
  year={2023},
  eprint={2212.05032},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url = "https://arxiv.org/abs/2212.05032"
}
```

## Model Card Contact

Open an issue or pull request in the [design-generators repository](https://github.com/creative-graphic-design/design-generators).

<!-- --8<-- [end:card] -->
