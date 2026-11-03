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
base_model: CompVis/stable-diffusion-v1-4
---

<!-- --8<-- [start:card] -->

# Model Card for StructureDiffusion

[StructureDiffusion](https://arxiv.org/abs/2212.05032) is a training-free text-to-image method from ICLR 2023. It parses a prompt into noun phrases, encodes the full prompt and each phrase, then combines cross-attention keys and values to improve object and attribute binding. This package provides a Diffusers pipeline for the [Stable Diffusion 1.4 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v1-4).

## Model Details

### Model Description

`StructureDiffusionPipeline` subclasses [`diffusers.StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), so it uses standard Diffusers loading, device placement, callbacks, scheduler, and image output behavior. `parser_type` selects constituency parsing or optional scene-graph parsing. `conjunction=True` uses phrase-specific keys with one combined value; the default uses the full prompt as the key and the full prompt plus phrase encodings as values.

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

### Out-of-Scope Use

Do not treat generated images as verified depictions of real people or events, or use them as evidence that prompt attributes were rendered correctly. The pipeline does not provide an image safety guarantee.

## Bias, Risks, and Limitations

Generation behavior inherits the training data, biases, and restrictions of the selected Stable Diffusion checkpoint. The default constituency parser is English-only and requires the [Stanza](https://stanfordnlp.github.io/stanza/) English tokenizer, POS, and constituency models to be installed separately. Scene-graph parsing is an optional extra with its own [spaCy](https://spacy.io/) English model prerequisite. Outputs vary with the checkpoint, parser, scheduler, device, precision, and library versions.

### Recommendations

Inspect generated images before downstream use, and record the checkpoint revision, parser, scheduler, dimensions, seed, and inference steps when comparing results.

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

## Training Details

### Training Data

This method adds no training data or learned weights. The selected Stable Diffusion checkpoint retains its original data provenance and terms.

### Training Procedure

No training or checkpoint conversion is performed by this package. It loads the released Diffusers-format base checkpoint and changes the denoising pipeline's text-conditioning path.

## Evaluation

### Testing Data, Factors & Metrics

The comparison fixture uses the prompt `A red car and a white sheep`, seed 42, 512×512 output, 10 DDIM steps, guidance scale 7.5, CPU float32 execution, and one precomputed starting latent shared by the official attention reference and this package. The 64×64 run is only a CPU rehearsal.

### Parity Results

| Check                                                                 | Cases | Criterion                                                                     | Result                 |
| --------------------------------------------------------------------- | ----: | ----------------------------------------------------------------------------- | ---------------------- |
| CPU image output against the pinned official attention implementation |     0 | Phrase and CLIP token-span match; mean absolute image difference below `1e-3` | Pending Colab CPU run. |

## Reproducibility

Run the CPU comparison test in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/structurediffusion/REPRODUCING.md) to compare generated images with the pinned official attention implementation using the same parser output, scheduler, seed, and initial latent.

## Environmental Impact

No model training is performed. CPU inference is computationally expensive; runtime depends on the host CPU, image dimensions, and denoising steps.

## Technical Specifications

### Model Architecture and Objective

The pipeline retains Stable Diffusion's VAE, CLIP text encoder, U-Net, and scheduler. It extracts phrase spans, encodes each phrase independently, aligns phrase tokens to the full prompt, and applies the official method's attention-map and value-composition rules in U-Net cross-attention layers.

### Compute Infrastructure

The package supports Diffusers CPU and accelerator execution. The currently planned agreement run uses Colab CPU only; no GPU comparison is included.

#### Software

Runtime dependencies are declared in `models/structurediffusion/pyproject.toml`. The default parser requires Stanza models; scene-graph parsing also requires the optional `scene-graph` extra and its spaCy model.

## Citation

```bibtex
@inproceedings{feng2023trainingfree,
  title={Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis},
  author={Feng, Weixi and He, Xuehai and Fu, Tsu-Jui and Jampani, Varun and Akula, Arjun and Narayana, Pradyumna and Basu, Sugato and Wang, Xin Eric and Wang, William Yang},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://arxiv.org/abs/2212.05032}
}
```

## Model Card Contact

Open an issue or pull request in the [design-generators repository](https://github.com/creative-graphic-design/design-generators).

<!-- --8<-- [end:card] -->
