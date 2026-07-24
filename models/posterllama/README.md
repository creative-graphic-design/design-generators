---
language:
  - en
license: "other"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layout-generation"
  - "poster-generation"
  - "posterllama"
datasets:
  - "creative-graphic-design/CGL-Dataset"
model-index:
  - name: "PosterLlama"
    results:
      - task:
          type: "other"
          name: "Content-aware poster layout generation"
        dataset:
          type: "creative-graphic-design/CGL-Dataset"
          name: "CGL"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for PosterLlama

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2404.00995&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2404.00995)
![venue](https://img.shields.io/static/v1?label=venue&message=ECCV+2024&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=blocked&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package exposes the ECCV 2024 PosterLlama image-conditioned poster layout
recipe, which prompts a CodeLLaMA/LLaMA-family language model to emit HTML/SVG
`<rect>` elements for poster components.

## Model Details

### Model Description

PosterLlama combines a MiniGPT-style vision-to-language adapter, a DINOv2/EVA
visual encoder path, and a CodeLLaMA/LLaMA-family language model. The public
package surface is a `PosterLlamaProcessor` plus `PosterLlamaPipeline`: the
processor builds HTML/SVG layout prompts and parses generated `<rect>` elements,
while the pipeline owns local runtime generation after conversion.

- **Developed by:** PosterLlama authors.
- **Converted and maintained by:** creative-graphic-design.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware poster layout generation recipe.
- **Language(s) (NLP):** English prompt metadata.
- **License:** source and converted-weight redistribution are unverified.

### Model Sources

- **Paper:** [PosterLlama: Bridging Design Ability of Language Model to Content-Aware Layout Generation](https://arxiv.org/abs/2404.00995)
- **Original implementation:** [PosterLlama repository](https://github.com/Poetryhan/PosterLlama)
- **Raw checkpoint:** [poong/PosterLlama](https://huggingface.co/poong/PosterLlama)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| Raw PosterLlama state dict | `poong/PosterLlama` | public source checkpoint |
| CGL recipe artifact | `creative-graphic-design/posterllama-cgl` | not published; redistribution blocked |

The raw checkpoint is a `pytorch_model.bin` state dict. Local conversion also
requires explicit CodeLLaMA or Llama-2-family backbone access and the selected
vision encoder assets.

## Uses

### Direct Use

Use the processor to construct PosterLlama HTML prompts and parse generated
markup into the common layout schema. Use the pipeline after local conversion
when the raw checkpoint, backbone, and vision components are available.

### Downstream Use

Poster generation systems can call the processor parser to normalize
PosterLlama-style pixel `ltwh` rectangles into normalized center `xywh` boxes,
dataset-local labels, mask, and `id2label`.

### Out-of-Scope Use

Do not publish converted weights or model cards for redistributed artifacts
until source-checkpoint, backbone, and adapter license terms are reviewed.
Text-only generation is not a supported condition for this package.

## Bias, Risks, and Limitations

Outputs inherit the CGL label vocabulary, the source prompt templates, and the
base language model's generation behavior. The first package scope includes
prompt/parser parity and local smoke paths; full 7B inference parity requires
local assets and one selected GPU.

### Recommendations

Record the exact raw checkpoint path, backbone path, generation arguments, seed,
and parser version for any reported result. Prefer deterministic generation
settings before comparing sampled outputs.

## How to Get Started with the Model

Install the package directly from this repository. The command includes `laygen`
and `posgen` because they are shared workspace libraries.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "posterllama @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/posterllama"
```

```python
from posterllama import PosterLlamaConfig, PosterLlamaProcessor

config = PosterLlamaConfig(canvas_size=(360, 504))
processor = PosterLlamaProcessor.from_config(config)
prompt = processor.build_prompt(
    condition_type="cond_cate_size_to_pos",
    labels=["text"],
    bbox=[[[0.5, 0.5, 0.3, 0.1]]],
)
layout = processor.parse_output(
    '<svg width="360" height="504"><rect data-category="text" x="10" y="20" width="100" height="40"/></svg>'
)
print(prompt)
print(layout.bbox)
```

Clone this repository and run the local smoke script after conversion:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package posterllama
uv run --package posterllama python models/posterllama/scripts/smoke_from_pretrained.py \
  .cache/posterllama/converted
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| CGL | [`creative-graphic-design/CGL-Dataset`](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset) | poster element labels used by the recipe interface |

The package records CGL labels through `posgen.common`. PKU-PosterLayout remains
metadata-only for this initial package until checkpoint evidence confirms a
PKU-trained PosterLlama artifact.

### Training Procedure

No training is performed by this package. The local conversion path records the
raw checkpoint and backbone requirements needed for inference.

#### Speeds, Sizes, Times

Prompt construction and parsing are CPU-only. Full inference uses 7B-class
backbone weights and should be run with an explicitly selected GPU.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Unit tests use synthetic prompts and representative generated HTML/SVG snippets.
Gated parity uses local original-code assets and does not commit generated text,
tensors, images, checkpoints, or references.

#### Factors

Checks are separated by condition alias, parser strictness, canvas source,
label mapping, and generation settings.

#### Metrics

Metrics include prompt field identity, parser behavior, normalized box values,
schema output fields, and local `save_pretrained` to `from_pretrained` smoke
loading.

### Parity Results

| Check | Cases | Match criterion | Result |
| --- | ---: | --- | --- |
| Prompt construction | 5 condition aliases | Deterministic condition mapping and `<FILL_i>` order | passed in unit tests |
| HTML/SVG parser | 4 parser cases | Safe numeric parse, unknown-label skip, pixel `ltwh` to normalized center `xywh` | passed in unit tests |
| Original GPU generation | 0 committed cases | Gated original `generate.py` run with fixed local assets | blocked until local 7B assets and GPU window are available |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/posterllama/REPRODUCING.md) for the commands that inspect source assets, generate reference metadata, run parity checks, convert local artifacts, and smoke-test local loading.

## Environmental Impact

No model training is performed by this package. Full inference and original-code
parity require local 7B-class backbone assets.

## Technical Specifications

### Model Architecture and Objective

The recipe constructs HTML/SVG prompts for unconditional, label, label-size,
completion, refinement, and content-image conditions. Generated `<rect>` tags
are parsed as pixel `ltwh` boxes and returned as normalized center `xywh`
layouts.

### Compute Infrastructure

Prompt and parser checks run on CPU. Original-code generation parity requires a
single selected GPU and local raw checkpoint/backbone paths.

### Hardware

CPU is sufficient for processor/parser tests. Full inference requires hardware
appropriate for CodeLLaMA/LLaMA-family 7B checkpoints.

### Software

The normal package depends on `torch`, `transformers`, `huggingface-hub`,
`laygen`, and `posgen`. The `vendor` extra isolates MiniGPT-style reference
dependencies such as PEFT, Deepspeed, DINO/EVA helpers, and parity-only pins.

## Citation

### BibTeX

```bibtex
@inproceedings{qin2024posterllama,
  title = {PosterLlama: Bridging Design Ability of Language Model to Content-Aware Layout Generation},
  author = {Qin, Zhe and others},
  booktitle = {European Conference on Computer Vision},
  year = {2024}
}
```
