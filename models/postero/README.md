---
language:
  - en
license: "apache-2.0"
library_name: "pydantic-ai"
pipeline_tag: "other"
tags:
  - "postero"
  - "layout-generation"
  - "content-aware-layout-generation"
  - "cvpr-2025"
datasets:
  - "creative-graphic-design/PKU-PosterLayout"
  - "creative-graphic-design/CGL-Dataset"
---

# Model Card for PosterO

[![paper](https://img.shields.io/static/v1?label=paper&message=CVPR+2025&color=blue&style=flat-square)](https://openaccess.thecvf.com/content/CVPR2025/html/Hsu_PosterO_Structuring_Layout_Trees_to_Enable_Language_Models_in_Generalized_CVPR_2025_paper.html)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&logo=apache&logoColor=white&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=pydantic-ai&color=blue&logo=pydantic&logoColor=white&style=flat-square)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PKU&color=informational&logo=huggingface&logoColor=white&style=flat-square)](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=CGL&color=informational&logo=huggingface&logoColor=white&style=flat-square)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=n%2Fa&color=lightgrey&style=flat-square)

This package builds PosterO-style SVG prompts for content-aware poster layout generation, runs a `pydantic-ai` provider, and parses generated rectangles into the shared layout schema.

## Model Details

### Model Description

PosterO is a CVPR 2025 method for generalized content-aware poster layout generation with structured layout-tree prompts. This package stores prompt/parser configuration and returns `bbox`, `labels`, `mask`, and `id2label` through `laygen.modeling_outputs.LayoutGenerationOutput`.

### Model Sources

- Paper: [PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Hsu_PosterO_Structuring_Layout_Trees_to_Enable_Language_Models_in_Generalized_CVPR_2025_paper.html)
- Source repository: <https://github.com/theKinsley/PosterO-CVPR2025>

## Supported Checkpoints

PosterO has no learned checkpoints in this package. `save_pretrained()` writes prompt and parser state to `postero_config.json`; no `model.safetensors` is produced.

| Artifact | Contents |
| --- | --- |
| `postero_config.json` | prompt structure, parser settings, sampling settings, canvas size, dataset name, and `id2label` |

## Uses

### Direct Use

Use PosterO for offline prompt construction, deterministic parser checks, and provider-backed content-aware poster layout generation.

### Downstream Use

Downstream systems can store prompt configs with `save_pretrained()`, load them with `from_pretrained()`, and pass provider output through the shared layout schema.

### Out-of-Scope Use

The package does not load design-intent detector weights, run vLLM, evaluate full poster metrics, or publish Hub model repositories.

## Bias, Risks, and Limitations

Provider output quality depends on the selected language model and exemplars. The first package scope supports rectangle SVG parsing and stores PKU/CGL label ids exactly as configured.

### Recommendations

Use deterministic fake providers in CI. Treat real provider runs as integration checks, and keep generated provider text outside git.

## How to Get Started with the Model

Install directly from this repository.

```bash
pip install \
  "laygen[agents] @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "postero @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/postero"
```

Use a local checkout for development and offline smoke tests.

PosterO has no learned checkpoints in this package; the local command below loads only prompt configuration.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package postero
uv run --package postero python models/postero/scripts/smoke_from_pretrained.py
```

```python
from pydantic_ai.models.test import TestModel

from postero import PosterOAgent, PosterOConfig
from postero.vendor_parity import fixture_records

model = TestModel(
    custom_output_args={
        "text": '<svg><rect data-label="text_1" x="10" y="20" width="30" height="40"/></svg>'
    }
)
agent = PosterOAgent(model=model, config=PosterOConfig(sample_size=1, n_valid_layouts=1))
query, candidates = fixture_records()
output = agent(query_record=query, candidate_records=candidates)
print(output.bbox.shape)
```

## Training Details

### Training Data

This prompt-config package uses in-memory records for tests. User data can follow the PKU-PosterLayout and CGL label mappings stored in `PosterOConfig`.

### Training Procedure

There is no training procedure in this package because generation is provider-backed prompting over no learned checkpoints.

## Evaluation

### Parity Results

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| Prompt bytes | 1 | SHA-256 metadata match | 1/1 |
| Exemplar selection | 1 | selected id list exact match | 1/1 |
| SVG parser | 1 | label ids and pixel `ltrb` boxes exact match | 1/1 |
| Retry policy | 1 | invalid response skipped before valid SVG | 1/1 |

## Reproducibility

Reproduce the PosterO prompt, selection, parser, retry, and `save_pretrained` checks with [models/postero/REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/postero/REPRODUCING.md).

## License

This package is released under the repository Apache-2.0 license. Source reuse beyond prompt/config behavior requires license review of the source repository.

## Citation

```bibtex
@inproceedings{hsu2025postero,
  title = {PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation},
  author = {Hsu, Hsiao Yuan and Chen, Jianan and Yang, Xi and Hsu, Wen-Huang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2025}
}
```
