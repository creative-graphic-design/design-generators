---
language:
  - en
license: "gpl-3.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "housegan"
  - "layout-generation"
  - "floorplan-generation"
datasets:
  - "housegan-floorplan-vectorized"
model-index:
  - name: "House-GAN"
    results:
      - task:
          type: "other"
          name: "Graph-constrained floorplan layout generation"
        dataset:
          type: "housegan-floorplan-vectorized"
          name: "House-GAN floorplan vectorized"
          split: "target set D parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "bit-exact with local assets"
            name: "Vendor parity"
---

# Model Card for House-GAN

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2003.06988&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2003.06988)
![venue](https://img.shields.io/static/v1?label=venue&message=ECCV+2020&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=GPL-3.0&color=orange&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=housegan-floorplan-vectorized&color=informational&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [House-GAN](https://arxiv.org/abs/2003.06988), a graph-constrained floorplan layout generator, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package.

## Model Details

### Model Description

House-GAN predicts room masks from a room-relation graph and decodes those masks into normalized layout boxes. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer room labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Nelson Nauata, Kai-Hung Chang, Chin-Yi Cheng, Greg Mori, and Yasutaka Furukawa.
- **Shared by:** creative-graphic-design.
- **Model type:** graph-constrained floorplan layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** gpl-3.0.

### Model Sources

- **Repository:** [House-GAN repository](https://github.com/ennauata/housegan)
- **Paper:** [House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation](https://arxiv.org/abs/2003.06988)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| Target set A | [`creative-graphic-design/housegan-floorplan-a`](https://huggingface.co/creative-graphic-design/housegan-floorplan-a) | not-published |
| Target set B | [`creative-graphic-design/housegan-floorplan-b`](https://huggingface.co/creative-graphic-design/housegan-floorplan-b) | not-published |
| Target set C | [`creative-graphic-design/housegan-floorplan-c`](https://huggingface.co/creative-graphic-design/housegan-floorplan-c) | not-published |
| Target set D | [`creative-graphic-design/housegan-floorplan-d`](https://huggingface.co/creative-graphic-design/housegan-floorplan-d) | not-published |
| Target set E | [`creative-graphic-design/housegan-floorplan-e`](https://huggingface.co/creative-graphic-design/housegan-floorplan-e) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of relation-conditioned floorplan generation.

Conditional generation accepts a scene graph with room nodes and adjacency relations:

```python
from housegan import HouseGanPipeline

pipe = HouseGanPipeline.from_pretrained("creative-graphic-design/housegan-floorplan-d")
out = pipe(
    condition_type="relation",
    scene_graph={
        "nodes": [
            {"id": 0, "label": "living_room"},
            {"id": 1, "label": "kitchen"},
        ],
        "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
    },
    seed=0,
)
print(out.bbox, out.labels, out.mask)
```

Model/processor usage is also available for lower-level checks:

```python
from housegan import HouseGanGenerator, HouseGanProcessor

model = HouseGanGenerator.from_pretrained(
    "creative-graphic-design/housegan-floorplan-d"
).eval()
processor = HouseGanProcessor.from_pretrained(
    "creative-graphic-design/housegan-floorplan-d"
)
encoded = processor(
    condition_type="relation",
    scene_graph={"nodes": [{"id": 0, "label": "living_room"}]},
)
out = model(**encoded)
print(out.masks.shape)
```

### Downstream Use

Generated layouts may feed floorplan visualization, relation-graph studies, or downstream design tooling after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as code-compliant architectural plans, safety-reviewed building documents, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoint format, graph vocabulary, and vectorized floorplan dataset. Dataset coverage, room taxonomy, and geometric quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite with the released assets before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

Install House-GAN with its required `laygen` workspace dependency:

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "housegan @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/housegan"
```

Run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/housegan/REPRODUCING.md) to create `.cache/housegan/converted/housegan-floorplan-d`. For local conversion workflows, clone the repository and install the workspace member:

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package housegan
uv run --package housegan python
```

```python
from housegan import HouseGanPipeline

path = ".cache/housegan/converted/housegan-floorplan-d"
# After Hub publication: from_pretrained("creative-graphic-design/housegan-floorplan-d")
pipe = HouseGanPipeline.from_pretrained(path, local_files_only=True)
out = pipe(
    condition_type="relation",
    scene_graph={"nodes": [{"id": 0, "label": "living_room"}]},
    seed=0,
)

print(out.bbox)
print(out.labels)
print(out.mask)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| House-GAN floorplan vectorized | housegan-floorplan-vectorized | upstream Dropbox assets; graph nodes are room categories and graph edges encode room adjacency |

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Room labels and relations are normalized into complete graph tensors before generation. Generated masks are decoded to vendor `ltrb` boxes and then converted to the public normalized center `xywh` schema.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures from the released House-GAN checkpoint and vectorized floorplan assets. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is scoped to target set D, fixed seed `0`, relation conditioning, and three selected floorplan graphs when the external assets are present.

#### Metrics

Metrics are exact key mapping checks, exact model-output checks against regenerated vendor fixtures, and exact public postprocessing checks for `bbox`, `labels`, and `mask`.

### Parity Results

| Check | Cases | Match criterion | Result |
| --- | ---: | --- | --- |
| State-dict key mapping | 1 synthetic model state dict | strict key prefix validation | passed |
| Forward masks | 3 target-set D graphs from `exp_demo_D_500000.pth` | bitwise equality against regenerated vendor fixture | passed |
| Mask-to-box postprocessing | 1 unit mask | vendor inclusive `x1+1` / `y1+1` behavior | passed |
| Public layout schema | 1 synthetic graph | `assert_layout_output_schema` | passed |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/housegan/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No new model training is performed by this conversion package. Conversion and parity costs depend on local hardware and the selected checkpoint.

## Technical Specifications

### Model Architecture and Objective

House-GAN loads the released graph-conditioned generator checkpoint. The generator embeds room labels, relation edges, and latent noise, applies convolutional message passing over the room graph, and predicts one mask per room.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and smoke tests. CUDA is recommended for heavyweight vendor parity against the released checkpoint.

#### Software

Use `uv run --package housegan ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository code is licensed under Apache-2.0. Upstream House-GAN code and checkpoint redistribution follow the upstream GPL-3.0 license posture, and converted weights should not be uploaded until maintainers approve publication.

## Citation

```bibtex
@inproceedings{nauata2020housegan,
  title = {House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation},
  author = {Nauata, Nelson and Chang, Kai-Hung and Cheng, Chin-Yi and Mori, Greg and Furukawa, Yasutaka},
  booktitle = {ECCV},
  year = {2020}
}
```
