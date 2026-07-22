---
license: gpl-3.0
library_name: transformers
pipeline_tag: layout-generation
tags:
  - layout-generation
  - floorplan-generation
  - housegan
datasets:
  - housegan-floorplan-vectorized
---

# House-GAN

House-GAN is an ECCV 2020 relational GAN for graph-constrained floorplan layout generation. This package ports the released generator into a Transformers-style `PreTrainedModel` plus `LayoutGenerationPipeline`.

## Usage

```python
from housegan import HouseGanPipeline

pipe = HouseGanPipeline.from_pretrained("creative-graphic-design/housegan-floorplan-d")
output = pipe(
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
```

## Supported Checkpoints

| Target set | Intended Hub id | Status |
| --- | --- | --- |
| A | `creative-graphic-design/housegan-floorplan-a` | conversion script ready; asset inventory pending |
| B | `creative-graphic-design/housegan-floorplan-b` | conversion script ready; asset inventory pending |
| C | `creative-graphic-design/housegan-floorplan-c` | conversion script ready; asset inventory pending |
| D | `creative-graphic-design/housegan-floorplan-d` | conversion script ready; asset inventory pending |
| E | `creative-graphic-design/housegan-floorplan-e` | conversion script ready; asset inventory pending |

## Training Data

The original implementation uses the vectorized House-GAN floorplan dataset from the upstream Dropbox release. No `creative-graphic-design/*` mirror is documented for these assets yet, so parity and conversion commands use a local copy of the original distribution.

## Parity Results

| Check | Cases | Match criterion | Result |
| --- | ---: | --- | --- |
| State-dict key mapping | synthetic model state dict | strict key prefix validation | passing in unit tests |
| Forward masks | external vendor artifacts | bitwise equality when generated assets are present | gated by `vendor_parity`; artifacts not committed |
| Mask-to-box postprocess | unit masks | vendor inclusive `x1+1` / `y1+1` behavior | passing in unit tests |
| Public layout schema | synthetic graph | `assert_layout_output_schema` | passing in unit tests |

## Reproducibility

Use `REPRODUCING.md` to reproduce the original-implementation agreement checks against local Dropbox assets and generated parity artifacts.

## License

The upstream repository carries GPL-3.0 plus a research-purpose notice. Do not upload converted weights to the Hub until maintainers explicitly approve redistribution under that license posture.

## Citation

```bibtex
@inproceedings{nauata2020housegan,
  title = {House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation},
  author = {Nauata, Nelson and Chang, Kai-Hung and Cheng, Chin-Yi and Mori, Greg and Furukawa, Yasutaka},
  booktitle = {ECCV},
  year = {2020}
}
```

Original implementation: `vendor/housegan`.
