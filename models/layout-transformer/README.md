---
language:
  - en
license: "other"
license_name: "review-needed"
license_link: "unknown"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layout-transformer"
  - "layout-generation"
  - "scene-graph-to-layout"
datasets:
  - "COCO"
  - "VG-MSDN"
model-index:
  - name: "LayoutTransformer"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "COCO"
          name: "COCO"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayoutTransformer

[![paper](https://img.shields.io/static/v1?label=paper&message=CVPR+2021&color=blue&style=flat-square)](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=review-needed&color=yellow&style=flat-square)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=COCO&color=informational&style=flat-square)
![dataset](https://img.shields.io/static/v1?label=dataset&message=VG-MSDN&color=informational&style=flat-square)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=bit-exact&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutTransformer](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html), also known as LT-Net, into a 🤗 [`transformers`](https://huggingface.co/docs/transformers/index)-style package for scene-graph-conditioned layout generation.

## Model Details

### Model Description

LayoutTransformer is a scene-graph-to-layout model for natural-image scene layouts. It encodes object and relation tokens, decodes box distributions with Gaussian mixture heads, and optionally refines boxes with visual-textual co-attention. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local integer labels, a valid-element `mask`, and `id2label`.

- **Developed by:** Cheng-Fu Yang, Wan-Cyuan Fan, Fu-En Yang, and Yu-Chiang Frank Wang.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** unknown.

### Model Sources

- **Repository:** [LayoutTransformer repository](https://github.com/davidhalladay/LayoutTransformer)
- **Paper:** [CVPR 2021 paper page](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html)
- **Vendored source:** `vendor/layout-transformer`

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| COCO full LT-Net | [`creative-graphic-design/layout-transformer-coco`](https://huggingface.co/creative-graphic-design/layout-transformer-coco) | not-published |
| VG-MSDN full LT-Net | [`creative-graphic-design/layout-transformer-vg-msdn`](https://huggingface.co/creative-graphic-design/layout-transformer-vg-msdn) | not-published |

## Uses

### Direct Use

Use this package for research inference, checkpoint conversion checks, and vendor-parity validation of relation-conditioned layout generation.

LayoutTransformer accepts scene graph objects and relations, serializes them through the package tokenizer and processor, and returns common `laygen` layout outputs.

### Downstream Use

Generated layouts may feed rendering, scene-composition tools, layout evaluation, or downstream text-to-image systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production scene annotations, object detection output, accessibility metadata, or license-cleared image assets without separate review.

## Bias, Risks, and Limitations

The package follows released checkpoint behavior and the dataset vocabularies used by the original implementation. Layout quality, object coverage, and relation coverage inherit the limits of COCO and VG-MSDN scene graph preprocessing.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints, changing tokenizer metadata, or comparing new results against the original implementation.

## How to Get Started with the Model

Clone this repository, install the workspace member, and run the download and conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-transformer/REPRODUCING.md). Those steps create `.cache/layout-transformer/converted/coco`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-transformer
uv run --package layout-transformer python
```

```python
from layout_transformer import LayoutObject, LayoutRelation, LayoutTransformerPipeline

path = ".cache/layout-transformer/converted/coco"
# After Hub publication: from_pretrained("creative-graphic-design/layout-transformer-coco")
pipe = LayoutTransformerPipeline.from_pretrained(path, local_files_only=True)
out = pipe(
    condition_type="relation",
    objects=[
        LayoutObject(id="person-1", label="person"),
        LayoutObject(id="table-1", label="table"),
    ],
    relations=[
        LayoutRelation(subject="person-1", predicate="left of", object="table-1"),
    ],
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
| COCO | unknown | original LT-Net COCO scene-graph preprocessing |
| VG-MSDN | unknown | original Visual Genome MSDN split and vocabulary |

COCO and VG-MSDN are not yet available in the `creative-graphic-design` Hugging Face org in LT-Net-ready scene-graph form. Parity and conversion scripts therefore follow the original repository's dataset and vocabulary paths.

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Scene graphs are serialized into object, relation, segment, and token-type sequences. Package processors normalize public scene-graph inputs before model execution and convert token-level box predictions back into object-level public layout outputs.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training-time and carbon measurements are unknown.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by checkpoint, dataset, sample index, and seed where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality or explicitly stated numeric tolerance against the vendor path.

### Parity Results

| Dataset | Cases | Compared tensors | Max abs |
| --- | ---: | --- | ---: |
| COCO | 1 | `vocab_logits`, `obj_id_logits`, `token_type_logits`, `coarse_box`, `refine_box` | 0 |
| VG-MSDN | 1 | `vocab_logits`, `obj_id_logits`, `token_type_logits`, `coarse_box`, `refine_box` | 0 |

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-transformer/REPRODUCING.md) for the commands that download vendor assets, prepare dataset metadata, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

LayoutTransformer encodes scene-graph token sequences with object, relation, segment, and token-type embeddings. Its decoder predicts Gaussian mixture parameters for center and size coordinates, and the optional refinement module updates object boxes through attention over decoded layout features.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU because the original path uses CUDA for checkpoint execution.

#### Hardware

CPU is sufficient for import, serialization, and most unit tests. CUDA is required for heavyweight vendor parity against the original implementation.

#### Software

Use `uv run --package layout-transformer ...` from the repository root so workspace dependency sources, optional extras, and package metadata resolve from the correct workspace member.

## License

Repository wrapper code is Apache-2.0. The original implementation has no top-level license file, so converted weight publication is blocked until license provenance is clarified.

## Citation

```bibtex
@inproceedings{yang2021layouttransformer,
  title = {LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity},
  author = {Yang, Cheng-Fu and Fan, Wan-Cyuan and Yang, Fu-En and Wang, Yu-Chiang Frank},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```
