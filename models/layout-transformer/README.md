---
library_name: transformers
pipeline_tag: text-to-image
tags:
  - layout-generation
  - scene-graph-to-layout
  - layout-transformer
license: unknown
---

# LayoutTransformer

LayoutTransformer, also known as LT-Net, is a CVPR 2021 scene-graph-to-layout model. This package provides a Transformers-style `PreTrainedModel`, a saved scene-graph tokenizer/processor, and a `laygen.pipelines.LayoutGenerationPipeline` subclass for relation-conditioned layout generation.

## Install

```bash
uv sync --all-packages
```

## Usage

```python
from layout_transformer import LayoutObject, LayoutRelation, LayoutTransformerPipeline

pipe = LayoutTransformerPipeline.from_pretrained("./converted/layout-transformer-coco")
output = pipe(
    condition_type="relation",
    objects=[
        LayoutObject(id="person-1", label="person"),
        LayoutObject(id="table-1", label="table"),
    ],
    relations=[
        LayoutRelation(subject="person-1", predicate="left of", object="table-1"),
    ],
)
```

`output.bbox` is normalized center `xywh`, `output.labels` contains dataset-local object ids, and `output.mask` marks valid elements.

## Supported Checkpoints

| Dataset | Intended Hub id | Source |
| --- | --- | --- |
| COCO | `creative-graphic-design/layout-transformer-coco` | Original Google Drive checkpoint |
| VG-MSDN | `creative-graphic-design/layout-transformer-vg-msdn` | Original Google Drive checkpoint |

Hub publication is blocked until the original implementation license provenance is resolved.

## Datasets

COCO and VG-MSDN are not currently hosted in the `creative-graphic-design` Hugging Face org in LT-Net-compatible scene-graph form. Until those imports exist, parity and conversion use the original repository's distributed data and vocab assets.

## Parity Results

| Checkpoint | Cases | Compared | Match criterion | Result |
| --- | ---: | --- | --- | --- |
| COCO full LT-Net | 0 | logits, coarse boxes, refined boxes | pending vendor reference export | pending |
| VG-MSDN full LT-Net | 0 | logits, coarse boxes, refined boxes | pending vendor reference export | pending |

## Reproducibility

The commands below reproduce the original-implementation agreement checks by downloading vendor assets, exporting references on GPU 2, running the gated parity tests, converting checkpoints, and smoke-testing local `from_pretrained`.

```bash
uv run --package layout-transformer python models/layout-transformer/scripts/download_weights.py \
  --output-dir artifacts/layout-transformer/vendor
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layout-transformer python models/layout-transformer/scripts/export_reference.py \
  --vendor-root vendor/layout-transformer \
  --cfg-path vendor/layout-transformer/configs/vg_msdn/vg_msdn_seq2seq_v24.yaml \
  --checkpoint-path artifacts/layout-transformer/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth \
  --data-dir artifacts/layout-transformer/data/vg_msdn/visual_genome \
  --dataset-name vg_msdn \
  --sample-indices 0 1 2 \
  --output-dir artifacts/layout-transformer/reference/vg_msdn \
  --seed 0
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layout-transformer pytest models/layout-transformer/tests/vendor_parity -m vendor_parity
```

```bash
uv run --package layout-transformer python models/layout-transformer/scripts/convert_original_checkpoint.py \
  --checkpoint-path artifacts/layout-transformer/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth \
  --cfg-path vendor/layout-transformer/configs/coco/coco_seq2seq_v9_ablation_4.yaml \
  --vocab-path artifacts/layout-transformer/vendor/object_pred_id2name.json \
  --dataset-name coco \
  --output-dir converted/layout-transformer-coco
```

```bash
uv run --package layout-transformer python - <<'PY'
from layout_transformer import LayoutObject, LayoutTransformerPipeline

pipe = LayoutTransformerPipeline.from_pretrained("converted/layout-transformer-coco", local_files_only=True)
out = pipe(objects=[LayoutObject(id="a", label=0)])
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

## License

The vendor repository has no top-level license file. Do not publish converted weights until license provenance is clarified.

## Citation

```bibtex
@inproceedings{yang2021layouttransformer,
  title = {LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity},
  author = {Yang, Cheng-Fu and Fan, Wan-Cyuan and Yang, Fu-En and Wang, Yu-Chiang Frank},
  booktitle = {CVPR},
  year = {2021}
}
```

## Original Implementation

The original implementation is vendored read-only under `vendor/layout-transformer`.
