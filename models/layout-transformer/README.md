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

pipe = LayoutTransformerPipeline.from_pretrained(".cache/layout-transformer/converted/coco")
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
| COCO full LT-Net | 1 | vocab logits, object-id logits, token-type logits, coarse boxes, refined boxes | bit-exact, `max_abs=0.0` | pass |
| VG-MSDN full LT-Net | 1 | vocab logits, object-id logits, token-type logits, coarse boxes, refined boxes | bit-exact, `max_abs=0.0` | pass |

## Reproducibility

The commands below reproduce the original-implementation agreement checks by downloading vendor assets, preparing dataset metadata, converting checkpoints, exporting references on GPU 2, running the gated parity tests, and smoke-testing local `from_pretrained`.

```bash
uv run --package layout-transformer python models/layout-transformer/scripts/download_weights.py \
  --output-dir .cache/layout-transformer/vendor
```

```bash
mkdir -p .cache/layout-transformer/data/coco
curl -L -o .cache/layout-transformer/data/coco/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -L -o .cache/layout-transformer/data/coco/stuff_trainval2017.zip \
  http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip
(
  cd .cache/layout-transformer/data/coco
  unzip -o annotations_trainval2017.zip \
    annotations/instances_train2017.json annotations/instances_val2017.json
  unzip -o stuff_trainval2017.zip stuff_train2017.json stuff_val2017.json
  mv annotations/instances_train2017.json annotations/instances_val2017.json .
  rmdir annotations
)
```

```bash
mkdir -p .cache/layout-transformer/data/vg_msdn/visual_genome
uv run --with gdown gdown \
  https://drive.google.com/uc?id=1WjetLwwH3CptxACrXnc1NCcccWUVDO76 \
  -O .cache/layout-transformer/data/vg_msdn/vg_msdn.zip
unzip -o .cache/layout-transformer/data/vg_msdn/vg_msdn.zip \
  -d .cache/layout-transformer/data/vg_msdn/visual_genome
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layout-transformer --extra vendor \
  --with coloredlogs --with bounding-box --with opencv-python-headless \
  --with matplotlib --with scikit-image --with scikit-learn --with pycocotools \
  --with joblib --with tqdm --with tensorboard \
  python models/layout-transformer/scripts/export_reference.py \
  --vendor-root vendor/layout-transformer \
  --cfg-path vendor/layout-transformer/configs/coco/coco_seq2seq_v9_ablation_4.yaml \
  --checkpoint-path .cache/layout-transformer/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth \
  --data-dir .cache/layout-transformer/data/coco \
  --dataset-name coco \
  --sample-indices 0 \
  --output-dir .cache/layout-transformer/reference/coco \
  --seed 0
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layout-transformer --extra vendor \
  --with coloredlogs --with bounding-box --with opencv-python-headless \
  --with matplotlib --with scikit-image --with scikit-learn --with pycocotools \
  --with joblib --with tqdm --with tensorboard \
  python models/layout-transformer/scripts/export_reference.py \
  --vendor-root vendor/layout-transformer \
  --cfg-path vendor/layout-transformer/configs/vg_msdn/vg_msdn_seq2seq_v24.yaml \
  --checkpoint-path .cache/layout-transformer/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth \
  --data-dir .cache/layout-transformer/data/vg_msdn/visual_genome \
  --dataset-name vg_msdn \
  --sample-indices 0 \
  --output-dir .cache/layout-transformer/reference/vg_msdn \
  --seed 0
```

```bash
uv run --package layout-transformer python models/layout-transformer/scripts/convert_original_checkpoint.py \
  --checkpoint-path .cache/layout-transformer/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth \
  --cfg-path vendor/layout-transformer/configs/coco/coco_seq2seq_v9_ablation_4.yaml \
  --vocab-path .cache/layout-transformer/data/coco/object_pred_id2name.json \
  --dataset-name coco \
  --output-dir .cache/layout-transformer/converted/coco
```

```bash
uv run --package layout-transformer python models/layout-transformer/scripts/convert_original_checkpoint.py \
  --checkpoint-path .cache/layout-transformer/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth \
  --cfg-path vendor/layout-transformer/configs/vg_msdn/vg_msdn_seq2seq_v24.yaml \
  --vocab-path .cache/layout-transformer/data/vg_msdn/visual_genome/object_pred_id2name.json \
  --dataset-name vg_msdn \
  --output-dir .cache/layout-transformer/converted/vg_msdn
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layout-transformer pytest \
  models/layout-transformer/tests/vendor_parity -m vendor_parity
```

```bash
uv run --package layout-transformer python - <<'PY'
from layout_transformer import LayoutObject, LayoutTransformerPipeline

pipe = LayoutTransformerPipeline.from_pretrained(".cache/layout-transformer/converted/coco", local_files_only=True)
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
