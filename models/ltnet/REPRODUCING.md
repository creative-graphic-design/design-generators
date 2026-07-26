# Reproducing LT-Net Parity

This guide reproduces the original-implementation agreement checks for the LT-Net package.

Workflow order: download assets, prepare dataset metadata, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

Run these commands from the repository root unless a block explicitly changes directory. The commands assume CUDA is available as device `2`, the vendored original implementation is present at `vendor/layout-transformer`, and generated files may be written under `.cache/`.

Prerequisite:

```bash
git submodule update --init vendor/layout-transformer
```

### 1. Download Original Weights

This downloads the original Google Drive folder. The repo-local cache location used by later steps is `.cache/ltnet/vendor`.

```bash
uv run --package ltnet python models/ltnet/scripts/download_weights.py \
  --output-dir .cache/ltnet/vendor
```

Expected local checkpoint paths:

```text
.cache/ltnet/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth
.cache/ltnet/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth
```

### 2. Prepare Dataset Metadata

COCO reference export needs the original annotation files and COCO-Stuff json files.

```bash
mkdir -p .cache/ltnet/data/coco
curl -L -o .cache/ltnet/data/coco/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -L -o .cache/ltnet/data/coco/stuff_trainval2017.zip \
  http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip
```

```bash
(
  cd .cache/ltnet/data/coco
  unzip -o annotations_trainval2017.zip \
    annotations/instances_train2017.json annotations/instances_val2017.json
  unzip -o stuff_trainval2017.zip stuff_train2017.json stuff_val2017.json
  mv annotations/instances_train2017.json annotations/instances_val2017.json .
  rmdir annotations
)
```

VG-MSDN reference export uses the original Visual Genome MSDN bundle.

```bash
mkdir -p .cache/ltnet/data/vg_msdn/visual_genome
uv run --with gdown gdown \
  https://drive.google.com/uc?id=1WjetLwwH3CptxACrXnc1NCcccWUVDO76 \
  -O .cache/ltnet/data/vg_msdn/vg_msdn.zip
unzip -o .cache/ltnet/data/vg_msdn/vg_msdn.zip \
  -d .cache/ltnet/data/vg_msdn/visual_genome
```

### 3. Generate Golden Reference Tensors

This writes local-only reference fixtures under `.cache/ltnet/reference/<dataset>/`.

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package ltnet --extra vendor \
  --with coloredlogs --with bounding-box --with opencv-python-headless \
  --with matplotlib --with scikit-image --with scikit-learn --with pycocotools \
  --with joblib --with tqdm --with tensorboard \
  python models/ltnet/scripts/export_reference.py \
  --vendor-root vendor/layout-transformer \
  --cfg-path vendor/layout-transformer/configs/coco/coco_seq2seq_v9_ablation_4.yaml \
  --checkpoint-path .cache/ltnet/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth \
  --data-dir .cache/ltnet/data/coco \
  --dataset-name coco \
  --sample-indices 0 \
  --output-dir .cache/ltnet/reference/coco \
  --seed 0
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package ltnet --extra vendor \
  --with coloredlogs --with bounding-box --with opencv-python-headless \
  --with matplotlib --with scikit-image --with scikit-learn --with pycocotools \
  --with joblib --with tqdm --with tensorboard \
  python models/ltnet/scripts/export_reference.py \
  --vendor-root vendor/layout-transformer \
  --cfg-path vendor/layout-transformer/configs/vg_msdn/vg_msdn_seq2seq_v24.yaml \
  --checkpoint-path .cache/ltnet/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth \
  --data-dir .cache/ltnet/data/vg_msdn/visual_genome \
  --dataset-name vg_msdn \
  --sample-indices 0 \
  --output-dir .cache/ltnet/reference/vg_msdn \
  --seed 0
```

### 4. Convert Checkpoints

The conversion step writes [`🤗transformers`](https://huggingface.co/docs/transformers/index) checkpoint directories under `.cache/ltnet/converted/`.

```bash
uv run --package ltnet python models/ltnet/scripts/convert_original_checkpoint.py \
  --checkpoint-path .cache/ltnet/vendor/coco_F_seq2seq_v9_ablation_4/checkpoint_50_0.44139538748348955.pth \
  --cfg-path vendor/layout-transformer/configs/coco/coco_seq2seq_v9_ablation_4.yaml \
  --vocab-path .cache/ltnet/data/coco/object_pred_id2name.json \
  --dataset-name coco \
  --output-dir .cache/ltnet/converted/coco
```

```bash
uv run --package ltnet python models/ltnet/scripts/convert_original_checkpoint.py \
  --checkpoint-path .cache/ltnet/vendor/vg_msdn_F_seq2seq_v24/checkpoint_50_0.16316922369277578.pth \
  --cfg-path vendor/layout-transformer/configs/vg_msdn/vg_msdn_seq2seq_v24.yaml \
  --vocab-path .cache/ltnet/data/vg_msdn/visual_genome/object_pred_id2name.json \
  --dataset-name vg_msdn \
  --output-dir .cache/ltnet/converted/vg_msdn
```

Expected local output roots:

```text
.cache/ltnet/converted/coco/
.cache/ltnet/converted/vg_msdn/
```

### 5. Run Vendor Parity Tests

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package ltnet pytest \
  models/ltnet/tests/vendor_parity -m vendor_parity -rs
```

Expected result with the fixtures above:

```text
2 passed
```

### 6. Smoke Test from_pretrained

Smoke test both converted checkpoints:

```bash
uv run --package ltnet python models/ltnet/scripts/smoke_from_pretrained.py \
  --path .cache/ltnet/converted/coco \
  --path .cache/ltnet/converted/vg_msdn
```
