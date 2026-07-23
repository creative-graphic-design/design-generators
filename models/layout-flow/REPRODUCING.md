# Reproducing LayoutFlow Parity

This guide reproduces the original-implementation agreement checks for the LayoutFlow package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

### Prerequisites

```bash
git submodule update --init vendor/layout-flow
```

Run the commands below from the repository root. The original source under `vendor/layout-flow` is read-only; downloads and generated artifacts are written under `.cache/layout-flow`.

1. Download the original checkpoints. This creates `.cache/layout-flow/original/checkpoints/checkpoint_PubLayNet_LayoutFlow.ckpt` and `.cache/layout-flow/original/checkpoints/checkpoint_RICO_LayoutFlow.ckpt`.

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py
```

2. Generate vendor golden/reference tensors. Set `CUDA_VISIBLE_DEVICES=<gpu-index>` to the GPU you want to use. The script writes `.cache/layout-flow/golden/*_vendor_vector_field.pt` and `.cache/layout-flow/golden/summary.json`.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-flow --extra vendor python models/layout-flow/scripts/generate_reference_outputs.py --dataset all
```

3. Run the parity pytest suite against both released checkpoints.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-flow --extra vendor pytest models/layout-flow/tests/vendor_parity -m vendor_parity -rs
```

4. Run the training-parity S0-S2 hooks. See `models/layout-flow/TRAINING.md` for the training config list, seed modes, and the full training-parity rerun notes.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package layout-flow --extra training --extra vendor pytest \
  models/layout-flow/tests/vendor_parity/test_layout_flow_training_parity.py \
  -m "vendor_parity and training" -rs
```

5. Convert both checkpoints to local [`🧨 diffusers`](https://huggingface.co/docs/diffusers/index) pipeline directories. These commands write `.cache/layout-flow/converted/publaynet` and `.cache/layout-flow/converted/rico25`, each with a `README.md`.

```bash
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset publaynet
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset rico25
```

6. Convert a newly trained Lightning checkpoint after training. The converter accepts both original vendor `model.*` keys and package-local training `model.backbone.*` keys. See `models/layout-flow/TRAINING.md` for the training launch commands.

```bash
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/layout-flow/training-runs/rico25/checkpoints/<ckpt>.ckpt \
  --output-dir .cache/layout-flow/converted-trained/rico25
```

7. Smoke-test local `from_pretrained` loading.

```bash
uv run --package layout-flow python models/layout-flow/scripts/smoke_from_pretrained.py \
  --path .cache/layout-flow/converted/publaynet \
  --path .cache/layout-flow/converted/rico25
```

Each script supports `--help` with defaults documented:

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py --help
uv run --package layout-flow python models/layout-flow/scripts/generate_reference_outputs.py --help
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py --help
uv run --package layout-flow --extra training python -m layout_flow.training.cli --help
```
