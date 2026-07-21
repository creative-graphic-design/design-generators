# Reproducing Flex-DM Conversion

These commands reproduce the Flex-DM vendor-reference generation, parity checks, checkpoint conversion, and local `from_pretrained` smoke test.

Workflow order: download assets, generate references and goldens, run parity checks, convert checkpoints, then smoke-test `from_pretrained` local loading.

## Crello

```bash
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset crello --assets data
uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/crello/crello/ours-exp-ft --output-json .cache/flex-dm/reports/crello-ours-exp-ft-tf-vars.json
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/generate_reference_outputs.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/crello-ours-exp-ft --seed 0 --tasks elem,pos,attr,img,txt --num-iter 1,4
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm pytest models/flex-dm/tests/vendor_parity -m vendor_parity
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello-explicit-ft
uv run --package flex-dm python models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-crello-explicit-ft
```

## RICO

```bash
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset rico --assets data
uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/rico/rico/ours-exp-ft --output-json .cache/flex-dm/reports/rico-ours-exp-ft-tf-vars.json
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/generate_reference_outputs.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/rico-ours-exp-ft --seed 0 --tasks elem,pos,attr --num-iter 1,4
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm pytest models/flex-dm/tests/vendor_parity -m vendor_parity
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-rico-explicit-ft
uv run --package flex-dm python models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-rico-explicit-ft
```
