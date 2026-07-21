# Reproducing Flex-DM Conversion

These commands reproduce the Flex-DM vendor-reference generation, parity checks, checkpoint conversion, and local `from_pretrained` smoke test.

Workflow order: download assets, generate references and goldens, run parity checks, convert checkpoints, then smoke-test `from_pretrained` local loading.

Vendor reference and probe generation disable TensorFlow TF32 with both
`NVIDIA_TF32_OVERRIDE=0` and `--disable-tf32`. In this environment,
`tf.config.experimental.enable_tensor_float_32_execution(False)` alone records
`tf32_enabled=false` but does not change the GPU matmul output.

## Crello

```bash
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset crello --assets data
uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/crello/crello/ours-exp-ft --output-json .cache/flex-dm/reports/crello-ours-exp-ft-tf-vars.json
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/generate_reference_outputs.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/crello-ours-exp-ft --seed 0 --tasks elem,pos,attr,img,txt --num-iter 1,4 --batch-size 1 --max-steps 1 --disable-tf32
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello
uv run --package flex-dm python models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-crello
for task in elem pos attr img txt; do NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/probe_layer_parity.py --mode export-vendor --dataset crello --task "$task" --probe-dir ".cache/flex-dm/probes/crello-$task" --disable-tf32; done
uv pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12
uv pip install --reinstall nvidia-nccl-cu13==2.29.7
for task in elem pos attr img txt; do uv run --package flex-dm python models/flex-dm/scripts/probe_layer_parity.py --mode compare-torch --dataset crello --task "$task" --probe-dir ".cache/flex-dm/probes/crello-$task"; done
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm pytest models/flex-dm/tests/vendor_parity -m vendor_parity
```

## RICO

```bash
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset rico --assets data
uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/rico/rico/ours-exp-ft --output-json .cache/flex-dm/reports/rico-ours-exp-ft-tf-vars.json
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/generate_reference_outputs.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/rico-ours-exp-ft --seed 0 --tasks elem,pos,attr --num-iter 1,4 --batch-size 1 --max-steps 1 --disable-tf32
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset rico --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-rico
uv run --package flex-dm python models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-rico
for task in elem pos attr; do NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/probe_layer_parity.py --mode export-vendor --dataset rico --task "$task" --probe-dir ".cache/flex-dm/probes/rico-$task" --disable-tf32; done
uv pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12
uv pip install --reinstall nvidia-nccl-cu13==2.29.7
for task in elem pos attr; do uv run --package flex-dm python models/flex-dm/scripts/probe_layer_parity.py --mode compare-torch --dataset rico --task "$task" --probe-dir ".cache/flex-dm/probes/rico-$task"; done
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm pytest models/flex-dm/tests/vendor_parity -m vendor_parity
```
