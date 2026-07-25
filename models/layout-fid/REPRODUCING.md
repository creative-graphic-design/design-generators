# Reproducing Layout FID Parity

This guide reproduces the original-implementation agreement checks for the Layout FID evaluator package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

This first package PR reproduces the LayoutFlow-sourced RICO25 and PubLayNet LayoutNet/FIDNet assets. Here `LayoutFlow-sourced` means the evaluator checkpoints and real-distribution statistics are hosted with the LayoutFlow assets; the converted model is still a FIDNet/LayoutNet evaluator, not a LayoutFlow generator.

The LayoutDM and RALF FIDNetV3 variants (`layout-fid-rico25-layoutdm`, `layout-fid-publaynet-layoutdm`, `layout-fid-pku10-ralf`, and `layout-fid-cgl-ralf`) are planned for follow-up slice 2 after this PR merges. Slice 2 differs from this PR in three ways: it uses the `fidnet_v3` architecture instead of LayoutFlow `layoutnet`, targets LayoutDM RICO25/PubLayNet and RALF PKU10/CGL checkpoint families, and prefers original vendor artifacts as the weight source while using the 2024 `creative-graphic-design/layout-fidnet-v3-*` Hub repositories only as secondary tensor-for-tensor cross-check targets.

The `--source` argument in conversion commands is therefore checkpoint provenance, not the generation method being scored. In this PR, `--source layoutflow` selects LayoutFlow-hosted LayoutNet assets; `--source layoutdm` is reserved for the follow-up FIDNetV3 conversion path; RALF is not a selectable conversion source until slice 2 lands.

### Artifact Provenance

The commands below do not download or commit large assets. They assume the following external artifacts have been placed under `vendor/` locally:

| Source family | Download or source location | Local path used by commands |
| --- | --- | --- |
| `layoutflow` | [JulianGuerreiro/LayoutFlow source repository](https://github.com/julianguerreiro/LayoutFlow) and [JulianGuerreiro/LayoutFlow checkpoint host](https://huggingface.co/JulianGuerreiro/LayoutFlow) | `vendor/layout-flow/pretrained/fid_rico.pth.tar`, `vendor/layout-flow/pretrained/fid_publaynet.pth.tar`, and `vendor/layout-flow/pretrained/FIDNet_musig_{val,test}_{rico,publaynet}.pt` |
| `layoutdm` | [CyberAgentAILab/layout-dm release `v1.0.0`](https://github.com/CyberAgentAILab/layout-dm/releases/tag/v1.0.0), asset `layoutdm_starter.zip` | `vendor/layout-dm/download/fid_weights/FIDNetV3/...` after unpacking the archive; not converted in this PR |
| `layoutdm` cross-check | 2024 Hub repos [`layout-fidnet-v3-layoutdm-rico25`](https://huggingface.co/creative-graphic-design/layout-fidnet-v3-layoutdm-rico25) and [`layout-fidnet-v3-layoutdm-publaynet`](https://huggingface.co/creative-graphic-design/layout-fidnet-v3-layoutdm-publaynet) | `model.safetensors`; secondary tensor-for-tensor check in slice 2 |
| `ralf` | Vendored source path `vendor/ralf/image2layout/train/fid/model.py`; slice 2 records the exact original weight release before conversion | PKU10 and CGL FIDNetV3 weights selected in slice 2 |
| `ralf` cross-check | 2024 Hub repos [`layout-fidnet-v3-ralf-pku10`](https://huggingface.co/creative-graphic-design/layout-fidnet-v3-ralf-pku10) and [`layout-fidnet-v3-ralf-cgl`](https://huggingface.co/creative-graphic-design/layout-fidnet-v3-ralf-cgl) | `model.safetensors`; secondary tensor-for-tensor check in slice 2 |

### Prerequisites

Run the commands below from the repository root. The original source under `vendor/` is read-only; downloads and generated artifacts are written under `.cache/layout-fid`.

1. Inspect the expected local assets.

```bash
uv run --package layout-fid python models/layout-fid/scripts/download_original.py
```

2. Generate a deterministic reference batch for CPU parity debugging.

```bash
uv run --package layout-fid python models/layout-fid/scripts/generate_reference_outputs.py \
  --output .cache/layout-fid/references/tiny-layout-batch.pt
```

3. Run the ordinary unit tests.

```bash
uv run --package layout-fid pytest models/layout-fid/tests -m "not vendor_parity"
```

4. Run the gated parity checks against local LayoutFlow assets.

```bash
PARITY_REQUIRE=1 uv run --package layout-fid --extra vendor pytest \
  models/layout-fid/tests/vendor_parity \
  -m vendor_parity -rs
```

5. Convert the LayoutFlow RICO25 evaluator checkpoint and reference statistics.

```bash
uv run --package layout-fid --extra vendor python models/layout-fid/scripts/convert_original_checkpoint.py \
  --source layoutflow \
  --dataset rico25 \
  --checkpoint vendor/layout-flow/pretrained/fid_rico.pth.tar \
  --stats-val vendor/layout-flow/pretrained/FIDNet_musig_val_rico.pt \
  --stats-test vendor/layout-flow/pretrained/FIDNet_musig_test_rico.pt \
  --output-dir .cache/layout-fid/converted/layout-fid-rico25-layoutflow
```

6. Convert the LayoutFlow PubLayNet evaluator checkpoint and reference statistics.

```bash
uv run --package layout-fid --extra vendor python models/layout-fid/scripts/convert_original_checkpoint.py \
  --source layoutflow \
  --dataset publaynet \
  --checkpoint vendor/layout-flow/pretrained/fid_publaynet.pth.tar \
  --stats-val vendor/layout-flow/pretrained/FIDNet_musig_val_publaynet.pt \
  --stats-test vendor/layout-flow/pretrained/FIDNet_musig_test_publaynet.pt \
  --output-dir .cache/layout-fid/converted/layout-fid-publaynet-layoutflow
```

7. Smoke-test local `from_pretrained` loading.

```bash
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py \
  --model-id .cache/layout-fid/converted/layout-fid-rico25-layoutflow
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py \
  --model-id .cache/layout-fid/converted/layout-fid-publaynet-layoutflow
```

Each script supports `--help` with defaults documented:

```bash
uv run --package layout-fid python models/layout-fid/scripts/download_original.py --help
uv run --package layout-fid python models/layout-fid/scripts/generate_reference_outputs.py --help
uv run --package layout-fid python models/layout-fid/scripts/convert_original_checkpoint.py --help
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py --help
```
