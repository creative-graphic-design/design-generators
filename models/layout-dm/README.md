# layout-dm

Diffusers-style conversion package for **LayoutDM: Discrete Diffusion Model for Controllable Layout Generation**.

LayoutDM is a discrete diffusion method for controllable layout generation. This package wraps the released Rico25 and PubLayNet checkpoints in a Diffusers-compatible `LayoutDMPipeline`, while shared output schemas, bbox helpers, labels, and sampling utilities live in `laygen.common`.

Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html

## Installation

From the repository root:

```bash
uv sync --package layout-dm
```

For conversion from the original checkpoint bundle:

```bash
uv sync --package layout-dm --extra convert
```

## Usage

```python
from layout_dm import LayoutDMPipeline

pipe = LayoutDMPipeline.from_pretrained(
    "creative-graphic-design/layoutdm-rico25",
)
out = pipe(batch_size=1, seed=0, sampling="deterministic")

print(out.bbox)    # normalized center xywh boxes
print(out.labels)  # integer category labels
print(out.mask)    # valid element mask
```

Conditional generation uses the same pipeline call with `condition_type`, `bbox`, `labels`, and `mask`:

```python
out = pipe(
    condition_type="label",
    labels=labels,
    bbox=bbox,
    mask=mask,
    seed=0,
)
```

## Checkpoints

| Dataset | Hub ID | Dataset ID |
| --- | --- | --- |
| Rico25 | `creative-graphic-design/layoutdm-rico25` | `creative-graphic-design/rico25` |
| PubLayNet | `creative-graphic-design/layoutdm-publaynet` | `creative-graphic-design/publaynet` |

## Training Data

The converted checkpoints follow the original LayoutDM release:

- Rico25: mobile UI layouts with at most 25 elements.
- PubLayNet: document page layouts with at most 25 elements.

## Parity Results

Local fixtures compare the converted Diffusers pipeline against the original LayoutDM implementation.

| Dataset | Tokenizer exact | Deterministic exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| Rico25 | 125/125 | 125/125 | 0 | 0 |
| PubLayNet | 125/125 | 125/125 | 0 | 0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Run these commands from the repository root unless a block explicitly changes directory. The commands assume CUDA is available as device `0`, the vendored original implementation is present at `vendor/layout-dm`, and generated files may be written under `.cache/` and `models/layout-dm/tests/vendor_parity/fixtures/`.

### 1. Download Original Weights

This downloads `layoutdm_starter.zip` and extracts the original release bundle. The repo-local cache location is `.cache/layout-dm/original`; the extracted starter directory used by later steps is `.cache/layout-dm/original/download`.

```bash
cd models/layout-dm
uv run --package layout-dm python scripts/download_original.py \
  --output-dir ../../.cache/layout-dm/original
cd ../..
```

### 2. Generate Golden Reference Tensors

This writes local-only parity fixtures for each dataset:

- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/tokenizer_io.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/denoiser_forward.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/sample_unconditional.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/meta.json`

```bash
cd models/layout-dm
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python scripts/generate_reference_outputs.py \
  --dataset rico25 \
  --starter-dir ../../.cache/layout-dm/original/download \
  --output-dir tests/vendor_parity/fixtures/rico25 \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1

CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python scripts/generate_reference_outputs.py \
  --dataset publaynet \
  --starter-dir ../../.cache/layout-dm/original/download \
  --output-dir tests/vendor_parity/fixtures/publaynet \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1
cd ../..
```

### 3. Run Parity Tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm pytest models/layout-dm/tests/vendor_parity -m vendor_parity -rs
```

Expected result with the fixtures above:

```text
6 passed
```

### 4. Convert Checkpoints And Smoke Test from_pretrained

The conversion step writes Diffusers pipeline directories under `.cache/layout-dm/converted/`. Each output includes model weights, tokenizer files, scheduler/processor config, and `README.md`.

Expected local output roots:

```text
.cache/layout-dm/converted/layoutdm-rico25/
.cache/layout-dm/converted/layoutdm-publaynet/
```

```bash
cd models/layout-dm
uv run --package layout-dm --extra convert python scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir ../../.cache/layout-dm/original/download \
  --output-dir ../../.cache/layout-dm/converted/layoutdm-rico25

uv run --package layout-dm --extra convert python scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --starter-dir ../../.cache/layout-dm/original/download \
  --output-dir ../../.cache/layout-dm/converted/layoutdm-publaynet
cd ../..
```

Smoke test both converted checkpoints:

```bash
uv run --package layout-dm python - <<'PY'
from pathlib import Path
from layout_dm import LayoutDMPipeline

for dataset in ("rico25", "publaynet"):
    path = Path(".cache/layout-dm/converted") / f"layoutdm-{dataset}"
    pipe = LayoutDMPipeline.from_pretrained(path)
    out = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
    assert out.bbox.shape[-1] == 4
    assert out.labels.shape == out.mask.shape
    assert (path / "README.md").exists()
    print(dataset, out.bbox.shape, out.labels.shape)
PY
```

## Vendor Links

- Original implementation: https://github.com/CyberAgentAILab/layout-dm
- Project page: https://cyberagentailab.github.io/layout-dm/
- Starter checkpoint bundle: https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip

The original implementation is released under the Apache-2.0 license.

## Citation

```bibtex
@inproceedings{inoue2023layout,
  title = {{LayoutDM: Discrete Diffusion Model for Controllable Layout Generation}},
  author = {Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023},
  pages = {10167-10176}
}
```
