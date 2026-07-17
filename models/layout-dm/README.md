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

## Vendor Parity

Local fixtures compare the converted Diffusers pipeline against the original LayoutDM implementation.

| Dataset | Tokenizer exact | Deterministic exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| Rico25 | 125/125 | 125/125 | 0 | 0 |
| PubLayNet | 125/125 | 125/125 | 0 | 0 |

## Reproducing Vendor Parity

Prerequisites: run from the repository root and use an environment with CUDA when regenerating golden outputs. The commands below use GPU index 5; change it if needed.

1. Download the original starter kit. This creates `.cache/layout-dm/original/layoutdm_starter.zip` and extracts `.cache/layout-dm/original/layoutdm_starter`.

```bash
uv run --package layout-dm python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

2. Generate reference/golden outputs. Outputs are written under `models/layout-dm/tests/vendor_parity/fixtures/<dataset>`.

```bash
for dataset in rico25 publaynet; do
  CUDA_VISIBLE_DEVICES=5 uv run --package layout-dm python models/layout-dm/scripts/generate_reference_outputs.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/layoutdm_starter \
    --output-dir "models/layout-dm/tests/vendor_parity/fixtures/${dataset}" \
    --sampling deterministic \
    --seed 0 \
    --batch-size 1
done
```

3. Convert checkpoints. Outputs are written under `.cache/layout-dm/converted/layoutdm-<dataset>`.

```bash
for dataset in rico25 publaynet; do
  uv run --package layout-dm python models/layout-dm/scripts/convert_original_checkpoint.py \
    --dataset "${dataset}" \
    --seed 0 \
    --starter-dir .cache/layout-dm/original/layoutdm_starter \
    --output-dir ".cache/layout-dm/converted/layoutdm-${dataset}"
done
```

4. Run parity.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --package layout-dm pytest \
  models/layout-dm/tests/vendor_parity \
  -q -m vendor_parity -s
```

5. Run a `from_pretrained` smoke against a converted checkpoint.

```bash
uv run --package layout-dm python - <<'PY'
from layout_dm import LayoutDMPipeline

pipe = LayoutDMPipeline.from_pretrained(".cache/layout-dm/converted/layoutdm-rico25")
out = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
print(out.sequences.shape)
PY
```

## Model Cards

Converted checkpoint directories include a model card as `README.md`.

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
