# LayoutFlow

Diffusers-style conversion package for **LayoutFlow: Flow Matching for Layout Generation** (ECCV 2024). The original method models layouts as continuous flow-matching trajectories over normalized boxes and analog-bit category labels, then solves the generation path with an increasing-time ODE from `t=0` to `t=1`.

- Paper: https://arxiv.org/abs/2403.18187
- Project page: https://julianguerreiro.github.io/layoutflow/
- Original implementation: https://github.com/julianguerreiro/LayoutFlow
- Original checkpoint host: https://huggingface.co/JulianGuerreiro/LayoutFlow

## Installation

From the repository root, use the uv workspace package:

```bash
uv run --package layout-flow pytest
```

For conversion or parity checks against the original Lightning checkpoints, include the vendor extra:

```bash
uv run --package layout-flow --extra vendor pytest models/layout-flow/tests/vendor_parity -m vendor_parity -rs
```

## Usage

```python
from layout_flow import LayoutFlowPipeline

pipe = LayoutFlowPipeline.from_pretrained(
    "creative-graphic-design/layout-flow-publaynet"
)
out = pipe(batch_size=1, num_elements=8, seed=0, num_inference_steps=100)

print(out.bbox)    # normalized center xywh, shape (B, S, 4)
print(out.labels)  # dataset-local label ids, shape (B, S)
print(out.mask)    # True for valid elements, shape (B, S)
print(out.id2label)
```

Conditioning aliases from the original implementation are accepted at the pipeline boundary:

```python
out = pipe(
    condition_type="cat_cond",  # canonical alias: "label"
    labels=[[1, 2, 3]],
    bbox=[[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.1, 0.2]]],
    mask=[[True, True, True]],
    seed=0,
)
```

## Checkpoints

| Hub id | Source checkpoint | Dataset |
| --- | --- | --- |
| `creative-graphic-design/layout-flow-rico25` | `checkpoint_RICO_LayoutFlow.ckpt` | `creative-graphic-design/Rico` |
| `creative-graphic-design/layout-flow-publaynet` | `checkpoint_PubLayNet_LayoutFlow.ckpt` | `creative-graphic-design/PubLayNet` |

## Data

The released LayoutFlow checkpoints were trained on the RICO and PubLayNet splits distributed by the original authors. In this repository, dataset metadata is aligned to the org datasets:

- `creative-graphic-design/Rico`
- `creative-graphic-design/PubLayNet`

Public outputs use normalized center `xywh` boxes in `[0, 1]`; padding is represented by `mask`, not by a public label id.

## Vendor Parity

Parity is tested against the original vendor `LayoutDMBackbone` and `torchdyn.NeuralODE(..., solver="euler")` path using the released checkpoints.

| Dataset | Vector field max abs | Vector field max rel | Euler trajectory max abs | Euler trajectory max rel |
| --- | ---: | ---: | ---: | ---: |
| PubLayNet | 0.0 | 0.0 | 0.0 | 0.0 |
| RICO25 | 0.0 | 0.0 | 0.0 | 0.0 |

## Reproducing vendor parity

Run the commands below from the repository root. The original source under `vendor/layout-flow` is read-only; downloads and generated artifacts are written under `.cache/layout-flow`.

1. Download the original checkpoints. This creates `.cache/layout-flow/original/checkpoints/checkpoint_PubLayNet_LayoutFlow.ckpt` and `.cache/layout-flow/original/checkpoints/checkpoint_RICO_LayoutFlow.ckpt`.

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py
```

2. Generate vendor golden/reference tensors. `CUDA_VISIBLE_DEVICES=3` selects the requested GPU; the script writes `.cache/layout-flow/golden/*_vendor_vector_field.pt` and `.cache/layout-flow/golden/summary.json`.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layout-flow --extra vendor python models/layout-flow/scripts/generate_reference_outputs.py --dataset all
```

3. Run the parity pytest suite against both released checkpoints.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layout-flow --extra vendor pytest models/layout-flow/tests/vendor_parity -m vendor_parity -rs
```

4. Convert both checkpoints to local Diffusers pipeline directories. These commands write `.cache/layout-flow/converted/publaynet` and `.cache/layout-flow/converted/rico25`, each with a `README.md`.

```bash
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset publaynet
uv run --package layout-flow --extra vendor python models/layout-flow/scripts/convert_original_checkpoint.py --dataset rico25
```

5. Smoke-test local `from_pretrained` loading.

```bash
uv run --package layout-flow python - <<'PY'
from layout_flow import LayoutFlowPipeline

for dataset in ["publaynet", "rico25"]:
    pipe = LayoutFlowPipeline.from_pretrained(f".cache/layout-flow/converted/{dataset}")
    out = pipe(batch_size=1, num_elements=4, seed=0, num_inference_steps=2)
    print(dataset, out.bbox.shape, out.labels.shape, type(out).__name__)
PY
```

Each script supports `--help` with defaults documented:

```bash
uv run --package layout-flow python models/layout-flow/scripts/download_original.py --help
uv run --package layout-flow python models/layout-flow/scripts/generate_reference_outputs.py --help
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py --help
```

## License

The original implementation is MIT licensed.

## Citation

```bibtex
@inproceedings{guerreiro2024layoutflow,
  title={LayoutFlow: Flow Matching For Layout Generation},
  author={Guerreiro, Julian Jorge Andrade and Inoue, Naoto and Masui, Kento and Otani, Mayu and Nakayama, Hideki},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2024},
  organization={Springer}
}
```
