# LACE

Diffusers-style LACE package for the released PubLayNet and Rico25 layout
generation checkpoints. LACE is the continuous diffusion model from "Towards
Aligned Layout Generation via Diffusion Model with Aesthetic Constraints"
(https://arxiv.org/abs/2402.04754).

This package converts the original implementation checkpoints into a
`LacePipeline` that generates normalized center `xywh` boxes, category labels,
and masks. Shared layout labels, bbox helpers, and output dataclasses come from
`laygen.common`.

## Installation

Use the repository workspace environment:

```bash
uv sync
```

Run package commands with the workspace package form:

```bash
uv run --package lace python -c "from lace import LacePipeline"
```

## Usage

Load a converted checkpoint from the Hub and run unconditional generation:

```python
from lace import LacePipeline

pipe = LacePipeline.from_pretrained("creative-graphic-design/lace-publaynet")
out = pipe(batch_size=1, seed=0, num_inference_steps=100)
print(out.bbox, out.labels, out.mask)
```

Convert a local original checkpoint. The output directory includes `README.md`
alongside the pipeline files:

```bash
uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output .cache/lace/converted/lace-publaynet
```

## Checkpoints

| Dataset | Hub id | Original checkpoint | Status |
| --- | --- | --- | --- |
| PubLayNet | `creative-graphic-design/lace-publaynet` | `publaynet_best.pt` | Converted and parity checked |
| Rico13 | `creative-graphic-design/lace-rico13` | `rico13_best.pt` | Supported by code; public vendor checkpoint was not present in `model.tar.gz` |
| Rico25 | `creative-graphic-design/lace-rico25` | `rico25_best.pt` | Converted and parity checked |

## Training Data

The original LACE project trains on PubLayNet and Rico annotations prepared as
max-25 layout sequences.

| Dataset | Dataset id |
| --- | --- |
| PubLayNet | `creative-graphic-design/publaynet` |
| Rico13 | `creative-graphic-design/rico13` |
| Rico25 | `creative-graphic-design/rico25` |

## Parity Results

Local parity tests use `vendor/lace/util/backbone.py` and the original
checkpoints from `puar-playground/LACE`.

| Dataset | Test | Shape | Max abs | rtol | atol |
| --- | --- | ---: | ---: | ---: | ---: |
| PubLayNet | Denoiser logits | `(2, 25, 10)` | 0.0 | 0 | 0 |
| Rico25 | Denoiser logits | `(2, 25, 30)` | 0.0 | 0 | 0 |

Conversion smoke tests pass for PubLayNet and Rico25. Rico13 conversion parity
is skipped unless `.cache/lace/original/model/rico13_best.pt` is supplied,
because the public `model.tar.gz` archive contains only `publaynet_best.pt` and
`rico25_best.pt`.

## Reproducibility

This section reproduces the parity verification against the original implementation.

Run the following commands from the repository root. They keep downloaded
weights under `.cache/lace/original`, local reference metadata under
`.cache/lace/reference`, and converted pipelines under `.cache/lace/converted`.
The commands use `CUDA_VISIBLE_DEVICES=2`; change that value if your CUDA device
assignment differs.

1. Download and extract the original checkpoint archive:

```bash
uv run --package lace python models/lace/scripts/download_vendor_assets.py \
  --output-dir .cache/lace/original \
  --filename model.tar.gz \
  --extract
```

Expected files after extraction:

```text
.cache/lace/original/model/publaynet_best.pt
.cache/lace/original/model/rico25_best.pt
```

2. Record local reference metadata for the available vendor checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output-dir .cache/lace/reference/publaynet \
  --seed 123

CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output-dir .cache/lace/reference/rico25 \
  --seed 123
```

Generated metadata:

```text
.cache/lace/reference/publaynet/metadata.json
.cache/lace/reference/rico25/metadata.json
```

3. Run the parity tests:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace pytest \
  models/lace/tests/vendor_parity/test_lace_parity.py \
  -m vendor_parity -vv -rs
```

Expected result with the public archive is `4 passed, 1 skipped`. The skipped
case is `test_checkpoint_conversion_smoke[rico13-rico13_best.pt]`, because the
public archive does not contain `.cache/lace/original/model/rico13_best.pt`.

4. Convert the available checkpoints:

```bash
rm -rf .cache/lace/converted/lace-publaynet .cache/lace/converted/lace-rico25

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output .cache/lace/converted/lace-publaynet

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output .cache/lace/converted/lace-rico25
```

Each output directory contains Diffusers pipeline files and `README.md`:

```text
.cache/lace/converted/lace-publaynet/README.md
.cache/lace/converted/lace-rico25/README.md
```

5. Load the converted checkpoints with `from_pretrained` and run a short smoke:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python - <<'PY'
from lace import LacePipeline

for name in ["lace-publaynet", "lace-rico25"]:
    pipe = LacePipeline.from_pretrained(f".cache/lace/converted/{name}")
    pipe = pipe.to("cuda")
    out = pipe(batch_size=1, seed=0, num_inference_steps=2)
    in_range = bool((out.bbox >= 0).all() and (out.bbox <= 1).all())
    print(name, tuple(out.bbox.shape), tuple(out.labels.shape), in_range)
PY
```

Expected output shapes are `(1, 25, 4)` for `bbox` and `(1, 25)` for `labels`;
the final boolean should be `True`.

## License

The original LACE implementation is released under the MIT License.

Vendor links:

- Original implementation: https://github.com/puar-playground/LACE
- Checkpoints and datasets: https://huggingface.co/datasets/puar-playground/LACE
- Paper: https://openreview.net/forum?id=kJ0qp9Xdsh

## Citation

```bibtex
@inproceedings{
    chen2024towards,
    title={Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints},
    author={Jian Chen and Ruiyi Zhang and Yufan Zhou and Changyou Chen},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=kJ0qp9Xdsh}
}
```
