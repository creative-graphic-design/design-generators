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

Convert a local original checkpoint and write the Hub model card into the
`save_pretrained` output:

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
max-25 layout sequences. Dataset ids used in generated model cards:

| Dataset | Dataset id |
| --- | --- |
| PubLayNet | `creative-graphic-design/publaynet` |
| Rico13 | `creative-graphic-design/rico13` |
| Rico25 | `creative-graphic-design/rico25` |

## Vendor Parity

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
