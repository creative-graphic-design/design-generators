# Coarse-to-Fine

Coarse-to-Fine is an autoregressive hierarchical layout generator for graphic layouts. This package converts the released RICO and PubLayNet checkpoints from `jzy124/Coarse2Fine` into a Transformers-style `PreTrainedModel` with `save_pretrained` / `from_pretrained` support and the shared `laygen.modeling_outputs.LayoutGenerationOutput` schema.

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/19994

Original implementation: https://github.com/microsoft/LayoutGeneration/tree/main/Coarse-to-Fine

## Install

From the repository root:

```bash
uv sync --package coarse-to-fine
```

Run package-local tests with the workspace member selected:

```bash
uv run --package coarse-to-fine pytest models/coarse-to-fine/tests -m "not vendor_parity and not integration"
```

## Usage

```python
from coarse_to_fine import (
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)

model_id = "creative-graphic-design/coarse-to-fine-rico25"
model = CoarseToFineForLayoutGeneration.from_pretrained(model_id)
processor = CoarseToFineProcessor.from_pretrained(model_id)
pipe = CoarseToFinePipeline(model=model, processor=processor)

out = pipe(batch_size=2, seed=0, return_intermediates=True)
print(out.bbox)    # normalized center xywh, shape (B, S, 4)
print(out.labels)  # dataset-local ids, padding controlled by out.mask
print(out.intermediates["hierarchy"]["group_bbox"])
```

## Checkpoints

| Dataset | Condition | Hub id |
| --- | --- | --- |
| RICO25 | unconditional | `creative-graphic-design/coarse-to-fine-rico25` |
| PubLayNet | unconditional | `creative-graphic-design/coarse-to-fine-publaynet` |

The released checkpoints support unconditional hierarchical generation only. Public v1/v2 condition names are accepted at the API boundary, but unsupported modes such as `label`, `completion`, `text`, `content_image`, `relation`, `hierarchical`, and `retrieval` raise `NotImplementedError`.

## Datasets

- `creative-graphic-design/Rico`: use the `ui-screenshots-and-hierarchies-with-semantic-annotations` config for RICO25 labels and hierarchy bounds. Public labels are zero-based; vendor tensors use one-based labels with SOS/EOS ids.
- `creative-graphic-design/PubLayNet`: COCO-style document boxes are converted to normalized `ltwh`, discretized on the vendor 128-bin grid, and returned publicly as normalized center `xywh`.

Coarse-to-Fine has structured tensor heads for labels, group label histograms, and four coordinate bins. There is no single unified token vocabulary for layout serialization, so this package uses a `ProcessorMixin` processor instead of a `PreTrainedTokenizer`.

## Parity Results

Local fixtures compare the converted Transformers model against the original
Microsoft Coarse-to-Fine implementation with a fixed latent tensor. Vendor
reference tensors are regenerated on demand and are not committed.

| Dataset | Comparison target | Cases | Agreement criterion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| RICO25 | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `3.004e-05` |
| PubLayNet | Strict checkpoint conversion + `from_pretrained` | 1 checkpoint | Missing/unexpected keys fail strict load | Pass |
| PubLayNet | Vendor vs converted hierarchy logits | batch 2, 20 groups/elements | `rtol=5e-5`, `atol=5e-5`; greedy argmax exact | Pass; max abs `7.75e-06` |

Detailed max-abs values for the logits checks were:

| Dataset | Group bbox | Group label histogram | Grouped bbox | Grouped label | Argmax |
| --- | ---: | ---: | ---: | ---: | --- |
| RICO25 | `3.004e-05` | `2.026e-06` | `2.098e-05` | `2.193e-05` | exact |
| PubLayNet | `3.814e-06` | `5.960e-07` | `5.600e-06` | `7.750e-06` | exact |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites:

- Run commands from the repository root.
- Use `uv sync --package coarse-to-fine --extra vendor` once before the first vendor run.
- Initialize the vendor implementation with `git submodule update --init vendor/ms-layout-generation`.
- Keep downloaded weights and generated references under `.cache/coarse-to-fine/`; these files are local artifacts and are not committed.
- Set `CUDA_VISIBLE_DEVICES=3` for the vendor reference/parity run.

1. Download the public checkpoints into `.cache/coarse-to-fine/original`.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/download_original.py \
  --output-dir .cache/coarse-to-fine/original
```

2. Convert both public checkpoints into Transformers format.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --dataset rico25 \
  --output-dir .cache/coarse-to-fine/converted/rico25

uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --dataset publaynet \
  --output-dir .cache/coarse-to-fine/converted/publaynet
```

3. Generate vendor reference tensors.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset rico25 \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/rico25

CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset publaynet \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/publaynet
```

4. Run vendor parity tests against cached references.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine pytest \
  models/coarse-to-fine/tests/vendor_parity \
  -m vendor_parity \
  -q
```

Expected result with the fixtures above:

```text
4 passed
```

5. Smoke-test local `from_pretrained`.

```bash
uv run --package coarse-to-fine python - <<'PY'
from coarse_to_fine import CoarseToFineForLayoutGeneration, CoarseToFineProcessor

for dataset in ("rico25", "publaynet"):
    path = f".cache/coarse-to-fine/converted/{dataset}"
    model = CoarseToFineForLayoutGeneration.from_pretrained(path)
    processor = CoarseToFineProcessor.from_pretrained(path)
    out = model.generate_layout(batch_size=1, seed=0, processor=processor)
    assert out.bbox.shape == (1, 20, 4)
    assert out.labels.shape == (1, 20)
    assert bool(out.mask.any())
    print(model.config.dataset, out.bbox.shape, out.labels.shape)
PY
```

Expected output:

```text
rico25 torch.Size([1, 20, 4]) torch.Size([1, 20])
publaynet torch.Size([1, 20, 4]) torch.Size([1, 20])
```

## Model Cards

Converted output directories include a `README.md` model card when generated by the publishing flow. Hub publishing is deferred to issue #78.

## License

The original Microsoft LayoutGeneration repository is MIT licensed. Converted code in this package is for research conversion and evaluation of the released Coarse-to-Fine checkpoints.

## Citation

```bibtex
@inproceedings{jiang2022coarse,
  title = {Coarse-to-Fine Generative Modeling for Graphic Layouts},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Jichen and Deng, Jingwen and Luo, Jian and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {AAAI},
  year = {2022}
}
```
