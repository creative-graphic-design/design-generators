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

Vendor parity fixtures are regenerated on demand and are not committed.

| Checkpoint | Status | Logits max abs | Logits max rel | Decoded hierarchy |
| --- | --- | ---: | ---: | --- |
| `ckpts/rico/checkpoint.pth.tar` | script-ready | pending local GPU run | pending local GPU run | fixed latent + greedy argmax |
| `ckpts/publaynet/checkpoint.pth.tar` | script-ready | pending local GPU run | pending local GPU run | fixed latent + greedy argmax |

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

2. Generate local reference metadata and vendor tensor placeholders.

```bash
for dataset in rico25 publaynet; do
  CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine python models/coarse-to-fine/scripts/export_reference.py \
    --dataset "$dataset" \
    --checkpoint ".cache/coarse-to-fine/original/ckpts/${dataset/rico25/rico}/checkpoint.pth.tar" \
    --seed 0 \
    --output-dir ".cache/coarse-to-fine/reference/${dataset}"
done
```

3. Run vendor parity tests against cached references.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine pytest \
  models/coarse-to-fine/tests/vendor_parity \
  -m vendor_parity \
  -q
```

4. Convert both public checkpoints into Transformers format.

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

5. Smoke-test local `from_pretrained`.

```bash
uv run --package coarse-to-fine python - <<'PY'
from coarse_to_fine import CoarseToFineForLayoutGeneration, CoarseToFineProcessor

for dataset in ("rico25", "publaynet"):
    path = f".cache/coarse-to-fine/converted/{dataset}"
    model = CoarseToFineForLayoutGeneration.from_pretrained(path)
    processor = CoarseToFineProcessor.from_pretrained(path)
    out = model.generate_layout(batch_size=1, seed=0, processor=processor)
    print(model.config.dataset, out.bbox.shape, out.labels.shape)
PY
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
