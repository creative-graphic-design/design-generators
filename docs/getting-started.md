# Getting Started

Install workspace members from the repository root with `uv`. The root workspace contains shared libraries under `lib/*` and model packages under `models/*`; member-specific commands should select the package so extras and dependency source mappings resolve correctly.

## Install

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --all-packages
```

For a narrower environment, install only the package you want to run.

```bash
uv sync --package layout-dm
uv run --package layout-dm pytest models/layout-dm/tests -m "not vendor_parity and not integration"
```

## Run A Converted Checkpoint

Most weight-backed packages use locally converted checkpoint directories until planned Hub repos are published. Each model package has a `REPRODUCING.md` file with the download, reference generation, parity, conversion, and smoke-test commands that create the local path.

For LayoutDM, run the minimal download and conversion commands from the repository root. See the full LayoutDM [reproducibility guide](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-dm/REPRODUCING.md) for vendor reference generation, parity tests, and smoke tests.

```bash
uv run --package layout-dm python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original

uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir .cache/layout-dm/converted/layoutdm-rico25
```

```bash
uv run --package layout-dm python
```

```python
from layout_dm import LayoutDMPipeline

path = ".cache/layout-dm/converted/layoutdm-rico25"
# After Hub publication: from_pretrained("creative-graphic-design/layoutdm-rico25")
pipe = LayoutDMPipeline.from_pretrained(path)
out = pipe(batch_size=1, seed=0, sampling="deterministic")

print(out.bbox.shape)
print(out.labels.shape)
print(out.mask.shape)
print(out.id2label[0])
```

```text
torch.Size([1, 25, 4])
torch.Size([1, 25])
torch.Size([1, 25])
Text
```

`bbox` uses normalized center `xywh` coordinates in `[0, 1]`, `labels` are dataset-local integer ids, `mask=True` marks valid elements, and `id2label` decodes labels for the active dataset.

## GPU Selection

Vendor parity and heavyweight conversion commands that need CUDA use a placeholder GPU selector. Replace `<gpu-index>` with one visible CUDA device on your machine, such as `0` on a single-GPU host.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm pytest models/layout-dm/tests/vendor_parity
```

## Prompt-Only Packages

LayoutGPT and LayoutPrompter do not convert learned checkpoints. Install their workspace member, configure the provider credentials required by the selected Pydantic AI model, and load prompt configuration or exemplars with `save_pretrained` and `from_pretrained` as documented on each model page.
