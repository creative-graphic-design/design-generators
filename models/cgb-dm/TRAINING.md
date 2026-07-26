# Training CGB-DM

CGB-DM training is reproducible with the package implementation when it uses the
reference architecture, reference dataset encoding, and raw-internal S5
evaluation protocol. The PKU PosterLayout package checkpoint trained for 500
epochs matches the reference S5 distribution: `val` is `1.000000 +/- 0.000000`
for both runs, and package underlay saliency is `0.991428 +/- 0.001467` against
reference `0.972385 +/- 0.000736` over seeds 1, 2, and 3.

The reproducible package run uses:

- `CGBDMTransformerModel` with 47.9M parameters, matching the reference
  `LayoutModel` architecture.
- PKU reference encoding, where the internal layout vocabulary is
  `0=padding/invalid` and `1..3=layout classes`.
- The captured source-order manifest for the original PKU training split.
- Adam with `lr=1e-4`, betas `(0.9, 0.999)`, `eps=1e-8`, no weight decay,
  `CosineAnnealingLR(T_max=500)`, and gradient clipping at `1.0`.
- S5 evaluation on the PKU validation split with 1,000 samples per seed, raw
  internal `argmax` class ids, and raw generated boxes passed to the original
  metric formulas.

## Package Training

Generate the PKU source-order manifest before starting a full training run:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm \
  python models/cgb-dm/scripts/generate_reference_outputs.py \
  --dataset pku_posterlayout \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --manifest-output .cache/cgb-dm/reference/pku_posterlayout_train_manifest.json
```

Use `models/cgb-dm/configs/training/smoke.yaml` for local configuration smoke
checks.

Train the package model with the reference-compatible PKU config:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml \
  --trainer.default_root_dir .cache/cgb-dm/full-run/ours-pku
```

The PKU config sets the reference architecture and data path:

```text
dim_model=512
n_head=8
num_layers=4
feature_dim=1024
original_encoding=reference
source_order_manifest=.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json
```

The final package checkpoint used for the PKU S5 comparison is the epoch-500
Lightning checkpoint, such as:

```text
.cache/cgb-dm/full-run/ours-pku-fixed/pku_full_ours_archfixed_20260724_122952/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt
```

## Reference Training

Run the reference implementation from the vendored `layout-dit` checkout with
the same extracted PKU split. The command below patches runtime paths without
editing committed vendor files:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import yaml

repo_root = Path.cwd()
data_root = repo_root / ".cache/cgb-dm/datasets/pku/split"
run_root = repo_root / ".cache/cgb-dm/full-run/vendor-pku"
run_id = "pku_full_vendor"
vendor_root = repo_root / "vendor/layout-dit"

checkpoint_root = run_root / "checkpoints" / run_id
image_order_root = run_root / "image_name_order"
tensorboard_root = run_root / "tensorboard" / run_id
checkpoint_root.mkdir(parents=True, exist_ok=True)
image_order_root.mkdir(parents=True, exist_ok=True)
tensorboard_root.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(vendor_root))
os.chdir(vendor_root)

import scripts.train as train_script
import scripts.train_util as train_util
from utils.util import Config, process_paths

with (vendor_root / "configs/pku.yaml").open("r", encoding="utf-8") as handle:
    raw_config = yaml.safe_load(handle)

raw_config["paths"]["base"] = str(data_root)
raw_config["base_check_dir"] = str(checkpoint_root)
raw_config["imgname_order_dir"] = str(image_order_root)
raw_config["datetime"] = run_id
config = Config(process_paths(raw_config))

original_summary_writer = train_util.SummaryWriter

def summary_writer_with_run_dir(*args, **kwargs):
    kwargs.setdefault("log_dir", str(tensorboard_root))
    return original_summary_writer(*args, **kwargs)

train_util.SummaryWriter = summary_writer_with_run_dir
train_script.load_config = lambda _path: config
train_script.main(SimpleNamespace(gpuid=0, dataset="pku", task="uncond"))
PY
```

The reference checkpoint used for the PKU S5 comparison is the epoch-500
checkpoint:

```text
.cache/cgb-dm/full-run/vendor-pku/checkpoints/pku_full_vendor_20260723_224914/Epoch500_cgbdm_weights.pth
```

## S5 Evaluation

Re-run the package checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/ours-pku-fixed/pku_full_ours_archfixed_20260724_122952/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-pku-val \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the reference checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend reference \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/vendor-pku/checkpoints/pku_full_vendor_20260723_224914/Epoch500_cgbdm_weights.pth \
  --output-dir .cache/cgb-dm/full-run/s5-eval-reference-pku-val \
  --gpu 0 \
  --seeds 1 2 3
```

## PKU S5 Results

Both evaluations use the PKU `pku.yaml` validation path (`val/inpaint`) with
1,000 samples per seed. The package and reference runs are statistically
equivalent under this S5 protocol.

| Metric | Reference mean +/- std (n=3) | Package mean +/- std (n=3) |
| --- | ---: | ---: |
| `val` | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 |
| `ove` | 0.002286 +/- 0.000156 | 0.003293 +/- 0.000801 |
| `undl` | 0.996406 +/- 0.001368 | 0.999345 +/- 0.000284 |
| `unds` | 0.972385 +/- 0.000736 | 0.991428 +/- 0.001467 |
| `occ` | 0.127496 +/- 0.000878 | 0.116661 +/- 0.000648 |
| `rea` | 0.015695 +/- 0.000349 | 0.014180 +/- 0.000295 |

The reference full training log emitted `val=1.000000`, `ove=0.002727`,
`undl=0.996477`, `unds=0.978788`, `occ=0.127215`, and `rea=0.015321` after
epoch 500. The standalone reference and package S5 runs above reload final
checkpoints and resample seeds 1, 2, and 3 with the same metric formulas.

## CGL Status

CGL uses the same reference architecture and raw-internal evaluation protocol.
Both evaluations use the CGL `cgl.yaml` validation path (`val/inpaint`) with
6,055 samples per seed. The package and reference runs are statistically
equivalent under this S5 protocol.

Re-run the package checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --dataset cgl \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/cgl/split \
  --checkpoint .cache/cgb-dm/full-run/ours-cgl/cgl_full_ours_archfixed_20260725_012723/lightning_logs/version_0/checkpoints/epoch=499-step=189500.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-cgl-val \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the reference checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --dataset cgl \
  --backend reference \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/cgl/split \
  --checkpoint .cache/cgb-dm/full-run/vendor-cgl/cgl_full_vendor_20260725_012722/checkpoints/cgl_full_vendor_20260725_012722/Epoch500_cgbdm_weights.pth \
  --output-dir .cache/cgb-dm/full-run/s5-eval-vendor-cgl-val \
  --gpu 0 \
  --seeds 1 2 3
```

## CGL S5 Results

| Metric | Reference mean +/- std (n=3) | Package mean +/- std (n=3) | Package - reference |
| --- | ---: | ---: | ---: |
| `val` | 0.999097 +/- 0.000109 | 0.999213 +/- 0.000044 | +0.000115 |
| `ove` | 0.001795 +/- 0.000044 | 0.001790 +/- 0.000203 | -0.000005 |
| `undl` | 0.997452 +/- 0.000849 | 0.996399 +/- 0.001292 | -0.001052 |
| `unds` | 0.983453 +/- 0.001646 | 0.987553 +/- 0.002680 | +0.004100 |
| `occ` | 0.115873 +/- 0.000318 | 0.116357 +/- 0.000279 | +0.000484 |
| `rea` | 0.005768 +/- 0.000118 | 0.005971 +/- 0.000115 | +0.000203 |

The reference full training log emitted `val=0.998943`, `ove=0.002324`,
`undl=0.996198`, `unds=0.982091`, `occ=0.115683`, and `rea=0.005327` after
epoch 500. The standalone reference and package S5 runs above reload final
checkpoints and resample seeds 1, 2, and 3 with the same metric formulas.
