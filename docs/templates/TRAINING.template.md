---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Template
---

# <Package> Training

This document records package-local training commands, staged reproduction
evidence, and per-dataset full-run status for `<package>`.

Run commands from the repository root. Keep generated data, logs, checkpoints,
converted local pipelines, and evaluation artifacts under `.cache/<package>/`.

## Install

```bash
uv sync --package <package> --extra training
```

Install the `vendor` extra only when rerunning original-code parity checks.

```bash
uv sync --package <package> --extra training --extra vendor
```

## Data

List every training dataset and the exact source or local artifact layout used by
the package-local and original-code runs.

| Dataset | Source | Config or path |
| --- | --- | --- |
| `<dataset>` | `<dataset id or local path>` | `<config, split, or layout notes>` |

## Configs

Training configs live under `models/<package>/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `<config>.yaml` | `<dataset>` | `<default or deterministic>` | `<purpose>` |

## Scheduler and Recipe Notes

Document optimizer, scheduler cadence, validation cadence, batch-size,
accumulation, initialization, and recipe differences that affect reproduction.
State observed environment constraints as verified setup, not as inherent
package requirements.

## Seed Policy

State the training seed scope and evaluation seed scope for every claimed
dataset. Use labels such as `training-seed n=3` or `evaluation-seed n=3`; do not
describe evaluation-seed evidence as training-seed reproduction.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | `<summary>` |
| S1 | Fixed-batch pre-optimizer trace parity | `<summary>` |
| S2 | One optimizer-step parity | `<summary>` |
| S3 | Short deterministic multi-batch run | `<summary>` |
| S4 | Deterministic loader stream | `<summary>` |
| S5 | Full-run statistical comparison | `<summary>` |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S1 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S2 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S3 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S4 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |
| S5 | `<command>` | `<repo/cache-relative path or project issue/PR URL>` | `<result>` |

## Reproduction Results

Open with a conclusion-first paragraph that states the verdict, covered
datasets, seed scope, and any partial coverage. Every dataset listed in the
package README `Supported Checkpoints` table must appear in this section's
results table.

Allowed `Status` values are:

- `s5-bit-parity`
- `s5-practical-reproduction`
- `recipe-unstable (documented)`
- `not-yet-run (<tracking ref>)`
- `blocked (<reason>)`

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| `<dataset>` | original | `not-yet-run (#<issue>)` | `<seed scope>` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |
| `<dataset>` | package | `not-yet-run (#<issue>)` | `<seed scope>` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |

## Regeneration Metadata

Record non-committed evidence locations, config hashes, seeds, command arguments,
hardware notes, and issue or PR evidence URLs needed to regenerate the results.
Use repository-relative, cache-relative, or project GitHub URLs.

```text
.cache/<package>/training-runs/
.cache/<package>/full-run/
```

## Training Commands

Run the staged parity checks.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package <package> --extra training --extra vendor pytest \
  models/<package>/tests/vendor_parity -m "vendor_parity and training" -rs
```

Train one dataset.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
  uv run --package <package> --extra training \
  traingen fit \
  --config models/<package>/configs/training/<dataset>.yaml \
  --trainer.devices=1
```

Convert a trained checkpoint.

```bash
uv run --package <package> python models/<package>/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/<package>/training-runs/<dataset>/checkpoints/<checkpoint>.ckpt \
  --output-dir .cache/<package>/converted-trained/<dataset>
```

Smoke-test local loading.

```bash
uv run --package <package> python - <<'PY'
from package_name import PackagePipeline

pipe = PackagePipeline.from_pretrained(".cache/<package>/converted-trained/<dataset>")
out = pipe(condition_type="unconditional", num_inference_steps=2)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```
