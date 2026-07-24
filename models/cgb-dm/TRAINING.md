# Training CGB-DM

CGB-DM training is driven by the shared class-path
[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
entry point. The model package provides the `LightningModule`,
`LightningDataModule`, and YAML configs; optimizer and scheduler defaults are
declared in those YAML files through LightningCLI class paths.

```bash
uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml
```

Use `models/cgb-dm/configs/training/smoke.yaml` for local and CI configuration
smoke checks. Full 500-epoch PKU/CGL training runs outside regular CI and is
reported as run metadata until the paired full-run statistical comparison is
complete.

## Parity Stages

| Stage | Scope |
| --- | --- |
| S0 | Batch loading and field ordering from the CGB-DM asset layout. |
| S1 | Training-step tensor trace through timestep sampling, masking, and noise addition. |
| S2 | Epsilon prediction and loss comparison against generated reference metadata. |

## S0 Data Order Deviation

The original training loader iterates `os.listdir(train/inpaint)`, so its row
order depends on filesystem state. The package default remains sorted for
deterministic training data access. Parity captures the original loader's
observed filename order into a regenerated `.cache` manifest and replays that
manifest explicitly for vendor-compatible S0 checks. The manifest is
regeneration metadata and is not committed.

The gated PKU check now runs a fixed-batch original-code training step and the
package training step with the same RNG state. S1 compares timestep ids, noise,
condition masks, noised layouts, predicted epsilon, loss, and content-graphic
balance weights. S2 compares clipped gradients, post-Adam parameters, and Adam
state after one update. Floating comparisons use narrow tolerances because the
same CUDA model path produces small roundoff differences across separate
forward/backward executions.

Full [PKU PosterLayout](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023)
and CGL asset runs remain outside regular CI.

## Reproduction Results

S5 compares generated metrics from full PKU training runs. The reference run is
the epoch-500 original-code process `pku_full_vendor_20260723_224914`; the
package run is the discarded diagnostic checkpoint
`pku_full_ours_20260724_013039` loaded from `epoch=499-step=121000.ckpt`. Both
use the PKU `pku.yaml` training-evaluation path (`val/inpaint`, 1,000 samples).
The first reference column is the metric line emitted by the original training
process after epoch 500. The standalone columns reload the final checkpoints and
sample seeds 1, 2, and 3 with the same metric formulas. Package evaluation feeds
raw internal class ids and boxes to the original metric functions; public
pipeline decoding is not used for this table.

| Metric | Reference full-run log | Reference standalone mean +/- std (n=3) | Package mean +/- std (n=3) |
| --- | ---: | ---: | ---: |
| `val` | 1.000000 | 1.000000 +/- 0.000000 | 0.090430 +/- 0.001914 |
| `ove` | 0.002727 | 0.002286 +/- 0.000156 | 0.362796 +/- 0.021026 |
| `undl` | 0.996477 | 0.996406 +/- 0.001368 | 0.015503 +/- 0.001400 |
| `unds` | 0.978788 | 0.972385 +/- 0.000736 | 0.009336 +/- 0.001625 |
| `occ` | 0.127215 | 0.127496 +/- 0.000878 | 0.048323 +/- 0.000940 |
| `rea` | 0.015321 | 0.015695 +/- 0.000349 | 0.006018 +/- 0.000509 |

The package checkpoint does not match the reference S5 distribution and must be
discarded. The primary root cause was architecture drift: the discarded package
model had 22.8M parameters and 129 state-dict keys, while the reference
`LayoutModel` has 47.9M parameters and 229 state-dict keys. The earlier S1/S2
training parity checks injected the reference `LayoutModel` into the package
training wrapper, so they verified the training step order but did not verify
the package `CGBDMTransformerModel` architecture. A direct same-seed/same-input
forward comparison before the fix had mean absolute output difference
`0.516288`.

A secondary full-run data bug also contributed to the invalid checkpoint. The
full-run data module used public encoding for the original PKU CSV labels while
S0-S2 parity used reference encoding. PKU original labels are internal ids where
`0` is padding/invalid and `1..3` are public labels `0..2`; public encoding
therefore wrote padded positions as internal class `3` (`underlay`) during
training. The corrected full-run config now uses reference encoding and the
captured source-order manifest. The corrected model now has 47.9M parameters,
229/229 matching state-dict keys, strict-loads the reference state dict, and
matches the reference forward path bitwise on the parity input. Its restarted
full-run loss also aligns with the reference curve at the start:
`Loss/train` epoch 1 is `0.733680` vs. reference `0.733431`, and epoch 5 is
`0.121871` vs. reference `0.121453`.

The discarded checkpoint is retained only as diagnostic evidence. For seed 1,
its raw internal labels are `{0: 14512, 1: 531, 2: 36, 3: 921}`, far from the
reference distribution `{0: 12267, 1: 2542, 2: 514, 3: 677}`.

| Component | Reference full run | Corrected package run |
| --- | --- | --- |
| Model architecture | `LayoutModel`, 47.9M params, 229 keys | `CGBDMTransformerModel`, 47.9M params, 229 matching keys |
| EMA/shadow weights | Not used for saved/evaluated checkpoints | Not used |
| Checkpoint selection | `Epoch500_cgbdm_weights.pth` after epoch 500 | Lightning `epoch=499-step=121000.ckpt` after 500 epochs |
| Optimizer | Adam, lr `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`, no weight decay | Same |
| LR scheduler | `CosineAnnealingLR(T_max=500)`, stepped once per epoch | Same |
| Gradient clipping | Norm clip `1.0` before optimizer step | Same |
| Training condition | `uncond`, no fixed layout channels | `content_image`, no fixed layout channels |
| Data order | Source `os.listdir` order | Captured source-order manifest |
| PKU label encoding | Internal `0=pad`, `1..3=classes` | Reference encoding, same internal ids |
| Sampling metrics | Raw internal classes and boxes into original metrics | Same |

Re-run the package checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/ours-pku/pku_full_ours_20260724_013039/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-pku-val-raw \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the standalone reference checkpoint comparison:

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
