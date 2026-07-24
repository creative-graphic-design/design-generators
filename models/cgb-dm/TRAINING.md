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
package run is `pku_full_ours_20260724_013039` loaded from
`epoch=499-step=121000.ckpt`. Both use the PKU `pku.yaml` training-evaluation
path (`val/inpaint`, 1,000 samples). The first reference column is the metric
line emitted by the original training process after epoch 500. The standalone
columns reload the final checkpoints and sample seeds 1, 2, and 3 with the same
metric formulas.

| Metric | Reference full-run log | Reference standalone mean +/- std (n=3) | Package mean +/- std (n=3) |
| --- | ---: | ---: | ---: |
| `val` | 1.000000 | 1.000000 +/- 0.000000 | 0.089132 +/- 0.003409 |
| `ove` | 0.002727 | 0.002286 +/- 0.000156 | 0.571271 +/- 0.004957 |
| `undl` | 0.996477 | 0.996406 +/- 0.001368 | NaN +/- NaN |
| `unds` | 0.978788 | 0.972385 +/- 0.000736 | NaN +/- NaN |
| `occ` | 0.127215 | 0.127496 +/- 0.000878 | 0.053226 +/- 0.001040 |
| `rea` | 0.015321 | 0.015695 +/- 0.000349 | 0.006598 +/- 0.000607 |

The package checkpoint does not match the reference S5 distribution. `undl` and
`unds` are undefined because the sampled package layouts contain no underlay
class in the checked seed. For seed 1, package labels are `{0: 14505, 1: 552,
2: 943}` and reference labels are `{0: 12267, 1: 2542, 2: 514, 3: 677}`. The
lower package `occ` and `rea` values are therefore not an improvement signal;
they are coupled with low validity and missing underlay/text-heavy structure.

Re-run the package checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/ours-pku/pku_full_ours_20260724_013039/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-pku-val \
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
