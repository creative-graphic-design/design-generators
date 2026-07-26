# Training DLT

DLT training uses the shared class-path-driven LightningCLI entry point. The package does not define `dlt.training.cli`.

## Smoke

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/smoke.yaml
```

## PubLayNet

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/dlt_publaynet.yaml
```

## RICO13

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/dlt_rico13.yaml
```

Magazine remains gated until polygon and train-only handling is amended.

## Reproduction Results

S5 compares a full vendor checkpoint with an independently trained package checkpoint
using the same PubLayNet validation split, `all` conditioning, and evaluation seeds
`42`, `43`, and `44`. Lower is better for FID, overlap, alignment, IoU, and loss.

| Dataset | Status | Checkpoints | Loss mean | FID | Overlap | Alignment | IoU | Conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PubLayNet | S5 near-parity, offsets disclosed | vendor epoch 799 vs ours epoch 799 | vendor `1.8069 +/- 0.0228`; ours `1.8072 +/- 0.0225`; delta `+0.0003` | vendor `2.3806 +/- 0.0546`; ours `1.9500 +/- 0.0379`; delta `-0.4306` | vendor `0.0241 +/- 0.0003`; ours `0.0257 +/- 0.0004`; delta `+0.0015` | vendor `0.0102 +/- 0.0000`; ours `0.0109 +/- 0.0001`; delta `+0.0007` | vendor `0.0034 +/- 0.0001`; ours `0.0033 +/- 0.0001`; delta `-0.0001` | Loss, FID, and IoU are aligned or better for the package checkpoint, while overlap and alignment retain small systematic one-directional offsets. |
| RICO13 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending full vendor/package training pair. |
| Magazine | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending vendor support and polygon/train-only handling. |

The PubLayNet S5 run is not a clean metric-identical pass. The package checkpoint
has better FID (`-0.4306`) and matching validation loss (`+0.0003`) and IoU
(`-0.0001`), but overlap is higher by `+0.00153` (about `+6.3%`, one-directional
and roughly `5` standard deviations at this seed scope) and alignment is higher
by `+0.00071` (about `+6.9%`, one-directional and roughly `10` standard
deviations at this seed scope). Treat the result as near-parity with disclosed
offsets, not as exact training reproduction.

| Seed | Vendor loss | Ours loss | Loss delta | Vendor FID | Ours FID | FID delta | Vendor overlap | Ours overlap | Overlap delta | Vendor align | Ours align | Align delta | Vendor IoU | Ours IoU | IoU delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.8391` | `1.8390` | `-0.0002` | `2.3065` | `1.9388` | `-0.3677` | `0.0245` | `0.0262` | `+0.0018` | `0.0102` | `0.0109` | `+0.0007` | `0.0035` | `0.0034` | `-0.0000` |
| 43 | `1.7927` | `1.7934` | `+0.0006` | `2.4362` | `2.0010` | `-0.4352` | `0.0242` | `0.0252` | `+0.0009` | `0.0102` | `0.0110` | `+0.0008` | `0.0035` | `0.0032` | `-0.0003` |
| 44 | `1.7889` | `1.7893` | `+0.0004` | `2.3993` | `1.9103` | `-0.4889` | `0.0237` | `0.0255` | `+0.0019` | `0.0103` | `0.0109` | `+0.0006` | `0.0034` | `0.0034` | `+0.0001` |

The per-epoch train-loss rows are instantaneous stochastic training-step logs,
not synchronized epoch means. The S5 diagnostic recomputed both final checkpoints
on the same 128 PubLayNet batches, noise, and timesteps and found bit-identical
package-vs-vendor loss definitions (`max_abs_ours_loss_def_delta=0.0`) plus
matching train-loss distributions: vendor mean `0.8764726606`, package mean
`0.8752792128`, and mean delta `-0.0011934477`. The remaining curve delta is a
logging-sample artifact; see
`.cache/dlt/full-run/s5-evaluation-lr-step/train_loss_diagnostic.{json,md}` and
`.cache/dlt/full-run/s5-evaluation-lr-step/results-gpu1-rerun.json`.
