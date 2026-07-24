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

| Dataset | Status | Checkpoints | Loss mean | FID | Overlap | IoU | Conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PubLayNet | run | vendor epoch 799 vs ours epoch 799 | vendor `1.8069 +/- 0.0228`; ours `1.8775 +/- 0.0215` | vendor `2.3806 +/- 0.0546`; ours `31.2920 +/- 0.1734` | vendor `0.0241 +/- 0.0003`; ours `0.2130 +/- 0.0019` | vendor `0.0034 +/- 0.0001`; ours `0.0448 +/- 0.0005` | Training loss is close, but generation metrics do not reproduce vendor quality. |
| RICO13 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending full vendor/package training pair. |
| Magazine | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending vendor support and polygon/train-only handling. |

For the PubLayNet training-loss curve, the regenerated final segment from epochs
733-799 has mean absolute train-loss delta `0.4549` versus the vendor log, with
final train losses `0.6560` for vendor and `0.7291` for ours. The first epoch in
that resumed segment with absolute delta greater than `0.5` is epoch 733; there
is no monotonic late-epoch divergence, but generated layouts remain much worse
than the vendor checkpoint under the same evaluation protocol.
