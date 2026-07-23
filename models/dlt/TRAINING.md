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
