# Training CGB-DM

CGB-DM training is driven by the shared class-path LightningCLI entry point.
The model package provides the `LightningModule`, `LightningDataModule`, and
YAML configs.

```bash
uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml
```

Use `models/cgb-dm/configs/training/smoke.yaml` for local configuration smoke
checks. Full 500-epoch PKU/CGL training is intentionally outside the first PR.
