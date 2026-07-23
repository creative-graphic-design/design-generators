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

Use `models/cgb-dm/configs/training/smoke.yaml` for local configuration smoke
checks. Full 500-epoch PKU/CGL training is intentionally outside the first PR.

## Parity Stages

| Stage | Scope |
| --- | --- |
| S0 | Batch loading and field ordering from the CGB-DM asset layout. |
| S1 | Training-step tensor trace through timestep sampling, masking, and noise addition. |
| S2 | Epsilon prediction and loss comparison against generated reference metadata. |

The first PR wires these stages behind gated parity tests. Full
[PKU PosterLayout](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023) and
CGL asset runs are deferred until the original assets are available locally.
